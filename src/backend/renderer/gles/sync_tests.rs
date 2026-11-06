//! These tests require a real EGL implementation. Without `SMITHAY_TEST_EGL_DEVICE`
//! they use the platform's surfaceless display, which libglvnd may route to a non-Mesa
//! driver; set `SMITHAY_TEST_EXPECT_RENDERER` to a substring of the expected
//! `GL_RENDERER` to fail instead of running on the wrong driver.
//! Run explicitly with `--ignored --test-threads=1`; repeat with
//! `MESA_GLES_VERSION_OVERRIDE=2.0` to exercise the GLES2 completion path, skipping
//! the `cross_thread_readback` and `cross_thread_blit` tests, which need GLES 3.
//!
//! The same-thread tests switch contexts with `eglMakeCurrent`, which flushes the
//! previously current context and hides missing synchronization. The `cross_thread`
//! tests keep the producer current while a consumer on another thread accesses the
//! texture. They detect missing publication on hardware drivers; llvmpipe completes
//! uploads on the CPU and cannot.

use super::*;
use crate::backend::egl::{
    context::{GlAttributes, PixelFormatRequirements},
    native::{EGLNativeDisplay, EGLPlatform, EGLSurfacelessDisplay},
    EGLDevice, EGLDisplay,
};

struct PbufferDisplay<T>(T);

impl<T: EGLNativeDisplay> EGLNativeDisplay for PbufferDisplay<T> {
    fn supported_platforms(&self) -> Vec<EGLPlatform<'_>> {
        self.0.supported_platforms()
    }

    fn surface_type(&self) -> i32 {
        ffi_egl::PBUFFER_BIT as i32
    }
}

fn display() -> EGLDisplay {
    if let Ok(index) = std::env::var("SMITHAY_TEST_EGL_DEVICE") {
        let device = EGLDevice::enumerate()
            .unwrap()
            .nth(index.parse().unwrap())
            .expect("EGL device index");
        unsafe { EGLDisplay::new(PbufferDisplay(device)).expect("device EGL display") }
    } else {
        unsafe { EGLDisplay::new(PbufferDisplay(EGLSurfacelessDisplay)).expect("surfaceless EGL display") }
    }
}

fn renderer(display: &EGLDisplay, fencing: bool) -> GlesRenderer {
    let renderer = with_context(EGLContext::new(display).unwrap(), fencing);
    if let Ok(expected) = std::env::var("SMITHAY_TEST_EXPECT_RENDERER") {
        let name = unsafe {
            renderer.egl.make_current().unwrap();
            let name = std::ffi::CStr::from_ptr(renderer.gl.GetString(ffi::RENDERER) as *const _)
                .to_string_lossy()
                .into_owned();
            // GlesRenderer::new leaves its context unbound; keep that state.
            renderer.egl.unbind().unwrap();
            name
        };
        assert!(
            name.contains(&expected),
            "GL_RENDERER {name:?} does not contain {expected:?}"
        );
    }
    renderer
}

fn with_context(context: EGLContext, fencing: bool) -> GlesRenderer {
    let mut renderer = unsafe { GlesRenderer::new(context).unwrap() };
    if !fencing {
        renderer.capabilities.retain(|cap| *cap != Capability::Fencing);
    }
    renderer
}

fn shared(renderer: &GlesRenderer, fencing: bool) -> GlesRenderer {
    with_context(
        EGLContext::new_shared(renderer.egl.display(), &renderer.egl).unwrap(),
        fencing,
    )
}

fn pixel(renderer: &mut GlesRenderer, texture: &GlesTexture) -> [u8; 4] {
    let target = renderer.bind_texture(texture).unwrap();
    target.0.make_current(&renderer.gl, &renderer.egl).unwrap();
    let mut pixel = [0; 4];
    unsafe {
        renderer.gl.ReadPixels(
            0,
            0,
            1,
            1,
            ffi::RGBA,
            ffi::UNSIGNED_BYTE,
            pixel.as_mut_ptr().cast(),
        );
        assert_eq!(renderer.gl.GetError(), ffi::NO_ERROR);
    }
    pixel
}

fn draw(renderer: &mut GlesRenderer, source: &GlesTexture, target: &GlesTexture) -> SyncPoint {
    let mut target = renderer.bind_texture(target).unwrap();
    let mut frame = renderer
        .render(&mut target, (1, 1).into(), Transform::Normal)
        .unwrap();
    Frame::render_texture_from_to(
        &mut frame,
        source,
        Rectangle::from_size((1.0, 1.0).into()),
        Rectangle::from_size((1, 1).into()),
        &[Rectangle::from_size((1, 1).into())],
        &[],
        Transform::Normal,
        1.0,
    )
    .unwrap();
    frame.finish().unwrap()
}

#[test]
#[ignore = "requires surfaceless EGL and a real GLES driver"]
fn late_sharing_and_shared_readers_preserve_pixels() {
    for fencing in [true, false] {
        let display = display();
        let mut producer = renderer(&display, fencing);
        let texture = producer
            .import_memory(&[255, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
            .unwrap();
        producer
            .update_memory(&texture, &[0, 255, 0, 255], Rectangle::from_size((1, 1).into()))
            .unwrap();
        // The upload has no texture fence: creating the first sibling must finish it.
        let mut first = shared(&producer, fencing);
        let mut second = shared(&producer, fencing);
        let first_target: GlesTexture = first.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
        let second_target: GlesTexture = second.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
        let first_done = draw(&mut first, &texture, &first_target);
        let second_done = draw(&mut second, &texture, &second_target);
        // Do not wait for either frame: updating must wait for both shared readers.
        producer
            .update_memory(&texture, &[0, 0, 255, 255], Rectangle::from_size((1, 1).into()))
            .unwrap();
        assert_eq!(pixel(&mut first, &first_target), [0, 255, 0, 255]);
        assert_eq!(pixel(&mut second, &second_target), [0, 255, 0, 255]);
        first_done.wait().unwrap();
        second_done.wait().unwrap();
        assert_eq!(pixel(&mut producer, &texture), [0, 0, 255, 255]);
        drop(first);
        drop(second);
        assert!(
            producer.egl.is_shared(),
            "sharing policy must not depend on live Arc count"
        );
    }
}

#[test]
#[ignore = "requires surfaceless EGL and a real GLES driver"]
fn render_to_texture_publishes_to_shared_consumer() {
    for fencing in [true, false] {
        let display = display();
        let mut producer = renderer(&display, fencing);
        let mut consumer = shared(&producer, fencing);
        let texture: GlesTexture = producer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
        let mut target = producer.bind_texture(&texture).unwrap();
        let mut frame = producer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        frame
            .clear(
                [1.0, 0.0, 0.0, 1.0].into(),
                &[Rectangle::from_size((1, 1).into())],
            )
            .unwrap();
        let completion = frame.finish().unwrap();
        drop(target);
        // No completion.wait(): the texture's own publication must suffice.
        assert_eq!(pixel(&mut consumer, &texture), [255, 0, 0, 255]);
        completion.wait().unwrap();
    }
}

#[test]
#[ignore = "requires surfaceless EGL, pbuffers, and a real GLES driver"]
fn first_share_restores_current_context_and_distinct_surfaces() {
    let display = display();
    let source = EGLContext::new(&display).unwrap();
    let mut requirements = PixelFormatRequirements::_8_bit();
    requirements.hardware_accelerated = None;
    let previous = EGLContext::new_with_config(
        &display,
        GlAttributes {
            version: (2, 0),
            profile: None,
            debug: false,
            vsync: false,
        },
        requirements,
    )
    .unwrap();
    unsafe {
        let display_ptr = **display.get_display_handle();
        let attributes = [
            ffi_egl::WIDTH as i32,
            1,
            ffi_egl::HEIGHT as i32,
            1,
            ffi_egl::NONE as i32,
        ];
        let draw = ffi_egl::CreatePbufferSurface(display_ptr, previous.config_id(), attributes.as_ptr());
        let read = ffi_egl::CreatePbufferSurface(display_ptr, previous.config_id(), attributes.as_ptr());
        assert_ne!(draw, ffi_egl::NO_SURFACE);
        assert_ne!(read, ffi_egl::NO_SURFACE);
        assert_eq!(
            ffi_egl::MakeCurrent(display_ptr, draw, read, previous.get_context_handle()),
            ffi_egl::TRUE
        );
        let sibling = EGLContext::new_shared(&display, &source).unwrap();
        assert_eq!(ffi_egl::GetCurrentDisplay(), display_ptr);
        assert_eq!(ffi_egl::GetCurrentContext(), previous.get_context_handle());
        assert_eq!(ffi_egl::GetCurrentSurface(ffi_egl::DRAW as i32), draw);
        assert_eq!(ffi_egl::GetCurrentSurface(ffi_egl::READ as i32), read);
        previous.unbind().unwrap();
        ffi_egl::DestroySurface(display_ptr, draw);
        ffi_egl::DestroySurface(display_ptr, read);
        drop(sibling);
        assert!(source.is_shared());

        let source = EGLContext::new(&display).unwrap();
        let _sibling = EGLContext::new_shared(&display, &source).unwrap();
        assert_eq!(ffi_egl::GetCurrentContext(), ffi_egl::NO_CONTEXT);

        let source = EGLContext::new(&display).unwrap();
        source.make_current().unwrap();
        let _sibling = EGLContext::new_shared(&display, &source).unwrap();
        assert_eq!(ffi_egl::GetCurrentContext(), source.get_context_handle());
        source.unbind().unwrap();
    }
}

#[test]
#[ignore = "requires surfaceless EGL and a real GLES driver"]
fn late_sharing_completes_previously_unfenced_reads() {
    let display = display();
    let mut producer = renderer(&display, true);
    let texture = producer
        .import_memory(&[255, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
        .unwrap();
    let target: GlesTexture = producer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    let completion = draw(&mut producer, &texture, &target);
    let mut consumer = shared(&producer, true);
    consumer
        .update_memory(&texture, &[0, 0, 255, 255], Rectangle::from_size((1, 1).into()))
        .unwrap();
    assert_eq!(pixel(&mut producer, &target), [255, 0, 0, 255]);
    assert_eq!(pixel(&mut consumer, &texture), [0, 0, 255, 255]);
    completion.wait().unwrap();
}

#[test]
#[ignore = "requires surfaceless EGL with OpenGL client API support"]
fn first_share_rejects_wrong_client_api_without_changing_it() {
    let display = display();
    let source = EGLContext::new(&display).unwrap();
    unsafe {
        assert_eq!(ffi_egl::BindAPI(ffi_egl::OPENGL_API), ffi_egl::TRUE);
        let result = EGLContext::new_shared(&display, &source);
        let api = ffi_egl::QueryAPI();
        // Restore even if an assertion fails, so subsequent driver tests stay isolated.
        assert_eq!(ffi_egl::BindAPI(ffi_egl::OPENGL_ES_API), ffi_egl::TRUE);
        assert!(result.is_err());
        assert_eq!(api, ffi_egl::OPENGL_API);
        assert!(!source.is_shared());
    }
}

const CROSS_THREAD_SIZE: i32 = 512;
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

fn cross_thread_size() -> Size<i32, BufferCoord> {
    (CROSS_THREAD_SIZE, CROSS_THREAD_SIZE).into()
}

fn solid(color: [u8; 4]) -> Vec<u8> {
    color.repeat((CROSS_THREAD_SIZE * CROSS_THREAD_SIZE) as usize)
}

// Samples the texture from a renderer on another thread, while the caller's renderer
// stays current on its own thread and so is never flushed implicitly.
fn sample_on_other_thread(context: EGLContext, texture: &GlesTexture, fencing: bool) -> [u8; 4] {
    let texture = texture.clone();
    std::thread::spawn(move || {
        let mut consumer = with_context(context, fencing);
        let target: GlesTexture = consumer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
        let _done = draw(&mut consumer, &texture, &target);
        pixel(&mut consumer, &target)
    })
    .join()
    .unwrap()
}

#[test]
#[ignore = "requires surfaceless EGL and a real GLES driver"]
fn cross_thread_first_share_completes_unflushed_upload() {
    for fencing in [true, false] {
        let display = display();
        let mut producer = renderer(&display, fencing);
        let texture = producer
            .import_memory(&solid(RED), Fourcc::Abgr8888, cross_thread_size(), false)
            .unwrap();
        producer
            .update_memory(&texture, &solid(GREEN), Rectangle::from_size(cross_thread_size()))
            .unwrap();
        // Never shared so far: the upload has no texture fence and was not flushed.
        assert!(producer.egl.is_current());
        let context = EGLContext::new_shared(producer.egl.display(), &producer.egl).unwrap();
        assert_eq!(
            sample_on_other_thread(context, &texture, fencing),
            GREEN,
            "fencing: {fencing}"
        );
    }
}

#[test]
#[ignore = "requires surfaceless EGL and a real GLES driver"]
fn cross_thread_shared_upload_is_published() {
    for fencing in [true, false] {
        let display = display();
        let mut producer = renderer(&display, fencing);
        let context = EGLContext::new_shared(producer.egl.display(), &producer.egl).unwrap();
        let texture = producer
            .import_memory(&solid(RED), Fourcc::Abgr8888, cross_thread_size(), false)
            .unwrap();
        producer
            .update_memory(&texture, &solid(GREEN), Rectangle::from_size(cross_thread_size()))
            .unwrap();
        assert!(producer.egl.is_current());
        assert_eq!(
            sample_on_other_thread(context, &texture, fencing),
            GREEN,
            "fencing: {fencing}"
        );
    }
}

// Lets a consumer thread queue a read of a green texture, overwrites the texture with
// blue on this thread, and only then lets the consumer resolve what it read. The
// overwrite is ordered after the read only through the read's publication.
fn read_then_overwrite<T: Send + 'static>(
    fencing: bool,
    read: impl FnOnce(&mut GlesRenderer, &GlesTexture) -> T + Send + 'static,
    resolve: impl FnOnce(&mut GlesRenderer, T) -> bool + Send + 'static,
) -> bool {
    let display = display();
    let mut producer = renderer(&display, fencing);
    let context = EGLContext::new_shared(producer.egl.display(), &producer.egl).unwrap();
    let texture = producer
        .import_memory(&solid(GREEN), Fourcc::Abgr8888, cross_thread_size(), false)
        .unwrap();
    let (read_tx, read_rx) = std::sync::mpsc::channel();
    let (overwritten_tx, overwritten_rx) = std::sync::mpsc::channel();
    let shared_texture = texture.clone();
    let consumer = std::thread::spawn(move || {
        let mut consumer = with_context(context, fencing);
        assert!(
            consumer.capabilities().contains(&Capability::Blit),
            "requires GLES 3; skip this test under MESA_GLES_VERSION_OVERRIDE=2.0"
        );
        let pending = read(&mut consumer, &shared_texture);
        read_tx.send(()).unwrap();
        overwritten_rx.recv().unwrap();
        resolve(&mut consumer, pending)
    });
    if read_rx.recv().is_ok() {
        producer
            .update_memory(&texture, &solid(BLUE), Rectangle::from_size(cross_thread_size()))
            .unwrap();
        overwritten_tx.send(()).unwrap();
    }
    consumer.join().unwrap()
}

#[test]
#[ignore = "requires surfaceless EGL and a GLES 3 driver"]
fn cross_thread_readback_completes_before_overwrite() {
    for fencing in [true, false] {
        let observed_green = read_then_overwrite(
            fencing,
            |consumer, texture| {
                consumer
                    .copy_texture(
                        texture,
                        Rectangle::from_size(cross_thread_size()),
                        Fourcc::Abgr8888,
                    )
                    .unwrap()
            },
            |consumer, mapping| {
                let pixels = consumer.map_texture(&mapping).unwrap();
                pixels.chunks_exact(4).all(|pixel| pixel == GREEN)
            },
        );
        assert!(
            observed_green,
            "readback observed the overwrite (fencing: {fencing})"
        );
    }
}

#[test]
#[ignore = "requires surfaceless EGL and a GLES 3 driver"]
fn cross_thread_blit_source_completes_before_overwrite() {
    for fencing in [true, false] {
        let observed_green = read_then_overwrite(
            fencing,
            |consumer, texture| {
                let copy: GlesTexture = consumer
                    .create_buffer(Fourcc::Abgr8888, cross_thread_size())
                    .unwrap();
                {
                    let source = consumer.bind_texture(texture).unwrap();
                    let mut target = consumer.bind_texture(&copy).unwrap();
                    let rect = Rectangle::from_size((CROSS_THREAD_SIZE, CROSS_THREAD_SIZE).into());
                    consumer
                        .blit(&source, &mut target, rect, rect, TextureFilter::Nearest)
                        .unwrap();
                }
                copy
            },
            |consumer, copy| pixel(consumer, &copy) == GREEN,
        );
        assert!(observed_green, "blit observed the overwrite (fencing: {fencing})");
    }
}
