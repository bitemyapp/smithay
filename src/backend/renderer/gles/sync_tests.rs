//! These tests require a real EGL implementation (Mesa surfaceless works).
//! Run explicitly with `--ignored --test-threads=1`; repeat with
//! `MESA_GLES_VERSION_OVERRIDE=2.0` to exercise the GLES2 completion path.

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
    let context = EGLContext::new(display).unwrap();
    let mut renderer = unsafe { GlesRenderer::new(context).unwrap() };
    if !fencing {
        renderer.capabilities.retain(|cap| *cap != Capability::Fencing);
    }
    renderer
}

fn shared(renderer: &GlesRenderer, fencing: bool) -> GlesRenderer {
    let context = EGLContext::new_shared(renderer.egl.display(), &renderer.egl).unwrap();
    let mut renderer = unsafe { GlesRenderer::new(context).unwrap() };
    if !fencing {
        renderer.capabilities.retain(|cap| *cap != Capability::Fencing);
    }
    renderer
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
