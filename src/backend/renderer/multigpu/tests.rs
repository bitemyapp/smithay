//! A deterministic two-queue renderer: commands read shared pixels only when their
//! queue executes. Draining the producer first exposes missing read-before-write
//! dependencies without relying on GPU timing. These tests exercise MultiRenderer
//! and MultiFrame, rather than a second implementation of their fence bookkeeping.
//! Queue-renderer integration tests need two DRM identities because DrmNode's
//! constructor validates sysfs, but do not execute GPU commands. They are ignored
//! by default; set SMITHAY_TEST_RENDER_NODE/SMITHAY_TEST_TARGET_NODE or allow
//! discovery of two /dev/dri/renderD* nodes, then run with --ignored. The
//! copy_completion_* tests require neither DRM identities nor hardware.
use super::*;
use crate::backend::allocator::dmabuf::DmabufFlags;
use std::{collections::VecDeque, fs::File, os::fd::OwnedFd};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("test renderer failure: {self:?}")]
enum Failure {
    Injected,
    Interrupted,
    Unsupported,
}
impl From<Failure> for SwapBuffersError {
    fn from(error: Failure) -> Self {
        Self::TemporaryFailure(Box::new(error))
    }
}

type Command = Box<dyn FnOnce() + Send>;
#[derive(Default)]
struct Queue {
    commands: Mutex<VecDeque<Command>>,
    submitted: std::sync::atomic::AtomicUsize,
    completed: std::sync::atomic::AtomicUsize,
}
use std::sync::atomic::Ordering::SeqCst;
impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue")
            .field("submitted", &self.submitted.load(SeqCst))
            .field("completed", &self.completed.load(SeqCst))
            .finish()
    }
}
impl Queue {
    fn submit(&self, command: impl FnOnce() + Send + 'static) {
        self.commands.lock().unwrap().push_back(Box::new(command));
        self.submitted.fetch_add(1, SeqCst);
    }
    fn drain_to(&self, serial: usize) {
        while self.completed.load(SeqCst) < serial {
            let command = self
                .commands
                .lock()
                .unwrap()
                .pop_front()
                .expect("cyclic GPU wait");
            command();
            self.completed.fetch_add(1, SeqCst);
        }
    }
    fn drain(&self) {
        self.drain_to(self.submitted.load(SeqCst));
    }
    fn fence(self: &Arc<Self>) -> SyncPoint {
        QueueFence {
            queue: self.clone(),
            serial: self.submitted.load(SeqCst),
        }
        .into()
    }
    fn wait(&self, sync: &SyncPoint) {
        let sync = sync.clone();
        self.submit(move || sync.wait().unwrap());
    }
}
#[derive(Debug)]
struct QueueFence {
    queue: Arc<Queue>,
    serial: usize,
}
impl sync::Fence for QueueFence {
    fn is_signaled(&self) -> bool {
        self.queue.completed.load(SeqCst) >= self.serial
    }
    fn wait(&self) -> Result<(), sync::Interrupted> {
        self.queue.drain_to(self.serial);
        Ok(())
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

#[derive(Debug, Clone)]
struct Image {
    size: Size<i32, BufferCoords>,
    format: Fourcc,
    pixels: Arc<Mutex<Vec<u8>>>,
}
impl Image {
    fn new(size: Size<i32, BufferCoords>, format: Fourcc) -> Self {
        Self {
            size,
            format,
            pixels: Arc::new(Mutex::new(vec![0; (size.w * size.h * 4) as usize])),
        }
    }
    fn pixel(&self, x: usize) -> [u8; 4] {
        self.pixels.lock().unwrap()[x * 4..x * 4 + 4].try_into().unwrap()
    }
}
impl Texture for Image {
    fn width(&self) -> u32 {
        self.size.w as u32
    }
    fn height(&self) -> u32 {
        self.size.h as u32
    }
    fn format(&self) -> Option<Fourcc> {
        Some(self.format)
    }
}
#[derive(Debug)]
struct Mapping {
    image: Image,
    data: Vec<u8>,
}
impl Texture for Mapping {
    fn width(&self) -> u32 {
        self.image.width()
    }
    fn height(&self) -> u32 {
        self.image.height()
    }
    fn format(&self) -> Option<Fourcc> {
        Some(self.image.format)
    }
}
impl TextureMapping for Mapping {
    fn flipped(&self) -> bool {
        false
    }
}

type Images = Arc<Mutex<HashMap<Dmabuf, Image>>>;
#[derive(Debug, Default)]
struct Faults {
    wait: Option<Failure>,
    draw: bool,
    finish: bool,
    cleanup: bool,
    binds: usize,
    waits: usize,
}
#[derive(Debug)]
struct QueuedRenderer {
    queue: Arc<Queue>,
    images: Images,
    faults: Arc<Mutex<Faults>>,
    context: ContextId<Image>,
    flags: DebugFlags,
    import_dma: bool,
}
impl RendererSuper for QueuedRenderer {
    type Error = Failure;
    type TextureId = Image;
    type Framebuffer<'buffer> = Image;
    type Frame<'frame, 'buffer>
        = QueuedFrame
    where
        'buffer: 'frame;
}
impl Renderer for QueuedRenderer {
    fn context_id(&self) -> ContextId<Image> {
        self.context.clone()
    }
    fn downscale_filter(&mut self, _: TextureFilter) -> Result<(), Failure> {
        Ok(())
    }
    fn upscale_filter(&mut self, _: TextureFilter) -> Result<(), Failure> {
        Ok(())
    }
    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.flags = flags;
    }
    fn debug_flags(&self) -> DebugFlags {
        self.flags
    }
    fn render<'frame, 'buffer>(
        &'frame mut self,
        target: &'frame mut Image,
        _: Size<i32, Physical>,
        transform: Transform,
    ) -> Result<QueuedFrame, Failure>
    where
        'buffer: 'frame,
    {
        Ok(QueuedFrame {
            target: target.clone(),
            queue: self.queue.clone(),
            faults: self.faults.clone(),
            context: self.context.clone(),
            transform,
        })
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Failure> {
        let mut faults = self.faults.lock().unwrap();
        faults.waits += 1;
        if let Some(error) = faults.wait {
            return Err(error);
        }
        self.queue.wait(sync);
        Ok(())
    }
    fn cleanup_texture_cache(&mut self) -> Result<(), Failure> {
        if self.faults.lock().unwrap().cleanup {
            Err(Failure::Injected)
        } else {
            Ok(())
        }
    }
}
impl Bind<Image> for QueuedRenderer {
    fn bind(&mut self, target: &mut Image) -> Result<Image, Failure> {
        Ok(target.clone())
    }
}
impl Bind<Dmabuf> for QueuedRenderer {
    fn bind(&mut self, target: &mut Dmabuf) -> Result<Image, Failure> {
        self.faults.lock().unwrap().binds += 1;
        self.images
            .lock()
            .unwrap()
            .get(target)
            .cloned()
            .ok_or(Failure::Unsupported)
    }
    fn supported_formats(&self) -> Option<FormatSet> {
        Some(formats())
    }
}
fn formats() -> FormatSet {
    [Fourcc::Abgr8888, Fourcc::Argb8888]
        .into_iter()
        .map(|code| Format {
            code,
            modifier: Modifier::Linear,
        })
        .collect()
}
impl ImportDma for QueuedRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        formats()
    }
    fn import_dmabuf(
        &mut self,
        buffer: &Dmabuf,
        _: Option<&[Rectangle<i32, BufferCoords>]>,
    ) -> Result<Image, Failure> {
        if !self.import_dma {
            return Err(Failure::Unsupported);
        }
        self.images
            .lock()
            .unwrap()
            .get(buffer)
            .cloned()
            .ok_or(Failure::Unsupported)
    }
}
impl ImportMem for QueuedRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoords>,
        flipped: bool,
    ) -> Result<Image, Failure> {
        if flipped || data.len() != (size.w * size.h * 4) as usize {
            return Err(Failure::Unsupported);
        }
        let image = Image::new(size, format);
        image.pixels.lock().unwrap().copy_from_slice(data);
        Ok(image)
    }
    fn update_memory(
        &mut self,
        texture: &Image,
        data: &[u8],
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Failure> {
        let mut pixels = texture.pixels.lock().unwrap();
        for y in 0..region.size.h {
            for x in 0..region.size.w {
                let dst = ((region.loc.y + y) * texture.size.w + region.loc.x + x) as usize * 4;
                let src = (y * region.size.w + x) as usize * 4;
                pixels[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
            }
        }
        Ok(())
    }
    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new([Fourcc::Abgr8888, Fourcc::Argb8888].into_iter())
    }
}
impl ExportMem for QueuedRenderer {
    type TextureMapping = Mapping;
    fn copy_framebuffer(
        &mut self,
        image: &Image,
        rect: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Mapping, Failure> {
        self.copy_texture(image, rect, format)
    }
    fn copy_texture(
        &mut self,
        image: &Image,
        rect: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Mapping, Failure> {
        self.queue.drain();
        let pixels = image.pixels.lock().unwrap();
        let mut data = Vec::new();
        for y in rect.loc.y..rect.loc.y + rect.size.h {
            let start = (y * image.size.w + rect.loc.x) as usize * 4;
            data.extend_from_slice(&pixels[start..start + rect.size.w as usize * 4]);
        }
        Ok(Mapping {
            image: Image::new(rect.size, format),
            data,
        })
    }
    fn can_read_texture(&mut self, _: &Image) -> Result<bool, Failure> {
        Ok(true)
    }
    fn map_texture<'a>(&mut self, mapping: &'a Mapping) -> Result<&'a [u8], Failure> {
        Ok(&mapping.data)
    }
}
#[derive(Debug)]
struct QueuedFrame {
    target: Image,
    queue: Arc<Queue>,
    faults: Arc<Mutex<Faults>>,
    context: ContextId<Image>,
    transform: Transform,
}
impl Frame for QueuedFrame {
    type Error = Failure;
    type TextureId = Image;
    fn context_id(&self) -> ContextId<Image> {
        self.context.clone()
    }
    fn clear(&mut self, color: Color32F, damage: &[Rectangle<i32, Physical>]) -> Result<(), Failure> {
        let target = self.target.clone();
        let damage = damage.to_vec();
        let color = color.components().map(|v| (v * 255.0).round() as u8);
        self.queue.submit(move || {
            let mut pixels = target.pixels.lock().unwrap();
            for rect in damage {
                for y in rect.loc.y..rect.loc.y + rect.size.h {
                    for x in rect.loc.x..rect.loc.x + rect.size.w {
                        let at = (y * target.size.w + x) as usize * 4;
                        pixels[at..at + 4].copy_from_slice(&color);
                    }
                }
            }
        });
        Ok(())
    }
    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Failure> {
        let damage: Vec<_> = damage
            .iter()
            .map(|rect| Rectangle::new(rect.loc + dst.loc, rect.size))
            .collect();
        self.clear(color, &damage)
    }
    fn render_texture_from_to(
        &mut self,
        texture: &Image,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _: &[Rectangle<i32, Physical>],
        transform: Transform,
        alpha: f32,
    ) -> Result<(), Failure> {
        if transform != Transform::Normal || alpha != 1.0 {
            return Err(Failure::Unsupported);
        }
        let source = texture.clone();
        let target = self.target.clone();
        let damage = damage.to_vec();
        self.queue.submit(move || {
            let source_pixels = source.pixels.lock().unwrap().clone();
            let mut target_pixels = target.pixels.lock().unwrap();
            for rect in damage {
                for y in rect.loc.y..rect.loc.y + rect.size.h {
                    for x in rect.loc.x..rect.loc.x + rect.size.w {
                        let sx = src.loc.x as i32 + x;
                        let sy = src.loc.y as i32 + y;
                        let from = (sy * source.size.w + sx) as usize * 4;
                        let to = ((dst.loc.y + y) * target.size.w + dst.loc.x + x) as usize * 4;
                        target_pixels[to..to + 4].copy_from_slice(&source_pixels[from..from + 4]);
                    }
                }
            }
        });
        if self.faults.lock().unwrap().draw {
            Err(Failure::Injected)
        } else {
            Ok(())
        }
    }
    fn transformation(&self) -> Transform {
        self.transform
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Failure> {
        self.queue.wait(sync);
        Ok(())
    }
    fn finish(self) -> Result<SyncPoint, Failure> {
        if self.faults.lock().unwrap().finish {
            Err(Failure::Injected)
        } else {
            Ok(self.queue.fence())
        }
    }
}

#[derive(Debug)]
struct TestAllocator {
    images: Images,
}
impl Allocator for TestAllocator {
    type Buffer = Dmabuf;
    type Error = AnyError;
    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        format: Fourcc,
        _: &[Modifier],
    ) -> Result<Dmabuf, AnyError> {
        // Only identity/metadata are used by this renderer, never the fd contents.
        let mut builder = Dmabuf::builder(
            (width as i32, height as i32),
            format,
            Modifier::Linear,
            DmabufFlags::empty(),
        );
        builder.add_plane(File::open("/dev/null").unwrap().into(), 0, 0, width * 4);
        let buffer = builder.build().unwrap();
        self.images
            .lock()
            .unwrap()
            .insert(buffer.clone(), Image::new(buffer.size(), format));
        Ok(buffer)
    }
}
#[derive(Debug)]
struct TestDevice {
    renderer: QueuedRenderer,
    allocator: TestAllocator,
    node: DrmNode,
}
impl ApiDevice for TestDevice {
    type Renderer = QueuedRenderer;
    fn renderer(&self) -> &QueuedRenderer {
        &self.renderer
    }
    fn renderer_mut(&mut self) -> &mut QueuedRenderer {
        &mut self.renderer
    }
    fn allocator(&mut self) -> &mut dyn Allocator<Buffer = Dmabuf, Error = AnyError> {
        &mut self.allocator
    }
    fn node(&self) -> &DrmNode {
        &self.node
    }
}
#[derive(Debug)]
struct TestApi;
impl GraphicsApi for TestApi {
    type Device = TestDevice;
    type Error = Failure;
    fn enumerate(&self, _: &mut Vec<TestDevice>) -> Result<(), Failure> {
        Ok(())
    }
    fn identifier() -> &'static str {
        "queued-test"
    }
}

struct Fixture {
    manager: GpuManager<TestApi>,
    source: DrmNode,
    target: DrmNode,
}
impl Fixture {
    fn new(cpu_copy: bool) -> Self {
        // Only look for render nodes if the environment does not select both.
        let discovered = || -> Vec<DrmNode> {
            std::fs::read_dir("/dev/dri")
                .expect("requires DRM render node identities")
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
                .filter_map(|entry| DrmNode::from_path(entry.path()).ok())
                .collect()
        };
        let source = std::env::var("SMITHAY_TEST_RENDER_NODE")
            .ok()
            .map(|path| DrmNode::from_path(path).unwrap())
            .or_else(|| discovered().first().copied())
            .expect("requires a source DRM identity");
        let target = std::env::var("SMITHAY_TEST_TARGET_NODE")
            .ok()
            .map(|path| DrmNode::from_path(path).unwrap())
            .or_else(|| discovered().into_iter().find(|node| *node != source))
            .expect("requires a second DRM identity");
        assert_ne!(source, target);
        let images = Images::default();
        let devices = [source, target]
            .into_iter()
            .map(|node| TestDevice {
                node,
                renderer: QueuedRenderer {
                    queue: Arc::default(),
                    images: images.clone(),
                    faults: Arc::default(),
                    context: ContextId::new(),
                    flags: DebugFlags::empty(),
                    import_dma: node == source || !cpu_copy,
                },
                allocator: TestAllocator {
                    images: images.clone(),
                },
            })
            .collect();
        Self {
            source,
            target,
            manager: GpuManager {
                api: TestApi,
                devices,
                dmabuf_cache: HashMap::new(),
                span: tracing::Span::none(),
            },
        }
    }
    fn renderer(&mut self) -> MultiRenderer<'_, '_, TestApi, TestApi> {
        self.manager
            .renderer(&self.source, &self.target, Fourcc::Abgr8888)
            .unwrap()
    }
    fn drain(&self) {
        // Deliberately give the producer all available execution time first.
        self.manager.devices[0].renderer.queue.drain();
        self.manager.devices[1].renderer.queue.drain();
    }
    fn completion(&self) -> &CopyCompletion {
        &self.manager.dmabuf_cache[&(self.source, self.target)]
            .as_ref()
            .unwrap()
            .copy_completion
    }
    fn draw(&mut self, output: &mut Image, color: Color32F, drop_frame: bool) {
        let size = output.size.to_logical(1, Transform::Normal).to_physical(1);
        let mut renderer = self.renderer();
        let mut fb = renderer.bind(output).unwrap();
        let mut frame = renderer.render(&mut fb, size, Transform::Normal).unwrap();
        frame.clear(color, &[Rectangle::from_size(size)]).unwrap();
        if drop_frame {
            drop(frame);
        } else {
            let _ = frame.finish().unwrap();
        }
    }
}
const RED: Color32F = Color32F::new(1.0, 0.0, 0.0, 1.0);
const BLUE: Color32F = Color32F::new(0.0, 0.0, 1.0, 1.0);
fn output() -> Image {
    Image::new((2, 1).into(), Fourcc::Abgr8888)
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn target_read_finishes_before_shared_scratch_is_overwritten() {
    for drop_frame in [false, true] {
        let mut fixture = Fixture::new(false);
        let mut first = output();
        let mut second = output();
        fixture.draw(&mut first, RED, drop_frame);
        fixture.draw(&mut second, BLUE, drop_frame);
        fixture.drain();
        assert_eq!(first.pixel(0), [255, 0, 0, 255]);
        assert_eq!(second.pixel(0), [0, 0, 255, 255]);
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn internal_flush_orders_copy_before_resuming_source_frame() {
    let mut fixture = Fixture::new(false);
    let mut output = output();
    let observed = output.clone();
    let size = (2, 1).into();
    let mut renderer = fixture.renderer();
    let mut fb = renderer.bind(&mut output).unwrap();
    let mut frame = renderer.render(&mut fb, size, Transform::Normal).unwrap();
    frame.clear(RED, &[Rectangle::from_size(size)]).unwrap();
    frame.flush_frame().unwrap();
    // Run the resumed producer before the first target copy. The dependency must
    // make that first copy see red, not this subsequently queued blue overwrite.
    let first_copy = match &*frame.target.as_ref().unwrap().copy_completion {
        CopyCompletion::Idle => frame.target.as_ref().unwrap().device.renderer.queue.fence(),
        other => panic!("restart did not acquire buffer: {other:?}"),
    };
    frame.clear(BLUE, &[Rectangle::from_size(size)]).unwrap();
    let source_queue = frame.frame.as_ref().unwrap().queue.clone();
    source_queue.drain();
    first_copy.wait().unwrap();
    assert_eq!(observed.pixel(0), [255, 0, 0, 255]);
    let _ = frame.finish().unwrap();
    fixture.drain();
    assert_eq!(output.pixel(0), [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn failed_or_interrupted_wait_retains_fence_and_does_not_rebind() {
    for error in [Failure::Injected, Failure::Interrupted] {
        let mut fixture = Fixture::new(false);
        let mut output = output();
        fixture.draw(&mut output, RED, false);
        let faults = fixture.manager.devices[0].renderer.faults.clone();
        let binds = faults.lock().unwrap().binds;
        faults.lock().unwrap().wait = Some(error);
        {
            let mut renderer = fixture.renderer();
            let mut fb = renderer.bind(&mut output).unwrap();
            assert!(
                matches!(renderer.render(&mut fb, (2, 1).into(), Transform::Normal), Err(Error::Render(e)) if e == error)
            );
        }
        assert_eq!(faults.lock().unwrap().binds, binds);
        assert!(matches!(fixture.completion(), CopyCompletion::Pending(sync) if !sync.is_reached()));
        faults.lock().unwrap().wait = None;
        // The retry must still order the first copy before the next producer write.
        let mut second = self::output();
        fixture.draw(&mut second, BLUE, false);
        fixture.drain();
        assert_eq!(output.pixel(0), [255, 0, 0, 255]);
        assert_eq!(second.pixel(0), [0, 0, 255, 255]);
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn copy_errors_and_later_cleanup_errors_retain_reader_fence() {
    for draw_error in [false, true] {
        let mut fixture = Fixture::new(false);
        fixture.manager.devices[usize::from(draw_error)]
            .renderer
            .faults
            .lock()
            .unwrap()
            .cleanup = !draw_error;
        fixture.manager.devices[1].renderer.faults.lock().unwrap().draw = draw_error;
        let mut first = output();
        {
            let mut renderer = fixture.renderer();
            let mut fb = renderer.bind(&mut first).unwrap();
            let mut frame = renderer
                .render(&mut fb, (2, 1).into(), Transform::Normal)
                .unwrap();
            frame.clear(RED, &[Rectangle::from_size((2, 1).into())]).unwrap();
            assert!(frame.finish().is_err());
        }
        assert!(matches!(fixture.completion(), CopyCompletion::Pending(_)));
        fixture.manager.devices[0].renderer.faults.lock().unwrap().cleanup = false;
        fixture.manager.devices[1].renderer.faults.lock().unwrap().draw = false;
        let mut second = output();
        fixture.draw(&mut second, BLUE, false);
        fixture.drain();
        assert_eq!(first.pixel(0), [255, 0, 0, 255]);
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn failed_target_finish_poison_prevents_reuse_including_drop() {
    for drop_frame in [false, true] {
        let mut fixture = Fixture::new(false);
        fixture.manager.devices[1].renderer.faults.lock().unwrap().finish = true;
        let mut output = output();
        {
            let mut renderer = fixture.renderer();
            let mut fb = renderer.bind(&mut output).unwrap();
            let mut frame = renderer
                .render(&mut fb, (2, 1).into(), Transform::Normal)
                .unwrap();
            frame.clear(RED, &[Rectangle::from_size((2, 1).into())]).unwrap();
            if drop_frame {
                drop(frame);
            } else {
                assert!(frame.finish().is_err());
            }
        }
        assert!(matches!(fixture.completion(), CopyCompletion::Unfenced));
        let binds = fixture.manager.devices[0].renderer.faults.lock().unwrap().binds;
        {
            let mut renderer = fixture.renderer();
            let mut fb = renderer.bind(&mut output).unwrap();
            assert!(matches!(
                renderer.render(&mut fb, (2, 1).into(), Transform::Normal),
                Err(Error::MissingCopyCompletion)
            ));
        }
        assert_eq!(
            fixture.manager.devices[0].renderer.faults.lock().unwrap().binds,
            binds
        );
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn size_and_format_replacement_wait_before_discarding_old_completion() {
    for (size, format) in [((3, 1), Fourcc::Abgr8888), ((2, 1), Fourcc::Argb8888)] {
        let mut fixture = Fixture::new(false);
        let mut first = output();
        fixture.draw(&mut first, RED, false);
        let old = fixture.manager.dmabuf_cache[&(fixture.source, fixture.target)]
            .as_ref()
            .unwrap()
            .dmabuf
            .clone();
        let faults = fixture.manager.devices[0].renderer.faults.clone();
        faults.lock().unwrap().wait = Some(Failure::Interrupted);
        let mut next = Image::new(size.into(), format);
        {
            let mut renderer = fixture
                .manager
                .renderer(&fixture.source, &fixture.target, format)
                .unwrap();
            let mut fb = renderer.bind(&mut next).unwrap();
            assert!(renderer.render(&mut fb, size.into(), Transform::Normal).is_err());
        }
        assert_eq!(
            fixture.manager.dmabuf_cache[&(fixture.source, fixture.target)]
                .as_ref()
                .unwrap()
                .dmabuf,
            old
        );
        faults.lock().unwrap().wait = None;
        {
            let mut renderer = fixture
                .manager
                .renderer(&fixture.source, &fixture.target, format)
                .unwrap();
            let mut fb = renderer.bind(&mut next).unwrap();
            let mut frame = renderer.render(&mut fb, size.into(), Transform::Normal).unwrap();
            frame.clear(BLUE, &[Rectangle::from_size(size.into())]).unwrap();
            let _ = frame.finish().unwrap();
        }
        assert_ne!(
            fixture.manager.dmabuf_cache[&(fixture.source, fixture.target)]
                .as_ref()
                .unwrap()
                .dmabuf,
            old
        );
        fixture.drain();
        // Only the allocation identities above discriminate: `next` is drawn into a
        // new allocation, so these pixels hold even if the old copy were not waited for.
        assert_eq!(first.pixel(0), [255, 0, 0, 255]);
        assert_eq!(next.pixel(0), [0, 0, 255, 255]);
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn memory_copy_and_single_gpu_do_not_add_reader_waits() {
    for single_gpu in [false, true] {
        let mut fixture = Fixture::new(true);
        if single_gpu {
            fixture.target = fixture.source;
        }
        let mut first = output();
        let mut second = output();
        fixture.draw(&mut first, RED, false);
        fixture.draw(&mut second, BLUE, true);
        fixture.drain();
        assert_eq!(first.pixel(0), [255, 0, 0, 255]);
        assert_eq!(second.pixel(0), [0, 0, 255, 255]);
        assert_eq!(
            fixture.manager.devices[0].renderer.faults.lock().unwrap().waits,
            0
        );
        if single_gpu {
            assert!(fixture.manager.dmabuf_cache.is_empty());
        } else {
            assert!(matches!(fixture.completion(), CopyCompletion::Idle));
        }
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn raw_damage_copies_only_reconstructed_pixels_of_current_output() {
    for cpu_copy in [false, true] {
        let mut fixture = Fixture::new(cpu_copy);
        let mut first = output();
        let mut second = output();
        fixture.draw(&mut first, RED, false);
        fixture.draw(&mut second, BLUE, false);
        {
            let mut renderer = fixture.renderer();
            let mut fb = renderer.bind(&mut first).unwrap();
            let mut frame = renderer
                .render(&mut fb, (2, 1).into(), Transform::Normal)
                .unwrap();
            let damage = [Rectangle::new((0, 0).into(), (1, 1).into())];
            frame.damage(&damage);
            frame.as_mut().clear(Color32F::BLACK, &damage).unwrap();
            let _ = frame.finish().unwrap();
        }
        fixture.drain();
        assert_eq!(first.pixel(0), [0, 0, 0, 255]);
        assert_eq!(
            first.pixel(1),
            [255, 0, 0, 255],
            "undamaged scratch belongs to the other output"
        );
        assert_eq!(second.pixel(0), [0, 0, 255, 255]);
    }
}

#[cfg(all(feature = "backend_gbm", feature = "backend_egl", feature = "renderer_gl"))]
mod gpu;

#[cfg(all(feature = "backend_gbm", feature = "backend_egl", feature = "renderer_gl"))]
#[test]
#[ignore = "requires two explicitly selected physical DRM render nodes"]
fn real_gpu_raw_damage() {
    use crate::backend::renderer::gles::ffi;
    gpu::exercise(|frame, color, damage| {
        frame.damage(damage);
        frame
            .as_mut()
            .with_context(|gl| unsafe {
                gl.Enable(ffi::SCISSOR_TEST);
                gl.ClearColor(color.r(), color.g(), color.b(), color.a());
                for rect in damage {
                    gl.Scissor(rect.loc.x, rect.loc.y, rect.size.w, rect.size.h);
                    gl.Clear(ffi::COLOR_BUFFER_BIT);
                }
                gl.Disable(ffi::SCISSOR_TEST);
            })
            .unwrap();
    });
}

#[test]
fn copy_completion_orders_delayed_reader_before_producer_without_cpu_wait() {
    use std::sync::atomic::AtomicUsize;
    let source = Arc::new(Queue::default());
    let target = Arc::new(Queue::default());
    let pixels = Arc::new(AtomicUsize::new(1));
    let observed = Arc::new(AtomicUsize::new(0));
    let read_pixels = pixels.clone();
    let read_result = observed.clone();
    target.submit(move || read_result.store(read_pixels.load(SeqCst), SeqCst));
    let mut completion = CopyCompletion::Idle;
    completion.begin_read();
    let returned = completion.finish_read::<Failure>(Ok(target.fence())).unwrap();
    drop(returned); // The caller need not keep or wait the returned frame fence.
    completion
        .wait(
            |sync| {
                source.wait(sync);
                Ok(())
            },
            || Failure::Injected,
        )
        .unwrap();
    assert_eq!(observed.load(SeqCst), 0, "acquisition must not CPU-wait");
    source.submit(move || pixels.store(2, SeqCst));
    source.drain();
    assert_eq!(observed.load(SeqCst), 1, "target read must precede the overwrite");
    assert!(matches!(completion, CopyCompletion::Idle));
}

#[test]
fn copy_completion_failed_wait_keeps_unsatisfied_dependency_for_retry() {
    for error in [Failure::Interrupted, Failure::Injected] {
        let queue = Arc::new(Queue::default());
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let on_read = observed.clone();
        queue.submit(move || on_read.store(true, SeqCst));
        let mut completion = CopyCompletion::Idle;
        completion.begin_read();
        let _ = completion.finish_read::<Failure>(Ok(queue.fence())).unwrap();
        let mut rebound = false;
        let result = completion
            .wait(|_| Err(error), || Failure::Unsupported)
            .map(|()| rebound = true);
        assert_eq!(result, Err(error));
        assert!(!rebound);
        assert!(!observed.load(SeqCst));
        assert!(matches!(&completion, CopyCompletion::Pending(sync) if !sync.is_reached()));
        completion
            .wait(
                |sync| sync.wait().map_err(|_| Failure::Interrupted),
                || Failure::Unsupported,
            )
            .unwrap();
        assert!(observed.load(SeqCst));
        assert!(matches!(completion, CopyCompletion::Idle));
    }
}

#[test]
fn copy_completion_uncertain_finish_poison_prevents_acquisition() {
    let mut completion = CopyCompletion::Idle;
    completion.begin_read();
    assert_eq!(
        completion
            .finish_read::<Failure>(Err(Failure::Injected))
            .unwrap_err(),
        Failure::Injected
    );
    assert_eq!(
        completion.wait::<Failure>(|_| panic!("no trustworthy fence exists"), || Failure::Unsupported),
        Err(Failure::Unsupported),
    );
    assert!(matches!(completion, CopyCompletion::Unfenced));
}

#[test]
fn copy_completion_idle_and_completed_release_do_not_add_repeated_waits() {
    let mut completion = CopyCompletion::Idle;
    completion
        .wait::<Failure>(
            |_| panic!("idle allocation needs no wait"),
            || Failure::Unsupported,
        )
        .unwrap();
    completion.begin_read();
    let _ = completion
        .finish_read::<Failure>(Ok(SyncPoint::signaled()))
        .unwrap();
    let mut waits = 0;
    completion
        .wait(
            |sync| {
                assert!(sync.is_reached());
                waits += 1;
                Ok(())
            },
            || Failure::Unsupported,
        )
        .unwrap();
    completion
        .wait::<Failure>(|_| panic!("already acquired dependency"), || Failure::Unsupported)
        .unwrap();
    assert_eq!(waits, 1);
}

#[derive(Debug)]
struct InterruptibleFence {
    attempts: Arc<std::sync::atomic::AtomicUsize>,
    interruptions: usize,
}
impl sync::Fence for InterruptibleFence {
    fn is_signaled(&self) -> bool {
        self.attempts.load(SeqCst) > self.interruptions
    }
    fn wait(&self) -> Result<(), sync::Interrupted> {
        if self.attempts.fetch_add(1, SeqCst) < self.interruptions {
            Err(sync::Interrupted)
        } else {
            Ok(())
        }
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

#[test]
fn imported_shadow_failed_gpu_and_interrupted_cpu_wait_retains_fence() {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut pending = Some(
        InterruptibleFence {
            attempts: attempts.clone(),
            interruptions: 1,
        }
        .into(),
    );
    assert_eq!(
        wait_for_pending(&mut pending, |_| Err(Failure::Injected)),
        Err(Failure::Injected)
    );
    assert!(pending.as_ref().is_some_and(|sync| !sync.is_reached()));
    assert_eq!(attempts.load(SeqCst), 1);
    wait_for_pending(&mut pending, |_| Err(Failure::Injected)).unwrap();
    assert!(pending.is_none());
    assert_eq!(attempts.load(SeqCst), 2);
}

#[test]
fn imported_shadow_successful_gpu_wait_does_not_cpu_wait() {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut pending = Some(
        InterruptibleFence {
            attempts: attempts.clone(),
            interruptions: 1,
        }
        .into(),
    );
    wait_for_pending(&mut pending, |_| Ok::<_, Failure>(())).unwrap();
    assert!(pending.is_none());
    assert_eq!(attempts.load(SeqCst), 0);
}

#[test]
fn cpu_shadow_conversion_retries_interruptions_before_releasing_fence() {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sync = InterruptibleFence {
        attempts: attempts.clone(),
        interruptions: 3,
    }
    .into();
    wait_uninterrupted(&sync);
    assert!(sync.is_reached());
    assert_eq!(attempts.load(SeqCst), 4);
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn failed_shadow_import_preserves_cache_until_dependency_can_be_retried() {
    // Exercise both the destination-device and other-device import branches.
    for destination_device in [true, false] {
        let mut fixture = Fixture::new(false);
        let format = Fourcc::Abgr8888;
        let original = fixture.manager.devices[1]
            .allocator
            .create_buffer(2, 1, format, &[Modifier::Linear])
            .unwrap();
        original.set_node(fixture.target);
        let shadow = fixture.manager.devices[1]
            .allocator
            .create_buffer(2, 1, format, &[Modifier::Linear])
            .unwrap();
        let images = fixture.manager.devices[1].renderer.images.clone();
        images.lock().unwrap()[&original]
            .pixels
            .lock()
            .unwrap()
            .copy_from_slice(&[255, 0, 0, 255, 255, 0, 0, 255]);
        let shadow_image = images.lock().unwrap()[&shadow].clone();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let texture = MultiTexture::new(
            (2, 1).into(),
            Format {
                code: format,
                modifier: Modifier::Linear,
            },
        );
        texture.0.lock().unwrap().format = Some(format);
        texture
            .0
            .lock()
            .unwrap()
            .textures
            .entry(TypeId::of::<TestApi>())
            .or_default()
            .insert(
                fixture.source,
                GpuSingleTexture::Dma {
                    texture: Box::new(shadow_image),
                    dmabuf: shadow.clone(),
                    sync: Some(
                        InterruptibleFence {
                            attempts: attempts.clone(),
                            interruptions: 1,
                        }
                        .into(),
                    ),
                },
            );
        fixture.manager.devices[1].renderer.faults.lock().unwrap().wait = Some(Failure::Injected);
        {
            let mut renderer = if destination_device {
                fixture.renderer()
            } else {
                fixture.manager.single_renderer(&fixture.source).unwrap()
            };
            assert!(renderer
                .import_dmabuf_internal(&original, texture.clone(), None)
                .is_err());
        }
        {
            let cached = texture.0.lock().unwrap();
            assert!(
                matches!(
                    cached.textures[&TypeId::of::<TestApi>()].get(&fixture.source),
                    Some(GpuSingleTexture::Dma { dmabuf, sync: Some(sync), .. })
                        if dmabuf == &shadow && !sync.is_reached()
                ),
                "failed import must retain the unchanged shadow and its unsatisfied dependency",
            );
        }
        assert_eq!(attempts.load(SeqCst), 1);
        assert_eq!(
            fixture.manager.devices[1].renderer.faults.lock().unwrap().binds,
            0
        );
        fixture.manager.devices[1].renderer.faults.lock().unwrap().wait = None;
        {
            let mut renderer = if destination_device {
                fixture.renderer()
            } else {
                fixture.manager.single_renderer(&fixture.source).unwrap()
            };
            renderer
                .import_dmabuf_internal(&original, texture.clone(), None)
                .unwrap();
        }
        fixture.drain();
        let copied = texture.get::<TestApi>(&fixture.source).unwrap();
        assert_eq!(copied.pixel(0), [255, 0, 0, 255]);
        assert_eq!(copied.pixel(1), [255, 0, 0, 255]);
        assert_eq!(
            attempts.load(SeqCst),
            2,
            "retry must complete the retained dependency"
        );
        let cached = texture.0.lock().unwrap();
        assert!(
            matches!(&cached.textures[&TypeId::of::<TestApi>()][&fixture.source],
                GpuSingleTexture::Dma { dmabuf, .. } if dmabuf == &shadow),
            "retry must reuse the existing shadow, not replace the failed import",
        );
    }
}

#[test]
#[ignore = "requires two real DRM node identities; uses deterministic software queues"]
fn failed_internal_restart_keeps_copy_fence_and_does_not_open_source_frame() {
    let mut fixture = Fixture::new(false);
    let faults = fixture.manager.devices[0].renderer.faults.clone();
    let mut first = output();
    {
        let mut renderer = fixture.renderer();
        let mut fb = renderer.bind(&mut first).unwrap();
        let mut frame = renderer
            .render(&mut fb, (2, 1).into(), Transform::Normal)
            .unwrap();
        frame.clear(RED, &[Rectangle::from_size((2, 1).into())]).unwrap();
        faults.lock().unwrap().wait = Some(Failure::Interrupted);
        assert!(matches!(
            frame.flush_frame(),
            Err(Error::Render(Failure::Interrupted))
        ));
        assert!(
            frame.frame.is_none(),
            "a failed wait must not resume producer writes"
        );
        assert!(matches!(
            &*frame.target.as_ref().unwrap().copy_completion,
            CopyCompletion::Pending(_)
        ));
        // The aborted frame reports errors instead of panicking, and a later restart
        // must not revive it even once waiting would succeed.
        faults.lock().unwrap().wait = None;
        assert!(matches!(
            frame.clear(BLUE, &[Rectangle::from_size((2, 1).into())]),
            Err(Error::FrameAborted)
        ));
        assert_eq!(frame.transformation(), Transform::Normal);
        let _ = frame.context_id();
        assert!(matches!(frame.flush_frame(), Err(Error::FrameAborted)));
        assert!(frame.frame.is_none());
        assert!(matches!(frame.finish(), Err(Error::FrameAborted)));
    }
    assert!(matches!(fixture.completion(), CopyCompletion::Pending(_)));
    let mut second = output();
    fixture.draw(&mut second, BLUE, false);
    fixture.drain();
    assert_eq!(first.pixel(0), [255, 0, 0, 255]);
    assert_eq!(second.pixel(0), [0, 0, 255, 255]);
}
