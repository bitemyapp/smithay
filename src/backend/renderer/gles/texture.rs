use ffi::Gles2;

use super::*;
use std::sync::Arc;

/// A handle to a GLES texture
///
/// The texture can be used with the same [`GlesRenderer`] it was created with, or one using a
/// shared [`EGLContext`].
#[derive(Debug, Clone)]
pub struct GlesTexture(pub(super) Arc<GlesTextureInternal>);

impl GlesTexture {
    /// Create a GlesTexture from a raw gl texture id.
    ///
    /// It is expected to not be external or y_inverted.
    ///
    /// Ownership over the texture is taken by the renderer, you should not free the texture yourself.
    ///
    /// # Safety
    ///
    /// The renderer cannot make sure `tex` is a valid texture id.
    /// Any access submitted outside this renderer must have completed before this
    /// call; the renderer cannot track fences belonging to the raw texture.
    pub unsafe fn from_raw(
        renderer: &GlesRenderer,
        internal_format: Option<ffi::types::GLenum>,
        opaque: bool,
        tex: ffi::types::GLuint,
        size: Size<i32, BufferCoord>,
    ) -> GlesTexture {
        GlesTexture(Arc::new(GlesTextureInternal {
            texture: tex,
            sync: RwLock::default(),
            format: internal_format,
            has_alpha: !opaque,
            is_external: false,
            y_inverted: false,
            size,
            egl_images: None,
            destruction_callback_sender: renderer.destruction_callback_sender.clone(),
        }))
    }

    /// OpenGL texture id of this texture
    ///
    /// This id will become invalid, when the GlesTexture is dropped and does not transfer ownership.
    pub fn tex_id(&self) -> ffi::types::GLuint {
        self.0.texture
    }

    /// Whether the texture is upside down
    pub fn is_y_inverted(&self) -> bool {
        self.0.y_inverted
    }

    /// Whether this is the only reference to this texture (strong or weak)
    ///
    /// Note that this tracks only references to this Smithay object (that you can get by cloning
    /// it). If you make a reference via OpenGL directly somehow, you need to keep track of it on
    /// your own.
    pub fn is_unique_reference(&mut self) -> bool {
        Arc::get_mut(&mut self.0).is_some()
    }
}

#[derive(Debug, Default)]
pub(super) struct TextureSync {
    read_sync: Mutex<Option<ffi::types::GLsync>>,
    write_sync: Mutex<Option<ffi::types::GLsync>>,
}

unsafe fn wait_for_syncpoint(sync: &mut Option<ffi::types::GLsync>, gl: &Gles2) {
    if let Some(sync_obj) = *sync {
        match gl.ClientWaitSync(sync_obj, 0, 0) {
            ffi::ALREADY_SIGNALED | ffi::CONDITION_SATISFIED => {
                let _ = sync.take();
                gl.DeleteSync(sync_obj);
            }
            _ => {
                gl.WaitSync(sync_obj, 0, ffi::TIMEOUT_IGNORED);
            }
        };
    }
}

// The caller holds the texture's read/write lock until publication is complete.
// A fence in an unsubmitted producer stream cannot be waited on by another context.
unsafe fn publish_sync(gl: &Gles2, fencing: bool) -> Option<ffi::types::GLsync> {
    if fencing {
        let sync = gl.FenceSync(ffi::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if !sync.is_null() {
            gl.Flush();
            return Some(sync);
        }
    }
    // GLES2 and allocation failure both require actual completion, not a missing
    // fence that would incorrectly advertise the access as already complete.
    gl.Finish();
    None
}

impl TextureSync {
    pub(super) fn wait_for_upload(&self, gl: &Gles2) {
        unsafe {
            wait_for_syncpoint(&mut self.write_sync.lock().unwrap(), gl);
        }
    }

    pub(super) fn update_read(&self, gl: &Gles2, shared: bool, fencing: bool) {
        if !shared {
            return;
        }
        let mut read_sync = self.read_sync.lock().unwrap();
        if let Some(old) = read_sync.take() {
            unsafe {
                gl.WaitSync(old, 0, ffi::TIMEOUT_IGNORED);
                gl.DeleteSync(old);
            };
        }
        *read_sync = unsafe { publish_sync(gl, fencing) };
    }

    pub(super) fn wait_for_all(&mut self, gl: &Gles2) {
        unsafe {
            wait_for_syncpoint(self.read_sync.get_mut().unwrap(), gl);
            wait_for_syncpoint(self.write_sync.get_mut().unwrap(), gl);
        }
    }

    pub(super) fn update_write(&mut self, gl: &Gles2, shared: bool, fencing: bool) {
        if !shared {
            return;
        }
        let write_sync = self.write_sync.get_mut().unwrap();
        if let Some(old) = write_sync.take() {
            unsafe {
                gl.WaitSync(old, 0, ffi::TIMEOUT_IGNORED);
                gl.DeleteSync(old);
            };
        }

        *write_sync = unsafe { publish_sync(gl, fencing) };
    }
}

#[derive(Debug)]
pub(super) struct GlesTextureInternal {
    pub(super) texture: ffi::types::GLuint,
    pub(super) sync: RwLock<TextureSync>,
    pub(super) format: Option<ffi::types::GLenum>,
    pub(super) has_alpha: bool,
    pub(super) is_external: bool,
    pub(super) y_inverted: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) egl_images: Option<Vec<EGLImage>>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
}
unsafe impl Send for GlesTextureInternal {}
unsafe impl Sync for GlesTextureInternal {}

impl Drop for GlesTextureInternal {
    fn drop(&mut self) {
        let _ = self
            .destruction_callback_sender
            .send(CleanupResource::Texture(self.texture));
        let mut sync = self.sync.write().unwrap();
        if let Some(sync) = sync.read_sync.get_mut().unwrap().take() {
            let _ = self
                .destruction_callback_sender
                .send(CleanupResource::Sync(sync as *const _));
        }
        if let Some(sync) = sync.write_sync.get_mut().unwrap().take() {
            let _ = self
                .destruction_callback_sender
                .send(CleanupResource::Sync(sync as *const _));
        }
        if let Some(images) = self.egl_images.take() {
            for image in images {
                let _ = self
                    .destruction_callback_sender
                    .send(CleanupResource::EGLImage(image));
            }
        }
    }
}

impl Texture for GlesTexture {
    fn width(&self) -> u32 {
        self.0.size.w as u32
    }
    fn height(&self) -> u32 {
        self.0.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.0.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_internal_format_to_fourcc(self.0.format?);
        if self.0.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}

/// Texture mapping of a GLES2 texture
#[derive(Debug)]
pub struct GlesMapping {
    pub(super) pbo: ffi::types::GLuint,
    pub(super) format: ffi::types::GLenum,
    pub(super) layout: ffi::types::GLenum,
    pub(super) has_alpha: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) mapping: AtomicPtr<std::ffi::c_void>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
}

impl Texture for GlesMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }
    fn height(&self) -> u32 {
        self.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_read_format_to_fourcc(self.format, self.layout);
        if self.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}
impl TextureMapping for GlesMapping {
    fn flipped(&self) -> bool {
        true
    }
    fn format(&self) -> Fourcc {
        Texture::format(self).expect("Should never happen")
    }
}

impl Drop for GlesMapping {
    fn drop(&mut self) {
        let _ = self.destruction_callback_sender.send(CleanupResource::Mapping(
            self.pbo,
            self.mapping.load(Ordering::SeqCst),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, collections::HashMap};

    #[derive(Clone, Copy)]
    enum Command {
        Write(u32),
        Read(u32),
        Wait(usize),
        Fence,
    }

    #[derive(Default)]
    struct Queue {
        commands: Vec<Command>,
        submitted: usize,
        completed: usize,
    }

    #[derive(Default)]
    struct Driver {
        queues: [Queue; 3],
        current: usize,
        value: u32,
        fences: HashMap<usize, (usize, usize)>,
        next_fence: usize,
        fail_fence: bool,
        fence_calls: usize,
        finishes: usize,
    }

    impl Driver {
        fn queue(&mut self, command: Command) {
            self.queues[self.current].commands.push(command);
        }

        fn submit(&mut self) {
            let queue = &mut self.queues[self.current];
            queue.submitted = queue.commands.len();
        }

        fn execute(&mut self, context: usize, end: usize) {
            assert!(end <= self.queues[context].submitted, "producer was not flushed");
            while self.queues[context].completed < end {
                let index = self.queues[context].completed;
                match self.queues[context].commands[index] {
                    Command::Write(value) => self.value = value,
                    Command::Read(expected) => assert_eq!(self.value, expected, "unordered texture access"),
                    Command::Wait(fence) => {
                        let (producer, end) = self.fences[&fence];
                        self.execute(producer, end);
                    }
                    Command::Fence => {}
                }
                self.queues[context].completed += 1;
            }
        }
    }

    thread_local! {
        static DRIVER: RefCell<Driver> = RefCell::default();
    }

    extern "system" fn fence(_: u32, _: u32) -> ffi::types::GLsync {
        DRIVER.with_borrow_mut(|driver| {
            driver.fence_calls += 1;
            if driver.fail_fence {
                return std::ptr::null();
            }
            driver.next_fence += 1;
            let id = driver.next_fence;
            driver.queue(Command::Fence);
            driver
                .fences
                .insert(id, (driver.current, driver.queues[driver.current].commands.len()));
            id as ffi::types::GLsync
        })
    }

    extern "system" fn flush() {
        DRIVER.with_borrow_mut(Driver::submit);
    }

    extern "system" fn finish() {
        DRIVER.with_borrow_mut(|driver| {
            driver.finishes += 1;
            driver.submit();
            driver.execute(driver.current, driver.queues[driver.current].submitted);
        });
    }

    extern "system" fn wait(sync: ffi::types::GLsync, _: u32, _: u64) {
        DRIVER.with_borrow_mut(|driver| driver.queue(Command::Wait(sync as usize)));
    }

    extern "system" fn client_wait(_: ffi::types::GLsync, _: u32, _: u64) -> u32 {
        ffi::TIMEOUT_EXPIRED
    }

    extern "system" fn delete(_: ffi::types::GLsync) {
        // Queued waits retain the GL sync object even after its name is deleted.
    }

    fn gl() -> Gles2 {
        DRIVER.with_borrow_mut(|driver| *driver = Driver::default());
        Gles2::load_with(|name| match name {
            "glFenceSync" => fence as *const (),
            "glFlush" => flush as *const (),
            "glFinish" => finish as *const (),
            "glWaitSync" => wait as *const (),
            "glClientWaitSync" => client_wait as *const (),
            "glDeleteSync" => delete as *const (),
            _ => std::ptr::null(),
        } as *const _)
    }

    fn command(context: usize, command: Command) {
        DRIVER.with_borrow_mut(|driver| {
            driver.current = context;
            driver.queue(command);
        });
    }

    #[test]
    fn same_context_updates_need_no_texture_fences() {
        let gl = gl();
        let mut sync = TextureSync::default();
        command(0, Command::Write(1));
        sync.update_write(&gl, false, true);
        sync.wait_for_upload(&gl);
        command(0, Command::Read(1));
        sync.update_read(&gl, false, true);
        sync.wait_for_all(&gl);
        command(0, Command::Write(2));
        sync.update_write(&gl, false, true);
        command(0, Command::Read(2));
        finish();
        DRIVER.with_borrow(|driver| assert_eq!(driver.fence_calls, 0));
    }

    #[test]
    fn shared_readers_must_complete_before_next_writer() {
        let gl = gl();
        let mut sync = TextureSync::default();
        command(0, Command::Write(1));
        sync.update_write(&gl, true, true);
        for context in [1, 2] {
            DRIVER.with_borrow_mut(|driver| driver.current = context);
            sync.wait_for_upload(&gl);
            command(context, Command::Read(1));
            sync.update_read(&gl, true, true);
        }
        DRIVER.with_borrow_mut(|driver| driver.current = 0);
        sync.wait_for_all(&gl);
        command(0, Command::Write(2));
        sync.update_write(&gl, true, true);
        // Execute the writer first: the real synchronization must pull in both
        // readers and their producer, rather than depending on scheduling luck.
        finish();
        DRIVER.with_borrow(|driver| {
            assert_eq!(driver.value, 2);
            for queue in &driver.queues {
                assert_eq!(queue.completed, queue.submitted);
            }
        });
    }

    #[test]
    fn failed_reader_fence_still_completes_previous_readers() {
        let gl = gl();
        let mut sync = TextureSync::default();
        command(0, Command::Write(3));
        sync.update_write(&gl, true, true);
        DRIVER.with_borrow_mut(|driver| driver.current = 1);
        sync.wait_for_upload(&gl);
        command(1, Command::Read(3));
        sync.update_read(&gl, true, true);
        DRIVER.with_borrow_mut(|driver| {
            driver.current = 2;
            driver.fail_fence = true;
        });
        sync.wait_for_upload(&gl);
        command(2, Command::Read(3));
        sync.update_read(&gl, true, true);
        assert!(sync.read_sync.lock().unwrap().is_none());
        DRIVER.with_borrow(|driver| {
            for queue in &driver.queues {
                assert_eq!(queue.completed, queue.commands.len());
            }
        });
    }

    #[test]
    fn missing_or_failed_fences_complete_accesses_synchronously() {
        for fencing in [false, true] {
            let gl = gl();
            DRIVER.with_borrow_mut(|driver| driver.fail_fence = fencing);
            let mut sync = TextureSync::default();
            command(0, Command::Write(7));
            sync.update_write(&gl, true, fencing);
            assert!(sync.write_sync.lock().unwrap().is_none());
            command(1, Command::Read(7));
            sync.update_read(&gl, true, fencing);
            assert!(sync.read_sync.lock().unwrap().is_none());
            DRIVER.with_borrow(|driver| {
                assert_eq!(driver.finishes, 2);
                assert_eq!(driver.fence_calls, if fencing { 2 } else { 0 });
                assert_eq!(driver.queues[1].completed, 1);
            });
        }
    }
}
