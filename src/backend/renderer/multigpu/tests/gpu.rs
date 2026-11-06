//! Hardware exerciser using only the pre-fix public API, so this module can be
//! included unchanged in v0.7.0 for a baseline comparison. The separate raw-damage
//! test supplies the new API through the drawing callback.
use crate::{
    backend::{
        allocator::{dmabuf::Dmabuf, gbm::GbmDevice, Fourcc},
        drm::{DrmDeviceFd, DrmNode},
        renderer::{
            gles::{Capability, GlesRenderbuffer, GlesRenderer},
            multigpu::{gbm::GbmGlesBackend, ApiDevice, GpuManager, MultiFrame},
            Bind, Color32F, ExportMem, Frame, ImportMem, Offscreen, Renderer,
        },
    },
    utils::{DeviceFd, Physical, Rectangle, Transform},
};
use std::{fs::File, os::fd::OwnedFd};

type Backend = GbmGlesBackend<GlesRenderer, DrmDeviceFd>;

/// Needs two different render nodes with GBM/EGL/GLES, a common renderable and
/// sampleable DMA-BUF format/modifier, and cross-device native-fence support.
/// No DRM master, compositor, display, or connector is required.
///
/// SMITHAY_TEST_RENDER_NODE=/dev/dri/renderD128
/// SMITHAY_TEST_TARGET_NODE=/dev/dri/renderD129
/// cargo test --lib real_gpu_shared_scratch -- --ignored --nocapture
#[test]
#[ignore = "requires two explicitly selected physical DRM render nodes"]
fn real_gpu_shared_scratch() {
    exercise(|frame, color, damage| frame.clear(color, damage).unwrap());
}

pub(super) fn exercise(
    draw: impl Fn(&mut MultiFrame<'_, '_, '_, '_, Backend, Backend>, Color32F, &[Rectangle<i32, Physical>]),
) {
    let source_path = std::env::var("SMITHAY_TEST_RENDER_NODE").expect("set SMITHAY_TEST_RENDER_NODE");
    let target_path = std::env::var("SMITHAY_TEST_TARGET_NODE").expect("set SMITHAY_TEST_TARGET_NODE");
    let source = DrmNode::from_path(&source_path).unwrap();
    let target = DrmNode::from_path(&target_path).unwrap();
    assert_ne!(source, target, "the cross-GPU exerciser needs distinct nodes");
    let mut backend = Backend::default();
    for (node, path) in [(source, source_path), (target, target_path)] {
        let file = File::options().read(true).write(true).open(path).unwrap();
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        backend.add_node(node, GbmDevice::new(fd).unwrap()).unwrap();
    }
    let mut manager = GpuManager::new(backend).unwrap();
    for device in manager.devices().unwrap() {
        assert!(
            device
                .renderer()
                .capabilities()
                .contains(&Capability::ExportFence),
            "requires native-fence export on both devices; synchronous finish would conceal this race"
        );
    }
    let format = Fourcc::Abgr8888;
    let size = (512, 512).into();
    let full = Rectangle::from_size(size);
    let left = Rectangle::new((0, 0).into(), (256, 512).into());
    let mut buffers: Vec<Dmabuf>;
    let mut busy: GlesRenderbuffer;
    let busy_texture;
    {
        let mut devices = manager.devices_mut().unwrap();
        let device = devices.find(|device| *device.node() == target).unwrap();
        let modifiers: Vec<_> = Bind::<Dmabuf>::supported_formats(device.renderer())
            .unwrap()
            .into_iter()
            .filter(|candidate| candidate.code == format)
            .map(|candidate| candidate.modifier)
            .collect();
        buffers = (0..4)
            .map(|_| {
                device
                    .allocator()
                    .create_buffer(512, 512, format, &modifiers)
                    .unwrap()
            })
            .collect();
        busy = device
            .renderer_mut()
            .create_buffer(format, (2048, 2048).into())
            .unwrap();
        busy_texture = device
            .renderer_mut()
            .import_memory(&[19, 71, 137, 255], format, (1, 1).into(), false)
            .unwrap();
    }
    let rounds = std::env::var("SMITHAY_TEST_ROUNDS")
        .ok()
        .map(|value| value.parse().unwrap())
        .unwrap_or(32usize);
    let load = std::env::var("SMITHAY_TEST_TARGET_LOAD")
        .ok()
        .map(|value| value.parse().unwrap())
        .unwrap_or(128usize);
    for round in 0..rounds {
        // Queue independent target work without a CPU wait, delaying subsequent
        // reads of the shared source while the source GPU can submit overwrites.
        {
            let mut devices = manager.devices_mut().unwrap();
            let renderer = devices
                .find(|device| *device.node() == target)
                .unwrap()
                .renderer_mut();
            let mut fb = renderer.bind(&mut busy).unwrap();
            let mut frame = renderer
                .render(&mut fb, (2048, 2048).into(), Transform::Normal)
                .unwrap();
            for _ in 0..load {
                Frame::render_texture_from_to(
                    &mut frame,
                    &busy_texture,
                    Rectangle::from_size((1, 1).into()).to_f64(),
                    Rectangle::from_size((2048, 2048).into()),
                    &[Rectangle::from_size((2048, 2048).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .unwrap();
            }
            let _ = frame.finish().unwrap();
        }
        let mut expected = Vec::new();
        for (index, buffer) in buffers.iter_mut().enumerate() {
            let base = [((round * 23 + index * 51) % 220 + 16) as u8, 41, 193, 255];
            let patch = [23, ((round * 31 + index * 43) % 220 + 16) as u8, 67, 255];
            let mut renderer = manager.renderer(&source, &target, format).unwrap();
            let mut fb = renderer.bind(buffer).unwrap();
            let mut frame = renderer.render(&mut fb, size, Transform::Normal).unwrap();
            assert!(
                frame.target.as_ref().unwrap().texture.is_some(),
                "requires GPU DMA-BUF copy; CPU fallback would conceal this race"
            );
            draw(
                &mut frame,
                base.map(|component| component as f32 / 255.0).into(),
                &[full],
            );
            if index % 2 == 0 {
                let _ = frame.finish().unwrap();
            } else {
                drop(frame);
            }
            expected.push((base, patch));
        }
        // Revisit every same-sized output after scratch has held other outputs.
        for (index, buffer) in buffers.iter_mut().enumerate() {
            let mut renderer = manager.renderer(&source, &target, format).unwrap();
            let mut fb = renderer.bind(buffer).unwrap();
            let mut frame = renderer.render(&mut fb, size, Transform::Normal).unwrap();
            draw(
                &mut frame,
                expected[index].1.map(|component| component as f32 / 255.0).into(),
                &[left],
            );
            if index % 2 == 0 {
                drop(frame);
            } else {
                let _ = frame.finish().unwrap();
            }
        }
        // Readback is intentionally postponed until all source writes and target
        // reads are queued. Waiting after each frame would conceal the race.
        for (index, buffer) in buffers.iter_mut().enumerate() {
            let mut renderer = manager.single_renderer(&target).unwrap();
            let fb = renderer.bind(buffer).unwrap();
            let mapping = renderer
                .copy_framebuffer(&fb, Rectangle::from_size((512, 512).into()), format)
                .unwrap();
            let pixels = renderer.map_texture(&mapping).unwrap();
            assert_eq!(pixels.len(), 512 * 512 * 4);
            for (pixel_index, pixel) in pixels.chunks_exact(4).enumerate() {
                let wanted = if pixel_index % 512 < 256 {
                    expected[index].1
                } else {
                    expected[index].0
                };
                assert_eq!(
                    pixel, wanted,
                    "round {round}, output {index}, pixel {pixel_index}"
                );
            }
        }
    }
}
