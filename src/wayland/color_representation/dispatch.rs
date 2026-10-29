use wayland_protocols::wp::color_representation::v1::server::{
    wp_color_representation_manager_v1::{self, WpColorRepresentationManagerV1},
    wp_color_representation_surface_v1::{self, WpColorRepresentationSurfaceV1},
};
use wayland_server::{
    backend::ClientId, protocol::wl_surface::WlSurface, Client, DataInit, Dispatch, DisplayHandle,
    GlobalDispatch, New, Resource, WEnum,
};

use super::{
    AlphaMode, Coefficients, ColorRepresentationState, ColorRepresentationSurfaceCachedState,
    ColorRepresentationSurfaceData, ColorRepresentationSurfaceUserData, SUPPORTED,
};
use crate::backend::allocator::{
    dmabuf::{is_yuv, ChromaLocation},
    Buffer as _,
};
use crate::wayland::compositor::{self, BufferAssignment, SurfaceAttributes};

impl<D> GlobalDispatch<WpColorRepresentationManagerV1, (), D> for ColorRepresentationState
where
    D: GlobalDispatch<WpColorRepresentationManagerV1, ()>,
    D: Dispatch<WpColorRepresentationManagerV1, ()>,
    D: Dispatch<WpColorRepresentationSurfaceV1, ColorRepresentationSurfaceUserData>,
    D: 'static,
{
    fn bind(
        _state: &mut D,
        _: &DisplayHandle,
        _: &Client,
        resource: New<WpColorRepresentationManagerV1>,
        _: &(),
        data_init: &mut DataInit<'_, D>,
    ) {
        let manager = data_init.init(resource, ());
        manager.supported_alpha_mode(AlphaMode::PremultipliedElectrical);
        for (coefficients, range) in SUPPORTED {
            manager.supported_coefficients_and_ranges(coefficients, range);
        }
        manager.done();
    }
}

impl<D> Dispatch<WpColorRepresentationManagerV1, (), D> for ColorRepresentationState
where
    D: Dispatch<WpColorRepresentationManagerV1, ()>,
    D: Dispatch<WpColorRepresentationSurfaceV1, ColorRepresentationSurfaceUserData>,
    D: 'static,
{
    fn request(
        _state: &mut D,
        _: &Client,
        manager: &WpColorRepresentationManagerV1,
        request: wp_color_representation_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            wp_color_representation_manager_v1::Request::GetSurface { id, surface } => {
                let taken = compositor::with_states(&surface, |states| {
                    let data = states
                        .data_map
                        .get_or_insert_threadsafe(ColorRepresentationSurfaceData::default);
                    let taken = data.hook.lock().unwrap().is_some();
                    taken
                });
                if taken {
                    manager.post_error(
                        wp_color_representation_manager_v1::Error::SurfaceExists,
                        "wl_surface already has a color representation object",
                    );
                    return;
                }
                let object = data_init.init(id, ColorRepresentationSurfaceUserData(surface.downgrade()));
                let hook = compositor::add_pre_commit_hook::<D, _>(&surface, move |_, _, surface| {
                    check_pixel_format(&object, surface);
                });
                compositor::with_states(&surface, |states| {
                    let data = states.data_map.get::<ColorRepresentationSurfaceData>().unwrap();
                    *data.hook.lock().unwrap() = Some(hook);
                });
            }
            wp_color_representation_manager_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }
}

impl<D> Dispatch<WpColorRepresentationSurfaceV1, ColorRepresentationSurfaceUserData, D>
    for ColorRepresentationState
where
    D: Dispatch<WpColorRepresentationSurfaceV1, ColorRepresentationSurfaceUserData>,
{
    fn request(
        _state: &mut D,
        _: &Client,
        obj: &WpColorRepresentationSurfaceV1,
        request: wp_color_representation_surface_v1::Request,
        data: &ColorRepresentationSurfaceUserData,
        _dh: &DisplayHandle,
        _: &mut DataInit<'_, D>,
    ) {
        use wp_color_representation_surface_v1::{Error, Request};

        // Destroying unsets everything (double-buffered) and frees the
        // surface for another object; once the surface is gone, there is
        // nothing to unset.
        if let Request::Destroy = request {
            if let Some(surface) = data.wl_surface() {
                let hook = compositor::with_states(&surface, |states| {
                    *states
                        .cached_state
                        .get::<ColorRepresentationSurfaceCachedState>()
                        .pending() = ColorRepresentationSurfaceCachedState::default();
                    states
                        .data_map
                        .get::<ColorRepresentationSurfaceData>()
                        .and_then(|d| d.hook.lock().unwrap().take())
                });
                // Outside `with_states`, whose lock removing a hook takes.
                if let Some(hook) = hook {
                    compositor::remove_pre_commit_hook(&surface, hook);
                }
            }
            return;
        }

        let Some(surface) = data.wl_surface() else {
            obj.post_error(Error::Inert, "wl_surface was destroyed");
            return;
        };
        let set = |f: &dyn Fn(&mut ColorRepresentationSurfaceCachedState)| {
            compositor::with_states(&surface, |states| {
                f(states
                    .cached_state
                    .get::<ColorRepresentationSurfaceCachedState>()
                    .pending())
            })
        };

        match request {
            Request::SetAlphaMode { alpha_mode } => match alpha_mode {
                WEnum::Value(mode @ AlphaMode::PremultipliedElectrical) => {
                    set(&|s| s.alpha_mode = Some(mode))
                }
                _ => obj.post_error(Error::AlphaMode, "alpha mode not advertised"),
            },
            Request::SetCoefficientsAndRange { coefficients, range } => match (coefficients, range) {
                (WEnum::Value(coefficients), WEnum::Value(range))
                    if SUPPORTED.contains(&(coefficients, range)) =>
                {
                    set(&|s| s.coefficients_and_range = Some((coefficients, range)))
                }
                _ => obj.post_error(
                    Error::Coefficients,
                    "matrix coefficients and range not advertised",
                ),
            },
            Request::SetChromaLocation { chroma_location } => {
                use wp_color_representation_surface_v1::ChromaLocation as Wire;
                let location = match chroma_location {
                    WEnum::Value(Wire::Type0) => ChromaLocation::Type0,
                    WEnum::Value(Wire::Type1) => ChromaLocation::Type1,
                    WEnum::Value(Wire::Type2) => ChromaLocation::Type2,
                    WEnum::Value(Wire::Type3) => ChromaLocation::Type3,
                    WEnum::Value(Wire::Type4) => ChromaLocation::Type4,
                    WEnum::Value(Wire::Type5) => ChromaLocation::Type5,
                    _ => {
                        obj.post_error(Error::ChromaLocation, "invalid chroma location");
                        return;
                    }
                };
                set(&|s| s.chroma_location = Some(location))
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(
        _state: &mut D,
        _client: ClientId,
        _object: &WpColorRepresentationSurfaceV1,
        _data: &ColorRepresentationSurfaceUserData,
    ) {
        // A graceful destroy is handled above, double-buffered; a client
        // gone takes its surfaces' state with it.
    }
}

/// At commit: the buffer committed with a color representation must be of
/// the family it describes, R′G′B′ for identity and Y′CbCr otherwise.
fn check_pixel_format(object: &WpColorRepresentationSurfaceV1, surface: &WlSurface) {
    let (buffer, representation) = compositor::with_states(surface, |states| {
        let buffer = match &states.cached_state.get::<SurfaceAttributes>().pending().buffer {
            Some(BufferAssignment::NewBuffer(buffer)) => Some(buffer.clone()),
            _ => None,
        };
        let representation = *states
            .cached_state
            .get::<ColorRepresentationSurfaceCachedState>()
            .pending();
        (buffer, representation)
    });
    let Some(buffer) = buffer else { return };
    if representation == ColorRepresentationSurfaceCachedState::default() {
        return;
    }
    let format = if let Ok(dmabuf) = crate::wayland::dmabuf::get_dmabuf(&buffer) {
        Some(dmabuf.format().code)
    } else {
        crate::wayland::shm::with_buffer_contents(&buffer, |_, _, data| {
            crate::wayland::shm::shm_format_to_fourcc(data.format)
        })
        .ok()
        .flatten()
    };
    // Buffers of no known format (single-pixel ones, say) aren't checked.
    let Some(format) = format else { return };
    let yuv = is_yuv(format);
    let fits = match representation.coefficients_and_range {
        Some((Coefficients::Identity, _)) => !yuv,
        Some(_) => yuv,
        None => true,
    } && (representation.chroma_location.is_none() || yuv);
    if !fits {
        object.post_error(
            wp_color_representation_surface_v1::Error::PixelFormat,
            format!("color representation doesn't fit the buffer's format, {format:?}"),
        );
    }
}
