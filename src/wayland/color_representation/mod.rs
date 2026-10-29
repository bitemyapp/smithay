//! Implementation of the `wp_color_representation_v1` protocol
//!
//! Clients use it to say how their buffers' samples become R′G′B′: for
//! Y′CbCr formats, the matrix coefficients, quantization range and chroma
//! location (and for any format, the alpha mode).
//!
//! The protocol's state is double-buffered surface state
//! ([`ColorRepresentationSurfaceCachedState`]). Renderers importing a
//! surface's dmabuf through [`ImportDmaWl`](crate::backend::renderer::ImportDmaWl)
//! apply it to the buffer ([`Dmabuf::set_yuv_conversion`]), so
//! [`WaylandSurfaceRenderElement`](crate::backend::renderer::element::surface::WaylandSurfaceRenderElement)s
//! sample Y′CbCr buffers with it, and the
//! [`DrmCompositor`](crate::backend::drm::compositor::DrmCompositor) shows
//! them on planes with it.
//!
//! The global advertises every matrix and range [`YuvConversion`] can
//! express (BT.601, BT.709 and BT.2020, each in limited and full range),
//! identity for R′G′B′ formats, and premultiplied electrical alpha, which
//! is what smithay's renderers assume.
//!
//! ### Example
//!
//! ```no_run
//! # extern crate wayland_server;
//! use smithay::{delegate_color_representation, wayland::color_representation::ColorRepresentationState};
//!
//! # struct State;
//! # let mut display = wayland_server::Display::<State>::new().unwrap();
//! ColorRepresentationState::new::<State>(&display.handle());
//! delegate_color_representation!(State);
//! ```
//!
//! [`Dmabuf::set_yuv_conversion`]: crate::backend::allocator::dmabuf::Dmabuf::set_yuv_conversion

use std::sync::Mutex;

use wayland_protocols::wp::color_representation::v1::server::{
    wp_color_representation_manager_v1::WpColorRepresentationManagerV1,
    wp_color_representation_surface_v1::{self, WpColorRepresentationSurfaceV1},
};
use wayland_server::{
    backend::GlobalId, protocol::wl_surface::WlSurface, Dispatch, DisplayHandle, GlobalDispatch, Weak,
};

use super::compositor::{Cacheable, HookId, SurfaceData};
use crate::backend::allocator::dmabuf::{ChromaLocation, YuvConversion, YuvMatrix, YuvRange};

mod dispatch;

pub use wp_color_representation_surface_v1::{AlphaMode, Coefficients, Range};

/// The matrix coefficients and ranges the global advertises
const SUPPORTED: [(Coefficients, Range); 7] = [
    (Coefficients::Identity, Range::Full),
    (Coefficients::Bt601, Range::Limited),
    (Coefficients::Bt601, Range::Full),
    (Coefficients::Bt709, Range::Limited),
    (Coefficients::Bt709, Range::Full),
    (Coefficients::Bt2020, Range::Limited),
    (Coefficients::Bt2020, Range::Full),
];

/// A surface's color representation, double-buffered surface state
///
/// ```no_run
/// use smithay::wayland::compositor;
/// use smithay::wayland::color_representation::ColorRepresentationSurfaceCachedState;
///
/// # let wl_surface = todo!();
/// compositor::with_states(&wl_surface, |states| {
///     let mut state = states.cached_state.get::<ColorRepresentationSurfaceCachedState>();
///     dbg!(state.current().coefficients_and_range());
/// });
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorRepresentationSurfaceCachedState {
    coefficients_and_range: Option<(Coefficients, Range)>,
    chroma_location: Option<ChromaLocation>,
    alpha_mode: Option<AlphaMode>,
}

impl ColorRepresentationSurfaceCachedState {
    /// The matrix coefficients and range set, if any
    pub fn coefficients_and_range(&self) -> Option<(Coefficients, Range)> {
        self.coefficients_and_range
    }

    /// The chroma location set, if any
    pub fn chroma_location(&self) -> Option<ChromaLocation> {
        self.chroma_location
    }

    /// The alpha mode set, if any (premultiplied electrical when unset)
    pub fn alpha_mode(&self) -> Option<AlphaMode> {
        self.alpha_mode
    }

    /// How the surface's Y′CbCr buffers convert to R′G′B′, if it said
    pub fn yuv_conversion(&self) -> Option<YuvConversion> {
        if self.coefficients_and_range.is_none() && self.chroma_location.is_none() {
            return None;
        }
        let mut yuv = YuvConversion {
            chroma_location: self.chroma_location,
            ..YuvConversion::default()
        };
        if let Some((coefficients, range)) = self.coefficients_and_range {
            yuv.matrix = match coefficients {
                Coefficients::Bt709 => YuvMatrix::Bt709,
                Coefficients::Bt2020 => YuvMatrix::Bt2020,
                _ => YuvMatrix::Bt601,
            };
            yuv.range = match range {
                Range::Full => YuvRange::Full,
                _ => YuvRange::Limited,
            };
        }
        Some(yuv)
    }
}

impl Cacheable for ColorRepresentationSurfaceCachedState {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        *self
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        *into = self;
    }
}

/// How a surface's Y′CbCr buffers convert to R′G′B′, from its current color
/// representation; none if it has said nothing.
pub fn yuv_conversion(states: &SurfaceData) -> Option<YuvConversion> {
    if !states.cached_state.has::<ColorRepresentationSurfaceCachedState>() {
        return None;
    }
    states
        .cached_state
        .get::<ColorRepresentationSurfaceCachedState>()
        .current()
        .yuv_conversion()
}

/// What a surface keeps of its color representation object
#[derive(Debug, Default)]
struct ColorRepresentationSurfaceData {
    /// The hook checking committed buffers' formats, while an object exists
    hook: Mutex<Option<HookId>>,
}

/// User data of [`WpColorRepresentationSurfaceV1`] objects
#[derive(Debug)]
pub struct ColorRepresentationSurfaceUserData(Weak<WlSurface>);

impl ColorRepresentationSurfaceUserData {
    fn wl_surface(&self) -> Option<WlSurface> {
        self.0.upgrade().ok()
    }
}

/// Delegate type for the [`WpColorRepresentationManagerV1`] global
#[derive(Debug)]
pub struct ColorRepresentationState {
    global: GlobalId,
}

impl ColorRepresentationState {
    /// Register a new [`WpColorRepresentationManagerV1`] global
    pub fn new<D>(display: &DisplayHandle) -> ColorRepresentationState
    where
        D: GlobalDispatch<WpColorRepresentationManagerV1, ()>
            + Dispatch<WpColorRepresentationManagerV1, ()>
            + Dispatch<WpColorRepresentationSurfaceV1, ColorRepresentationSurfaceUserData>
            + 'static,
    {
        let global = display.create_global::<D, WpColorRepresentationManagerV1, _>(1, ());
        ColorRepresentationState { global }
    }

    /// Returns the [`WpColorRepresentationManagerV1`] global id
    pub fn global(&self) -> GlobalId {
        self.global.clone()
    }
}

/// Macro to delegate implementation of the color representation protocol
#[macro_export]
macro_rules! delegate_color_representation {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        type __WpColorRepresentationManagerV1 =
            $crate::reexports::wayland_protocols::wp::color_representation::v1::server::wp_color_representation_manager_v1::WpColorRepresentationManagerV1;
        type __WpColorRepresentationSurfaceV1 =
            $crate::reexports::wayland_protocols::wp::color_representation::v1::server::wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1;

        $crate::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty:
            [
                __WpColorRepresentationManagerV1: ()
            ] => $crate::wayland::color_representation::ColorRepresentationState
        );

        $crate::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty:
            [
                __WpColorRepresentationManagerV1: ()
            ] => $crate::wayland::color_representation::ColorRepresentationState
        );

        $crate::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty:
            [
                __WpColorRepresentationSurfaceV1: $crate::wayland::color_representation::ColorRepresentationSurfaceUserData
            ] => $crate::wayland::color_representation::ColorRepresentationState
        );
    };
}
