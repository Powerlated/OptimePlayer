//! The list of resampler implementations under test, in one place. Every implementation under
//! `dsp/resample/impl/` appears here once, at each lane width worth timing, and both benchmarks walk
//! it — so adding an implementation and forgetting to benchmark it is not a thing that can happen,
//! and the two benchmarks can never disagree about who the contenders were.
//!
//! It is a visitor rather than a list of values because each entry is a *type*: the resampler is a
//! compile-time choice, so the roster hands each implementation to a caller that is itself generic
//! and collects whatever the caller makes of it.

use optime_core::{
    ResampleImplIir, ResampleImplPolyphase, ResampleImplSimd, ResampleImplSimdClosedForm, Resampler,
};

pub trait ResamplerVisitor {
    type Output;

    fn visit<R: Resampler + 'static>(&mut self, name: &'static str) -> Self::Output;
}

pub fn walk<V: ResamplerVisitor>(visitor: &mut V) -> Vec<V::Output> {
    vec![
        visitor.visit::<ResampleImplSimd<4>>("simd x4"),
        visitor.visit::<ResampleImplSimd<8>>("simd x8"),
        visitor.visit::<ResampleImplSimdClosedForm<4>>("closed x4"),
        visitor.visit::<ResampleImplSimdClosedForm<8>>("closed x8"),
        visitor.visit::<ResampleImplPolyphase<4>>("poly x4"),
        visitor.visit::<ResampleImplPolyphase<8>>("poly x8"),
        visitor.visit::<ResampleImplIir<4>>("iir x4"),
        visitor.visit::<ResampleImplIir<8>>("iir x8"),
    ]
}
