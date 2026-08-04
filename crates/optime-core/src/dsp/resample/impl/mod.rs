//! The resampler implementations, one module each, numbered by the order they were written. A file
//! here is a `Resampler` impl and nothing else; the trait, the gather, and the mode resolution all
//! live one level up, so adding an implementation means adding a module and a re-export.

pub mod resample_impl_1;
pub mod resample_impl_2;

pub use resample_impl_1::ResampleImpl1;
pub use resample_impl_2::ResampleImpl2;
