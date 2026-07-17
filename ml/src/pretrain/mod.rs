//! Self-supervised pretraining on *unlabeled* real game songs: teaches a trunk the *real*
//! note-event distribution, which warm-starts the supervised fine-tune ([`crate::train`]). Two
//! genuinely different pretexts, not merged: [`masked`] (BERT-style masked-frame MSE, m00 only —
//! needs the feature grid) and [`ar`] (AR next-frame BCE, generic over
//! [`crate::backbone::ArBackbone`], every learned-token generation).

pub mod ar;
pub mod masked;
