//! Self-supervised pretraining on *unlabeled* real game songs (harvested via
//! [`crate::harvest`]). No chord/key labels are used here; these stages only teach a
//! trunk the *real* note-event distribution, which then warm-starts the supervised
//! fine-tune ([`crate::train`]) whose labels come from synthetic theory.
//!
//! Two genuinely different pretexts — not one loop in disguise, so they are not
//! forced behind a single abstraction:
//!
//! * [`masked`] — BERT-style masked-frame reconstruction (MSE on pitch-class blocks).
//!   Generation 00 only: it needs the hand-engineered feature grid to mask and
//!   reconstruct.
//! * [`ar`] — autoregressive next-frame prediction (BCE on sounding pitch-classes +
//!   channels). Generic over [`crate::backbone::ArBackbone`], so it serves every
//!   learned-token generation.

pub mod ar;
pub mod masked;
