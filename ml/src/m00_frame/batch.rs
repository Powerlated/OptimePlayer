//! Generation 00's CPU-side batch: the hand-engineered feature grid plus flattened
//! labels.
//!
//! Shaped deliberately like the learned-token generations' batches
//! ([`crate::m01_event::EventBatchData`], [`crate::m02_hier::HierBatchData`]) —
//! flattened `chord_labels` / `key_labels` — so [`crate::shared`] builds targets and
//! scores metrics for all three the same way.

use burn::prelude::*;

use crate::data::Example;
use crate::features::FEATURE_DIM;
use crate::notes::Song;

/// Flattened feature grid + labels for a batch of equal-length songs.
pub struct FrameBatch {
    pub batch: usize,
    pub n_frames: usize,
    /// `batch * n_frames * FEATURE_DIM`, row-major.
    pub features: Vec<f32>,
    /// `batch * n_frames` joint chord labels.
    pub chord_labels: Vec<usize>,
    /// `batch` key labels.
    pub key_labels: Vec<usize>,
}

impl FrameBatch {
    /// Featurize a slice of songs (all must share `n_frames`).
    pub fn build(songs: &[Song]) -> FrameBatch {
        let batch = songs.len();
        let n_frames = songs.first().map(|s| s.n_frames).unwrap_or(0);
        let mut d = FrameBatch {
            batch,
            n_frames,
            features: Vec::with_capacity(batch * n_frames * FEATURE_DIM),
            chord_labels: Vec::with_capacity(batch * n_frames),
            key_labels: Vec::with_capacity(batch),
        };
        for song in songs {
            let ex = Example::from_song(song);
            d.features.extend_from_slice(&ex.features);
            d.chord_labels.extend_from_slice(&ex.chord_labels);
            d.key_labels.push(ex.key_label);
        }
        d
    }

    /// `[batch, n_frames, FEATURE_DIM]` input tensor.
    pub fn tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        Tensor::<B, 3>::from_data(
            TensorData::new(
                self.features.clone(),
                [self.batch, self.n_frames, FEATURE_DIM],
            ),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_songs;

    #[test]
    fn flattens_features_and_labels() {
        let songs = generate_songs(3, 32, 1);
        let b = FrameBatch::build(&songs);
        assert_eq!((b.batch, b.n_frames), (3, 32));
        assert_eq!(b.features.len(), 3 * 32 * FEATURE_DIM);
        assert_eq!(b.chord_labels.len(), 3 * 32);
        assert_eq!(b.key_labels.len(), 3);
    }
}
