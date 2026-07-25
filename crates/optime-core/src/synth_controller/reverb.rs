//! MP2K (GBA) reverb — a float delay-line port of pokeemerald's `SoundMainRAM` reverb pre-pass
//! (`src/m4a_1.s:96`).
//!
//! On hardware the reverb runs inside the 8-bit PCM mixer, seeding each frame's buffer segment
//! *before* the DirectSound channels are mixed into it:
//!
//! ```c
//! // dst = the ring segment about to be re-mixed this VBlank; src = the NEXT segment (dst + V).
//! for (i = 0; i < pcmSamplesPerVBlank; i++) {
//!     s32 sum = dstL[i] + dstR[i] + srcL[i] + srcR[i];   // L+R of two stale segments
//!     s32 seed = (sum * reverb) >> 9;                    // * reverb / 512
//!     dstL[i] = dstR[i] = seed;                          // mono seed to both channels
//! }
//! // ...then the mixer ADDS the dry DirectSound channels on top of `seed`.
//! ```
//!
//! The PCM buffer is a ring of `pcmDmaPeriod` segments of `pcmSamplesPerVBlank` samples each
//! (`1584 / 224 = 7` segments of `224` at the fixed 13379 Hz mix rate). A segment's content is
//! re-read exactly one full ring later, so **the echo delay is one whole buffer — `7 × 224 = 1568`
//! samples ≈ 117 ms — not one VBlank (~16.7 ms)**. `dst` is that full-buffer-old tap; `src` (the
//! next segment) is one VBlank more recent, giving a **two-tap feedback** `≈ 117 ms` and `≈ 100 ms`.
//!
//! Optime substitutes float voice mixing for the 8-bit PCM mixer (a documented backend seam), so
//! this reproduces that structure on the mono-summed sampled bus rather than being bit-exact: a
//! circular buffer of the L+R output sum, two feedback taps at the full-buffer and full-buffer-minus-
//! one-VBlank delays, `seed = (tap1 + tap2) * reverb / 512` added to both channels. It runs on the
//! sampled (mixer-set) bus only, matching the hardware where reverb touches the PCM buffer but not
//! the CGB/PSG channels.

#[cfg(test)]
use crate::waveform::Frame;
use crate::waveform::Sample;

/// One GBA VBlank in seconds: `CYCLES_PER_FRAME / GBA_CLOCK_RATE` (280896 / 16_777_216 ≈ 16.74 ms).
/// `pcmSamplesPerVBlank` at the mix rate is `mixer_rate * VBLANK_SECONDS` (≈ 224 at 13379 Hz).
const VBLANK_SECONDS: f64 = 280_896.0 / 16_777_216.0;

/// `pcmDmaPeriod` — the number of VBlanks (ring segments) the PCM buffer spans:
/// `PCM_DMA_BUF_SIZE / pcmSamplesPerVBlank = 1584 / 224 = 7`. The reverb's echo delay is this many
/// VBlanks (one full ring), which is the whole point of the effect's long slap-back character.
const PCM_DMA_PERIOD: usize = 7;

/// The `reverb`-sum shift from `m4a_1.s`: `(dstL + dstR + srcL + srcR) * reverb >> 9`.
const REVERB_SHIFT_DIV: Sample = 512.0;

/// The MP2K two-tap feedback-delay reverb applied to the sampled bus.
#[derive(Debug, Clone)]
pub struct Reverb {
    /// Circular buffer of the mono (L+R) output sum, one full PCM ring long (`PCM_DMA_PERIOD`
    /// VBlanks). A slot is re-read one whole ring after it is written, so it is the long echo tap.
    buf: Vec<Sample>,
    /// Write cursor into `buf`.
    pos: usize,
    /// Samples per VBlank at the current mix rate — the offset to the second (nearer) tap.
    vblank_samples: usize,
    /// The 7-bit song reverb amount (`soundInfo.reverb`); `0` disables the stage.
    amount: u8,
    /// The mixer rate `buf` was sized for (0 = not yet configured).
    rate: f64,
}

impl Reverb {
    pub fn new() -> Self {
        Self {
            buf: vec![0.0; 1],
            pos: 0,
            vblank_samples: 1,
            amount: 0,
            rate: 0.0,
        }
    }

    /// Sets the song reverb amount (0..=127 hardware units).
    pub fn set_amount(&mut self, amount: u8) {
        self.amount = amount;
    }

    /// (Re)sizes the delay ring for `mixer_rate` (one full PCM buffer = `PCM_DMA_PERIOD` VBlanks of
    /// samples), clearing it. Called when the mixer sample rate changes; a no-op when unchanged.
    pub fn set_rate(&mut self, mixer_rate: f64) {
        if self.rate == mixer_rate {
            return;
        }
        let vblank = (mixer_rate * VBLANK_SECONDS).round().max(1.0) as usize;
        self.vblank_samples = vblank;
        self.buf = vec![0.0; vblank * PCM_DMA_PERIOD];
        self.pos = 0;
        self.rate = mixer_rate;
    }

    /// Processes a block of consecutive mixer-rate stereo samples in place. When the stage is
    /// disabled (`enabled` false or amount 0) this is a bit-exact pass-through (the ring is left
    /// untouched). Otherwise it adds the mono reverb tail (two full-buffer-delayed taps) to both
    /// channels and feeds the summed output back.
    ///
    /// The taps sit a whole PCM buffer behind the write cursor — thousands of samples at any real
    /// mixer rate — so a block's writes can never reach its own taps. The samples are still walked
    /// one at a time because the second tap is only one VBlank ahead of the cursor and a very small
    /// ring (a degenerate rate) could bring the two within a block of each other; the gain here is
    /// hoisting the amount, the tap offset and the disabled check out of the loop.
    pub fn process_block(&mut self, l: &mut [Sample], r: &mut [Sample], enabled: bool) {
        if !enabled || self.amount == 0 {
            return;
        }
        let len = self.buf.len();
        let (vblank, gain) = (self.vblank_samples, Sample::from(self.amount));
        let mut pos = self.pos;
        for (l, r) in l.iter_mut().zip(r.iter_mut()) {
            // Tap 1 = this slot's content from one full ring ago (`dst`, delay = whole buffer).
            // Tap 2 = the slot one VBlank ahead (`src`, delay = buffer − one VBlank), still unread
            // this cycle. Both are L+R sums, matching the hardware's `dstL+dstR + srcL+srcR`.
            let tap1 = self.buf[pos];
            let tap2 = self.buf[(pos + vblank) % len];
            let seed = (tap1 + tap2) * gain / REVERB_SHIFT_DIV;
            (*l, *r) = (*l + seed, *r + seed);
            // Store the L+R sum of what this slot now holds (mono seed on both channels + stereo dry).
            self.buf[pos] = *l + *r;
            pos += 1;
            if pos >= len {
                pos = 0;
            }
        }
        self.pos = pos;
    }

    /// Processes one mixer-rate stereo sample. A one-sample [`Self::process_block`], which is how
    /// the impulse-response tests below walk the delay taps.
    #[cfg(test)]
    fn process(&mut self, dry_l: Sample, dry_r: Sample, enabled: bool) -> Frame {
        let (mut l, mut r) = ([dry_l], [dry_r]);
        self.process_block(&mut l, &mut r, enabled);
        (l[0], r[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::block::{TEST_BLOCK_LENGTHS, test_signal};

    /// A block of any length must give bit-identical results to processing one stereo sample at a
    /// time, including once the ring has filled and the taps start feeding back.
    #[test]
    fn process_block_matches_per_sample() {
        for n in TEST_BLOCK_LENGTHS {
            // Long enough to run well past the ~1568-sample echo delay, so the taps are live.
            let signal = test_signal(4000);
            let right: Vec<Sample> = signal.iter().map(|x| -0.4 * x).collect();
            let make = || {
                let mut r = Reverb::new();
                r.set_rate(13_379.0);
                r.set_amount(80);
                r
            };

            let mut blocked = make();
            let (mut got_l, mut got_r) = (signal.clone(), right.clone());
            for (l, r) in got_l.chunks_mut(n).zip(got_r.chunks_mut(n)) {
                blocked.process_block(l, r, true);
            }

            let mut per_sample = make();
            let (mut want_l, mut want_r) = (Vec::new(), Vec::new());
            for (&l, &r) in signal.iter().zip(&right) {
                let (l, r) = per_sample.process(l, r, true);
                want_l.push(l);
                want_r.push(r);
            }

            assert_eq!((got_l, got_r), (want_l, want_r), "block length {n}");
        }
    }

    #[test]
    fn disabled_is_passthrough() {
        let mut r = Reverb::new();
        r.set_rate(13_379.0);
        r.set_amount(80);
        // amount set but not enabled → untouched dry signal.
        assert_eq!(r.process(0.5, -0.25, false), (0.5, -0.25));
        // amount 0 → also pass-through.
        r.set_amount(0);
        assert_eq!(r.process(0.5, -0.25, true), (0.5, -0.25));
    }

    #[test]
    fn echo_delay_is_a_full_pcm_buffer_not_one_vblank() {
        let mut r = Reverb::new();
        let rate = 13_379.0;
        r.set_rate(rate);
        r.set_amount(64);
        let vblank = (rate * VBLANK_SECONDS).round() as usize; // ≈ 224
        let buffer = vblank * PCM_DMA_PERIOD; // ≈ 1568

        // One mono impulse, then silence.
        r.process(1.0, 1.0, true);
        let mut first_echo = None;
        for i in 1..(buffer + 5) {
            let (l, right) = r.process(0.0, 0.0, true);
            assert_eq!(l, right, "reverb tail must be mono (L == R)");
            if l != 0.0 && first_echo.is_none() {
                first_echo = Some(i);
            }
        }
        // The first tap (`src`) lands a full buffer minus one VBlank after the impulse — a ~100 ms
        // echo, i.e. many VBlanks out, NOT the ~16.7 ms one-VBlank comb the old port produced.
        assert_eq!(first_echo, Some(buffer - vblank));
        assert!(
            buffer - vblank > 5 * vblank,
            "echo delay must be many VBlanks, not one"
        );
    }

    #[test]
    fn two_taps_one_vblank_apart() {
        let mut r = Reverb::new();
        let rate = 13_379.0;
        r.set_rate(rate);
        r.set_amount(64); // seed gain = 2 taps * 64/512
        let vblank = (rate * VBLANK_SECONDS).round() as usize;
        let buffer = vblank * PCM_DMA_PERIOD;

        r.process(1.0, 1.0, true);
        let mut echoes = Vec::new();
        for i in 1..(buffer + 5) {
            let (l, _) = r.process(0.0, 0.0, true);
            if l != 0.0 {
                echoes.push(i);
            }
        }
        // Both taps of the single impulse fire: `src` at buffer−VBlank, `dst` at the full buffer.
        assert_eq!(echoes[0], buffer - vblank);
        assert_eq!(echoes[1], buffer);
    }
}
