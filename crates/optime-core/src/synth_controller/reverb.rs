//! MP2K (GBA) reverb — a float delay-line approximation of pokeemerald's `SoundMainRAM` reverb
//! pre-pass (`src/m4a_1.s:96`).
//!
//! On hardware the reverb runs inside the 8-bit PCM mixer: before each frame's DirectSound channels
//! are mixed, the buffer is seeded with `round((curL + curR + srcL + srcR) * reverb / 512)` — the
//! mono sum of the target buffer's stale content and the currently-playing (echo) buffer, scaled by
//! the song's 7-bit `reverb` amount. That makes it a **mono-summing feedback delay** one DMA buffer
//! (≈ one VBlank) long, with the reverb tail spread equally to both channels.
//!
//! Optime substitutes float voice mixing for that 8-bit PCM mixer (a documented backend seam), so
//! this is a structural approximation, not a bit-exact port: a single one-VBlank feedback comb on
//! the mono-summed sampled bus, feedback ≈ `reverb / 128` (the four `reverb/512` taps collapse to
//! one mono-summed prior output). It runs on the sampled (mixer-set) bus only, matching the
//! hardware where reverb touches the PCM buffer but not the CGB/PSG channels.

use crate::waveform::{Frame, Sample};

/// One GBA VBlank in seconds: `CYCLES_PER_FRAME / GBA_CLOCK_RATE` (280896 / 16_777_216 ≈ 16.74 ms),
/// the delay one DMA buffer of PCM represents.
const VBLANK_SECONDS: f64 = 280_896.0 / 16_777_216.0;

/// The mono feedback-delay reverb applied to the sampled bus.
#[derive(Debug, Clone)]
pub struct Reverb {
    /// Mono feedback buffer, one VBlank long at the current mixer rate. Holds the summed prior
    /// output so it decays by the feedback gain each pass.
    buf: Vec<Sample>,
    pos: usize,
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
            amount: 0,
            rate: 0.0,
        }
    }

    /// Sets the song reverb amount (0..=127 hardware units).
    pub fn set_amount(&mut self, amount: u8) {
        self.amount = amount;
    }

    /// (Re)sizes the delay buffer for `mixer_rate` (one VBlank of samples), clearing it. Called when
    /// the mixer sample rate changes; a no-op when the rate is unchanged.
    pub fn set_rate(&mut self, mixer_rate: f64) {
        if self.rate == mixer_rate {
            return;
        }
        let delay = (mixer_rate * VBLANK_SECONDS).round().max(1.0) as usize;
        self.buf = vec![0.0; delay];
        self.pos = 0;
        self.rate = mixer_rate;
    }

    /// The effective feedback gain: `amount / 128`, or 0 when disabled by `enabled`.
    #[inline]
    fn feedback(&self, enabled: bool) -> Sample {
        if enabled {
            Sample::from(self.amount) / 128.0
        } else {
            0.0
        }
    }

    /// Processes one mixer-rate stereo sample. When the stage is disabled (`enabled` false or amount
    /// 0) this is a bit-exact pass-through (the buffer is left untouched). Otherwise it adds the
    /// mono reverb tail to both channels and feeds the summed output back into the delay line.
    #[inline]
    pub fn process(&mut self, dry_l: Sample, dry_r: Sample, enabled: bool) -> Frame {
        let fb = self.feedback(enabled);
        if fb == 0.0 {
            return (dry_l, dry_r);
        }
        let delayed = self.buf[self.pos];
        let wet = delayed * fb;
        // Accumulate the mono output (dry + tail) so the echo decays by `fb` each VBlank.
        self.buf[self.pos] = (dry_l + dry_r) * 0.5 + wet;
        self.pos += 1;
        if self.pos >= self.buf.len() {
            self.pos = 0;
        }
        (dry_l + wet, dry_r + wet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn impulse_decays_at_one_vblank_spacing() {
        let mut r = Reverb::new();
        let rate = 13_379.0;
        r.set_rate(rate);
        r.set_amount(64); // fb = 0.5
        let delay = (rate * VBLANK_SECONDS).round() as usize;

        // One mono impulse, then silence.
        r.process(1.0, 1.0, true);
        let mut echoes = Vec::new();
        for i in 0..(delay * 3 + 5) {
            let (l, right) = r.process(0.0, 0.0, true);
            assert_eq!(l, right, "reverb tail must be mono (L == R)");
            if l != 0.0 {
                echoes.push((i + 1, l));
            }
        }
        // First echo lands one VBlank after the impulse, next at two VBlanks, decaying by fb.
        assert_eq!(echoes[0].0, delay);
        assert_eq!(echoes[1].0, delay * 2);
        assert!(echoes[1].1 < echoes[0].1 && echoes[1].1 > 0.0);
    }
}
