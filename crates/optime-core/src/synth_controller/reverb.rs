//! The GBA driver's echo, modelled from the timing of its DMA buffer.

#[cfg(test)]
use crate::waveform::Frame;
use crate::waveform::Sample;

const VBLANK_SECONDS: f64 = 280_896.0 / 16_777_216.0;

const PCM_DMA_PERIOD: usize = 7;

const REVERB_SHIFT_DIV: Sample = 512.0;

#[derive(Debug, Clone)]
pub struct Reverb {
    buf: Vec<Sample>,
    pos: usize,
    vblank_samples: usize,
    amount: u8,
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

    pub fn set_amount(&mut self, amount: u8) {
        self.amount = amount;
    }

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

    pub fn process_block(&mut self, l: &mut [Sample], r: &mut [Sample], enabled: bool) {
        if !enabled || self.amount == 0 {
            return;
        }
        let len = self.buf.len();
        let (vblank, gain) = (self.vblank_samples, Sample::from(self.amount));
        let mut pos = self.pos;
        for (l, r) in l.iter_mut().zip(r.iter_mut()) {
            let tap1 = self.buf[pos];
            let tap2 = self.buf[(pos + vblank) % len];
            let seed = (tap1 + tap2) * gain / REVERB_SHIFT_DIV;
            (*l, *r) = (*l + seed, *r + seed);
            self.buf[pos] = *l + *r;
            pos += 1;
            if pos >= len {
                pos = 0;
            }
        }
        self.pos = pos;
    }

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

    #[test]
    fn process_block_matches_per_sample() {
        for n in TEST_BLOCK_LENGTHS {
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
        assert_eq!(r.process(0.5, -0.25, false), (0.5, -0.25));
        r.set_amount(0);
        assert_eq!(r.process(0.5, -0.25, true), (0.5, -0.25));
    }

    #[test]
    fn echo_delay_is_a_full_pcm_buffer_not_one_vblank() {
        let mut r = Reverb::new();
        let rate = 13_379.0;
        r.set_rate(rate);
        r.set_amount(64);
        let vblank = (rate * VBLANK_SECONDS).round() as usize;
        let buffer = vblank * PCM_DMA_PERIOD;

        r.process(1.0, 1.0, true);
        let mut first_echo = None;
        for i in 1..(buffer + 5) {
            let (l, right) = r.process(0.0, 0.0, true);
            assert_eq!(l, right, "reverb tail must be mono (L == R)");
            if l != 0.0 && first_echo.is_none() {
                first_echo = Some(i);
            }
        }
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
        r.set_amount(64);
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
        assert_eq!(echoes[0], buffer - vblank);
        assert_eq!(echoes[1], buffer);
    }
}
