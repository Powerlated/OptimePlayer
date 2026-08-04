use crate::waveform::Sample;

#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<Sample>,
    pos_out: usize,
    delay: usize,
    pub gain: Sample,
}

impl DelayLine {
    pub fn new(max_length: usize) -> Self {
        Self {
            buffer: vec![0.0; max_length.max(1)],
            pos_out: 0,
            delay: 0,
            gain: 1.0,
        }
    }

    pub fn process_block(&mut self, block: &mut [Sample]) {
        let len = self.buffer.len();
        let n = block.len();
        if self.delay + n > len {
            for val in block.iter_mut() {
                *val = self.process_one(*val);
            }
            return;
        }
        let write_start = (self.pos_out + self.delay) % len;
        let first = (len - write_start).min(n);
        self.buffer[write_start..write_start + first].copy_from_slice(&block[..first]);
        self.buffer[..n - first].copy_from_slice(&block[first..]);

        let read_start = self.pos_out;
        let first = (len - read_start).min(n);
        block[..first].copy_from_slice(&self.buffer[read_start..read_start + first]);
        block[first..].copy_from_slice(&self.buffer[..n - first]);
        for val in block.iter_mut() {
            *val *= self.gain;
        }

        self.pos_out = (read_start + n) % len;
    }

    #[inline]
    pub fn process(&mut self, val: Sample) -> Sample {
        let mut block = [val];
        self.process_block(&mut block);
        block[0]
    }

    #[inline]
    fn process_one(&mut self, val: Sample) -> Sample {
        let len = self.buffer.len();
        self.buffer[(self.pos_out + self.delay) % len] = val;
        let out_val = self.buffer[self.pos_out];
        self.pos_out += 1;
        if self.pos_out >= len {
            self.pos_out = 0;
        }
        out_val * self.gain
    }

    pub fn set_delay(&mut self, length: usize) {
        self.delay = length.min(self.buffer.len());
    }

    pub fn set_capacity(&mut self, max_length: usize) {
        self.buffer = vec![0.0; max_length.max(1)];
        self.pos_out = 0;
        self.delay = self.delay.min(self.buffer.len());
    }

    pub fn delay(&self) -> usize {
        self.delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::block::{TEST_BLOCK_LENGTHS, test_signal};

    #[test]
    fn process_block_matches_per_sample() {
        for capacity in [1, 2, 7, 300, 4800] {
            for delay in [0, 1, 5, capacity / 2, capacity] {
                for n in TEST_BLOCK_LENGTHS {
                    let signal = test_signal(4 * n);
                    let make = || {
                        let mut d = DelayLine::new(capacity);
                        d.set_delay(delay);
                        d.gain = 0.75;
                        d
                    };

                    let mut blocked = make();
                    let mut got = signal.clone();
                    for chunk in got.chunks_mut(n) {
                        blocked.process_block(chunk);
                    }

                    let mut per_sample = make();
                    let want: Vec<_> = signal.iter().map(|&x| per_sample.process(x)).collect();

                    assert_eq!(got, want, "capacity {capacity}, delay {delay}, block {n}");
                }
            }
        }
    }
}
