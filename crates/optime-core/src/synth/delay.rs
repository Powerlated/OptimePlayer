//! A fixed-length delay line used to widen the stereo image (Haas effect).

/// A fixed-length delay line used to widen the stereo image (Haas effect).
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f64>,
    pos_out: usize,
    delay: usize,
    /// Output gain.
    pub gain: f64,
}

impl DelayLine {
    /// Creates a delay line able to hold up to `max_length` samples.
    pub fn new(max_length: usize) -> Self {
        Self {
            buffer: vec![0.0; max_length.max(1)],
            pos_out: 0,
            delay: 0,
            gain: 1.0,
        }
    }

    /// Pushes `val` and returns the delayed (and gain-scaled) output sample.
    pub fn process(&mut self, val: f64) -> f64 {
        let len = self.buffer.len();
        self.buffer[(self.pos_out + self.delay) % len] = val;
        let out_val = self.buffer[self.pos_out];
        self.pos_out += 1;
        if self.pos_out >= len {
            self.pos_out = 0;
        }
        out_val * self.gain
    }

    /// Sets the delay length in samples (clamped to the buffer capacity).
    pub fn set_delay(&mut self, length: usize) {
        self.delay = length.min(self.buffer.len());
    }

    /// The current delay length in samples.
    pub fn delay(&self) -> usize {
        self.delay
    }
}
