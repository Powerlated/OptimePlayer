//! Small shared helpers: byte readers, a fixed-capacity ring buffer, and byte-pattern search.

/// Reads a little-endian `u8` at `offset`, returning 0 if out of bounds.
///
/// The original engine read through a `DataView`, which throws on OOB; in practice the
/// interpreter never reads past a track's `0xFF` terminator. Returning 0 keeps the audio
/// thread panic-free without changing observable behavior on well-formed data.
#[inline]
pub fn read_u8(data: &[u8], offset: usize) -> u8 {
    data.get(offset).copied().unwrap_or(0)
}

/// Reads a little-endian `u16` at `offset`, returning 0 if out of bounds.
#[inline]
pub fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from(read_u8(data, offset)) | (u16::from(read_u8(data, offset + 1)) << 8)
}

/// Reads a little-endian `u32` at `offset`, returning 0 if out of bounds.
#[inline]
pub fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from(read_u8(data, offset))
        | (u32::from(read_u8(data, offset + 1)) << 8)
        | (u32::from(read_u8(data, offset + 2)) << 16)
        | (u32::from(read_u8(data, offset + 3)) << 24)
}

/// Tests whether bit `bit` is set in `value`.
#[inline]
pub fn bit_test(value: u32, bit: u32) -> bool {
    value & (1 << bit) != 0
}

/// Finds every offset in `haystack` where `needle` occurs.
pub fn search_for_sequence(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() || haystack.len() < needle.len() {
        return out;
    }
    for i in 0..=(haystack.len() - needle.len()) {
        if &haystack[i..i + needle.len()] == needle {
            out.push(i);
        }
    }
    out
}

/// A fixed-capacity FIFO ring buffer mirroring the original engine's `CircularBuffer`.
///
/// Unlike [`std::collections::VecDeque`] this has a hard capacity; [`Self::insert`] reports
/// failure on overflow rather than growing, matching the hardware-style message queues.
#[derive(Debug, Clone)]
pub struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    read_pos: usize,
    write_pos: usize,
    entries: usize,
}

impl<T> CircularBuffer<T> {
    /// Creates a ring buffer holding up to `size` items.
    pub fn new(size: usize) -> Self {
        let mut buffer = Vec::with_capacity(size);
        buffer.resize_with(size, || None);
        Self {
            buffer,
            read_pos: 0,
            write_pos: 0,
            entries: 0,
        }
    }

    /// The maximum number of items this buffer can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// The number of items currently queued.
    #[inline]
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Whether the buffer holds no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Whether the buffer is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.entries == self.buffer.len()
    }

    /// Pushes `data` to the back. Returns `false` (and drops `data`) if full.
    pub fn insert(&mut self, data: T) -> bool {
        if self.entries < self.buffer.len() {
            self.entries += 1;
            self.buffer[self.write_pos] = Some(data);
            self.write_pos += 1;
            if self.write_pos >= self.buffer.len() {
                self.write_pos = 0;
            }
            true
        } else {
            false
        }
    }

    /// Pops the front item, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.entries > 0 {
            self.entries -= 1;
            let data = self.buffer[self.read_pos].take();
            self.read_pos += 1;
            if self.read_pos >= self.buffer.len() {
                self.read_pos = 0;
            }
            data
        } else {
            None
        }
    }

    /// Peeks at the item `offset` slots from the front without removing it.
    pub fn peek(&self, offset: usize) -> Option<&T> {
        if offset >= self.entries {
            return None;
        }
        self.buffer[(self.read_pos + offset) % self.buffer.len()].as_ref()
    }

    /// Empties the buffer.
    pub fn reset(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.read_pos = 0;
        self.write_pos = 0;
        self.entries = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian() {
        let data = [0x78, 0x56, 0x34, 0x12, 0xFF];
        assert_eq!(read_u8(&data, 0), 0x78);
        assert_eq!(read_u16(&data, 0), 0x5678);
        assert_eq!(read_u32(&data, 0), 0x1234_5678);
    }

    #[test]
    fn out_of_bounds_reads_zero() {
        let data = [0x01u8];
        assert_eq!(read_u8(&data, 5), 0);
        assert_eq!(read_u16(&data, 0), 0x0001);
        assert_eq!(read_u32(&data, 10), 0);
    }

    #[test]
    fn bit_test_works() {
        assert!(bit_test(0b1010, 1));
        assert!(!bit_test(0b1010, 0));
        assert!(bit_test(0x8000_0000, 31));
    }

    #[test]
    fn search_finds_all_occurrences() {
        let hay = b"abXYabZab";
        assert_eq!(search_for_sequence(hay, b"ab"), vec![0, 4, 7]);
        assert_eq!(search_for_sequence(hay, b"XY"), vec![2]);
        assert!(search_for_sequence(hay, b"qq").is_empty());
        assert!(search_for_sequence(hay, b"").is_empty());
    }

    #[test]
    fn circular_buffer_fifo_and_wrap() {
        let mut cb = CircularBuffer::new(3);
        assert!(cb.is_empty());
        assert!(cb.insert(1));
        assert!(cb.insert(2));
        assert!(cb.insert(3));
        assert!(cb.is_full());
        // Overflow is rejected, not grown.
        assert!(!cb.insert(4));
        assert_eq!(cb.entries(), 3);

        assert_eq!(cb.peek(0), Some(&1));
        assert_eq!(cb.peek(2), Some(&3));
        assert_eq!(cb.peek(3), None);

        assert_eq!(cb.pop(), Some(1));
        // Now there's room; wrap the write position.
        assert!(cb.insert(4));
        assert_eq!(cb.pop(), Some(2));
        assert_eq!(cb.pop(), Some(3));
        assert_eq!(cb.pop(), Some(4));
        assert_eq!(cb.pop(), None);
    }
}
