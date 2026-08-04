//! Small shared helpers: bounds-checked binary reads, byte search, and a circular buffer.

#[inline]
pub fn read_u8(data: &[u8], offset: usize) -> u8 {
    data.get(offset).copied().unwrap_or(0)
}

#[inline]
pub fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from(read_u8(data, offset)) | (u16::from(read_u8(data, offset + 1)) << 8)
}

#[inline]
pub fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from(read_u8(data, offset))
        | (u32::from(read_u8(data, offset + 1)) << 8)
        | (u32::from(read_u8(data, offset + 2)) << 16)
        | (u32::from(read_u8(data, offset + 3)) << 24)
}

#[inline]
pub fn bit_test(value: u32, bit: u32) -> bool {
    value & (1 << bit) != 0
}

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

#[derive(Debug, Clone)]
pub struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    read_pos: usize,
    write_pos: usize,
    entries: usize,
    inserted: u64,
}

impl<T> CircularBuffer<T> {
    pub fn new(size: usize) -> Self {
        let mut buffer = Vec::with_capacity(size);
        buffer.resize_with(size, || None);
        Self {
            buffer,
            read_pos: 0,
            write_pos: 0,
            entries: 0,
            inserted: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn entries(&self) -> usize {
        self.entries
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.entries == self.buffer.len()
    }

    pub fn insert(&mut self, data: T) -> bool {
        if self.entries < self.buffer.len() {
            self.entries += 1;
            self.inserted += 1;
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

    #[inline]
    pub fn last_serial(&self) -> Option<u64> {
        self.inserted.checked_sub(1)
    }

    pub fn peek_mut_serial(&mut self, serial: u64) -> Option<&mut T> {
        let first = self.inserted - self.entries as u64;
        if serial < first || serial >= self.inserted {
            return None;
        }
        let offset = (serial - first) as usize;
        let len = self.buffer.len();
        self.buffer[(self.read_pos + offset) % len].as_mut()
    }

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

    pub fn peek(&self, offset: usize) -> Option<&T> {
        if offset >= self.entries {
            return None;
        }
        self.buffer[(self.read_pos + offset) % self.buffer.len()].as_ref()
    }

    pub fn reset(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.read_pos = 0;
        self.write_pos = 0;
        self.entries = 0;
        self.inserted = 0;
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
        assert!(!cb.insert(4));
        assert_eq!(cb.entries(), 3);

        assert_eq!(cb.peek(0), Some(&1));
        assert_eq!(cb.peek(2), Some(&3));
        assert_eq!(cb.peek(3), None);

        assert_eq!(cb.pop(), Some(1));
        assert!(cb.insert(4));
        assert_eq!(cb.pop(), Some(2));
        assert_eq!(cb.pop(), Some(3));
        assert_eq!(cb.pop(), Some(4));
        assert_eq!(cb.pop(), None);
    }

    #[test]
    fn serial_handles_survive_eviction() {
        let mut cb = CircularBuffer::new(2);
        cb.insert(10);
        let s_a = cb.last_serial().unwrap();
        cb.insert(20);
        let s_b = cb.last_serial().unwrap();
        assert_eq!(cb.peek_mut_serial(s_a).copied(), Some(10));
        assert_eq!(cb.peek_mut_serial(s_b).copied(), Some(20));

        *cb.peek_mut_serial(s_a).unwrap() = 11;
        assert_eq!(cb.peek(0).copied(), Some(11));

        cb.pop();
        cb.insert(30);
        let s_c = cb.last_serial().unwrap();
        assert_eq!(cb.peek_mut_serial(s_a), None, "evicted handle is stale");
        assert_eq!(cb.peek_mut_serial(s_b).copied(), Some(20));
        assert_eq!(cb.peek_mut_serial(s_c).copied(), Some(30));
    }
}
