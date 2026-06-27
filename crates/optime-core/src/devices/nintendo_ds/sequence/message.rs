//! [`Message`]s emitted by a [`SequenceTrack`](super::SequenceTrack) for the controller to act on.

/// The kind of a [`Message`] emitted by the sequence to the controller, carrying its operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Play a note: MIDI `note`, `velocity`, and `duration` in ticks.
    PlayNote {
        note: i32,
        velocity: i32,
        duration: i32,
    },
    /// Select an instrument by `bank` and `program`.
    InstrumentChange { bank: i32, program: i32 },
    /// The track jumped (used for loop detection).
    Jump,
    /// The track ended.
    TrackEnded,
    /// Channel `volume` (0..127) changed.
    VolumeChange { volume: i32 },
    /// Channel `pan` (0..128) changed.
    PanChange { pan: i32 },
    /// Pitch bend / bend-range changed; the controller reads the track state.
    PitchBend,
}

/// A control message produced by a [`SequenceTrack`](super::SequenceTrack) for the controller to
/// act on.
#[derive(Debug, Clone, Copy)]
pub struct Message {
    /// Which track emitted it.
    pub track_num: usize,
    /// The message kind and its operands.
    pub msg_type: MessageType,
    /// Tick the message was generated (filled in by consumers that need it).
    pub timestamp: u32,
}
