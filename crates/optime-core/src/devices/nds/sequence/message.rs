#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    PlayNote {
        note: i32,
        velocity: i32,
        duration: i32,
    },
    InstrumentChange {
        bank: i32,
        program: i32,
    },
    Jump,
    TrackEnded,
    VolumeChange {
        volume: i32,
    },
    PanChange {
        pan: i32,
    },
    PitchBend,
}

#[derive(Debug, Clone, Copy)]
pub struct Message {
    pub track_num: usize,
    pub msg_type: MessageType,
    pub timestamp: u32,
}
