//! The chord-annotation data model: chords, spans, keys, and the JSON shape that is the contract with the ml crate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Quality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    HalfDiminished7,
    Sus2,
    Sus4,
}

impl Quality {
    pub const ALL: [Quality; 10] = [
        Quality::Major,
        Quality::Minor,
        Quality::Diminished,
        Quality::Augmented,
        Quality::Dominant7,
        Quality::Major7,
        Quality::Minor7,
        Quality::HalfDiminished7,
        Quality::Sus2,
        Quality::Sus4,
    ];

    pub fn suffix(&self) -> &'static str {
        match self {
            Quality::Major => "",
            Quality::Minor => "m",
            Quality::Diminished => "dim",
            Quality::Augmented => "aug",
            Quality::Dominant7 => "7",
            Quality::Major7 => "maj7",
            Quality::Minor7 => "m7",
            Quality::HalfDiminished7 => "m7b5",
            Quality::Sus2 => "sus2",
            Quality::Sus4 => "sus4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chord {
    pub root: u8,
    pub quality: Quality,
    #[serde(rename = "qualityUncertain", default)]
    pub quality_uncertain: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    #[serde(rename = "startStep")]
    pub start_step: u32,
    #[serde(rename = "endStep")]
    pub end_step: u32,
    pub chord: Option<Chord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub tonic: u8,
    pub mode: Mode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongAnnotation {
    #[serde(rename = "songId")]
    pub song_id: u32,
    #[serde(rename = "beatsPerBar")]
    pub beats_per_bar: u32,
    #[serde(rename = "gridOffsetSteps")]
    pub grid_offset_steps: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<Key>,
    pub spans: Vec<Span>,
}

impl SongAnnotation {
    pub fn new(song_id: u32) -> Self {
        SongAnnotation {
            song_id,
            beats_per_bar: 4,
            grid_offset_steps: 0,
            key: None,
            spans: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAnnotations {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub source: String,
    #[serde(rename = "gameCode", default, skip_serializing_if = "Option::is_none")]
    pub game_code: Option<String>,
    #[serde(rename = "stepsPerBeat")]
    pub steps_per_beat: f64,
    pub songs: Vec<SongAnnotation>,
}

pub const SCHEMA_VERSION: u32 = 1;

impl GameAnnotations {
    pub fn new(source: String, game_code: Option<String>, steps_per_beat: f64) -> Self {
        GameAnnotations {
            schema_version: SCHEMA_VERSION,
            source,
            game_code,
            steps_per_beat,
            songs: Vec::new(),
        }
    }
}

const SHARP_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

impl Chord {
    pub fn symbol(&self) -> String {
        let root = SHARP_NAMES[(self.root % 12) as usize];
        let mark = if self.quality_uncertain { "?" } else { "" };
        format!("{root}{}{mark}", self.quality.suffix())
    }
}

pub fn parse_chord(s: &str) -> Option<Option<Chord>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let lower = s.to_ascii_lowercase();
    if matches!(lower.as_str(), "nc" | "n.c." | "n.c" | "none" | "-") {
        return Some(None);
    }

    let mut chars = s.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let base = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let mut rest = chars.as_str();
    let mut root = base;
    if let Some(r) = rest.strip_prefix('#') {
        root = (base + 1) % 12;
        rest = r;
    } else if let Some(r) = rest.strip_prefix('b') {
        root = (base + 11) % 12;
        rest = r;
    }

    let mut uncertain = false;
    if let Some(r) = rest.strip_suffix('?') {
        uncertain = true;
        rest = r;
    }

    let table: &[(&str, Quality)] = &[
        ("", Quality::Major),
        ("maj", Quality::Major),
        ("m", Quality::Minor),
        ("mi", Quality::Minor),
        ("min", Quality::Minor),
        ("-", Quality::Minor),
        ("dim", Quality::Diminished),
        ("o", Quality::Diminished),
        ("aug", Quality::Augmented),
        ("+", Quality::Augmented),
        ("7", Quality::Dominant7),
        ("dom7", Quality::Dominant7),
        ("maj7", Quality::Major7),
        ("m7", Quality::Minor7),
        ("min7", Quality::Minor7),
        ("m7b5", Quality::HalfDiminished7),
        ("sus2", Quality::Sus2),
        ("sus4", Quality::Sus4),
        ("sus", Quality::Sus4),
    ];
    let key = rest.to_ascii_lowercase();
    let quality = table.iter().find(|(pat, _)| key == *pat).map(|&(_, q)| q)?;

    Some(Some(Chord {
        root,
        quality,
        quality_uncertain: uncertain,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_spellings_annotators_type() {
        let c = |s: &str| parse_chord(s).unwrap().unwrap();
        assert_eq!(c("C").root, 0);
        assert_eq!(c("C").quality, Quality::Major);
        assert_eq!(c("Am7").root, 9);
        assert_eq!(c("Am7").quality, Quality::Minor7);
        assert_eq!(c("F#m").root, 6);
        assert_eq!(c("F#m").quality, Quality::Minor);
        assert_eq!(c("Bb").root, 10);
        assert_eq!(c("Bbmaj7").quality, Quality::Major7);
        assert_eq!(c("Csus4").quality, Quality::Sus4);
        assert_eq!(c("Csus").quality, Quality::Sus4);
        assert_eq!(c("C#dim").root, 1);
        assert_eq!(c("C#dim").quality, Quality::Diminished);
        assert_eq!(c("Gaug").quality, Quality::Augmented);
        assert_eq!(c("Dm7b5").quality, Quality::HalfDiminished7);
        assert_eq!(c("G7").quality, Quality::Dominant7);
    }

    #[test]
    fn no_chord_and_uncertainty_are_explicit() {
        assert_eq!(parse_chord("NC"), Some(None));
        assert_eq!(parse_chord("n.c."), Some(None));
        assert!(parse_chord("Am?").unwrap().unwrap().quality_uncertain);
        assert!(!parse_chord("Am").unwrap().unwrap().quality_uncertain);
    }

    #[test]
    fn rejects_junk_rather_than_guessing() {
        assert!(parse_chord("H").is_none());
        assert!(parse_chord("Amwhat").is_none());
        assert!(parse_chord("").is_none());
    }

    #[test]
    fn symbol_round_trips_through_the_parser() {
        for root in 0..12u8 {
            for quality in Quality::ALL {
                for quality_uncertain in [false, true] {
                    let c = Chord {
                        root,
                        quality,
                        quality_uncertain,
                    };
                    let parsed = parse_chord(&c.symbol())
                        .unwrap_or_else(|| panic!("{} must parse", c.symbol()))
                        .unwrap_or_else(|| panic!("{} is not N.C.", c.symbol()));
                    assert_eq!(parsed, c, "round-trip of {}", c.symbol());
                }
            }
        }
    }

    #[test]
    fn quality_matches_the_ml_label_space() {
        assert_eq!(Quality::ALL.len(), 10);
        assert_eq!(Quality::ALL[0], Quality::Major);
        assert_eq!(Quality::ALL[4], Quality::Dominant7);
        assert_eq!(Quality::ALL[7], Quality::HalfDiminished7);
        assert_eq!(Quality::ALL[9], Quality::Sus4);
    }

    #[test]
    fn json_round_trips() {
        let mut g =
            GameAnnotations::new("pokemon-emerald.gbaaudio".into(), Some("BPEE".into()), 24.0);
        let mut s = SongAnnotation::new(12);
        s.key = Some(Key {
            tonic: 9,
            mode: Mode::Minor,
        });
        s.spans.push(Span {
            start_step: 0,
            end_step: 96,
            chord: Some(Chord {
                root: 9,
                quality: Quality::Minor7,
                quality_uncertain: false,
            }),
        });
        s.spans.push(Span {
            start_step: 96,
            end_step: 192,
            chord: None,
        });
        g.songs.push(s);

        let json = serde_json::to_string(&g).unwrap();
        let back: GameAnnotations = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.steps_per_beat, 24.0);
        assert_eq!(back.songs[0].spans.len(), 2);
        assert_eq!(back.songs[0].spans[0].chord.unwrap().root, 9);
        assert!(back.songs[0].spans[1].chord.is_none());
        assert_eq!(back.songs[0].key.unwrap().tonic, 9);
    }
}
