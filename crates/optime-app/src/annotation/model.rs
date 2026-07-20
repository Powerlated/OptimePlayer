//! The annotation data model and its on-disk JSON.
//!
//! Vocabulary: **root is always captured; a plain triad is a complete answer; quality can be marked
//! uncertain.** Not timidity — it is what the data supports. Sparse game voicings are frequently
//! ambiguous above the triad (sus2? add9? two voices and a passing tone?), which is exactly what
//! made the heuristic reference pile up on Sus2. Forcing a guess would manufacture labels rather
//! than record them. So `C` and `Am` are valid complete inputs, [`Chord::quality_uncertain`] keeps
//! an ambiguous colour out of the quality metrics without discarding the root, and root + maj/min is
//! the primary number — everything above it is secondary and says so.
//!
//! The JSON is the **contract with `optime-ml`**, which is why the mapping into `theory::Quality`
//! deliberately lives *there*, not here: the app must never depend on the ml crate (it pulls in
//! burn), and `theory` stays the single source of the label space per the ml conventions.

use serde::{Deserialize, Serialize};

/// Chord quality — **exactly** the ten `optime_ml::theory::Quality` variants, in that order.
///
/// Deliberately one enum rather than a `(triad, extension)` pair. The pair reads as more
/// expressive, but its product is 18 combinations while the label space has 10: `mMaj7`, `aug7`,
/// `dimMaj7`, `7sus2` and friends are all typable and none of them are scoreable. A vocabulary the
/// model cannot express is a label that can never be evaluated, so the type refuses to represent
/// one (the repo's "make invalid states unrepresentable" rule, applied to a real trap this caught).
///
/// Extensions being *optional* is preserved where it belongs — in [`parse_chord`], where `C` and
/// `Am` are complete inputs — not as an axis that can be combined into nonsense.
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
    /// `m7b5`.
    HalfDiminished7,
    Sus2,
    Sus4,
}

impl Quality {
    /// Every quality, in `optime_ml::theory::Quality::ALL` order — the pinned 1:1 correspondence
    /// with the ml label space.
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

    /// The suffix written after the root in a chord symbol.
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

/// One annotated chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chord {
    /// Root pitch class, C = 0.
    pub root: u8,
    pub quality: Quality,
    /// The root is confident but the quality isn't. Such spans score for **root only** — they are
    /// still real data, just honest about which part of it is trustworthy.
    #[serde(rename = "qualityUncertain", default)]
    pub quality_uncertain: bool,
}

/// One chord span on the sequencer-step timeline.
///
/// `chord: None` is *no chord* (N.C.) — a deliberate annotation, not a gap. An unannotated stretch
/// of the song simply has no span covering it, which is a different thing entirely: "there is no
/// harmony here" versus "nobody has listened to this yet".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    #[serde(rename = "startStep")]
    pub start_step: u32,
    #[serde(rename = "endStep")]
    pub end_step: u32,
    /// `None` = N.C.
    pub chord: Option<Chord>,
}

/// Major or minor mode for a song-level key annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Major,
    Minor,
}

/// The song's overall key (the model also predicts a pooled key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    /// Tonic pitch class, C = 0.
    pub tonic: u8,
    pub mode: Mode,
}

/// Everything annotated about one song.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongAnnotation {
    #[serde(rename = "songId")]
    pub song_id: u32,
    /// The meter. One value per song: mid-song meter changes are not modelled — `schemaVersion`
    /// exists so adding a change list later stays additive.
    #[serde(rename = "beatsPerBar")]
    pub beats_per_bar: u32,
    /// Step at which bar 1 begins. Non-zero when the song opens with a pickup.
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

/// One game's annotation file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAnnotations {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    /// Source archive filename the labels were authored against.
    pub source: String,
    #[serde(rename = "gameCode", default, skip_serializing_if = "Option::is_none")]
    pub game_code: Option<String>,
    /// The device's steps per beat, so ml can convert steps → its 4-frames-per-beat grid without
    /// re-running the engine.
    #[serde(rename = "stepsPerBeat")]
    pub steps_per_beat: f64,
    pub songs: Vec<SongAnnotation>,
}

/// Current schema version. Bump when a change isn't backward-compatible.
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

/// Pitch-class names used for display; parsing also accepts the flat spellings.
const SHARP_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

impl Chord {
    /// The chord as a symbol (`Am7`, `F#sus4`, `Cmaj7`) — the same spelling [`parse_chord`] accepts.
    pub fn symbol(&self) -> String {
        let root = SHARP_NAMES[(self.root % 12) as usize];
        let mark = if self.quality_uncertain { "?" } else { "" };
        format!("{root}{}{mark}", self.quality.suffix())
    }
}

/// Parses a chord symbol as typed by an annotator. `None` on anything unrecognised, so a typo
/// leaves the existing label alone rather than silently writing a wrong one.
///
/// Typing is the fastest input path there is, so this accepts the spellings people actually use:
/// `Am7`, `F#m`, `Bbmaj7`, `Csus4`, `C#dim`, `Gaug`, `Dm7b5`. A trailing `?` marks the quality
/// uncertain. Returns `Ok(None)` for `NC`/`N.C.` (an explicit no-chord).
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
    // Accidentals: `#` sharp, `b` flat. (`s` is *not* accepted as a sharp: it would swallow the `s`
    // of `Csus4`. `#` is the standard spelling anyway.) A leading `b` here can only be an
    // accidental — the root letter is already consumed.
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

    // Every accepted spelling of each quality. Matched whole (not by prefix), so an unknown suffix
    // is rejected outright instead of degrading into a wrong-but-plausible chord.
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
        // Bare `sus` conventionally means sus4.
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
        // Flats: the accidental is unambiguous once the root letter is consumed.
        assert_eq!(c("Bb").root, 10);
        assert_eq!(c("Bbmaj7").quality, Quality::Major7);
        assert_eq!(c("Csus4").quality, Quality::Sus4);
        assert_eq!(c("Csus").quality, Quality::Sus4); // bare sus == sus4
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
        // A typo must not silently become a wrong label.
        assert!(parse_chord("H").is_none());
        assert!(parse_chord("Amwhat").is_none());
        assert!(parse_chord("").is_none());
    }

    /// Every chord the model can hold must survive `symbol() → parse_chord()`. This is what caught
    /// the original `(triad, ext)` product minting unscoreable chords like `mMaj7`.
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

    /// The vocabulary must stay 1:1 with `optime_ml::theory::Quality::ALL`, which is the whole
    /// reason this enum exists. ml owns the mapping; this pins the count and order it maps from.
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
        // An explicit no-chord survives as one, distinct from an absent span.
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
