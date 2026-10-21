//! project.json: the interchange file between the DAW and the AI pipeline.
//! Paths inside are relative to the project file. Optional fields are
//! `Option` so files written by other tools (the Python prep step) load.

use makepad_micro_serde::*;
use std::path::{Path, PathBuf};

#[derive(SerJson, DeJson, Clone, Debug)]
pub struct Project {
    pub name: String,
    pub duration: f64,
    pub tracks: Vec<Track>,
    pub sample_rate: Option<u32>,
    pub sections: Option<Vec<Section>>,
    pub regions: Option<Vec<Region>>,
}

#[derive(SerJson, DeJson, Clone, Debug)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub file: String,
    pub gain_db: Option<f64>,
    pub mute: Option<bool>,
    pub solo: Option<bool>,
}

#[derive(SerJson, DeJson, Clone, Debug)]
pub struct Section {
    pub start: f64,
    pub end: f64,
    pub label: String,
}

#[derive(SerJson, DeJson, Clone, Debug)]
pub struct Region {
    pub id: String,
    pub track: String,
    pub start: f64,
    pub end: f64,
    #[rename(type)]
    pub kind: String,
    pub gain_db: Option<f64>,
    pub note: Option<String>,
}

/// Region types. "bad" and "regenerate" are flags for the AI pipeline and do
/// not change the sound; "mute" silences; the others apply `gain_db`.
pub const REGION_KINDS: [&str; 6] = ["bad", "regenerate", "too_loud", "too_quiet", "mute", "gain"];

pub fn default_gain_for(kind: &str) -> f64 {
    match kind {
        "too_loud" => -6.0,
        "too_quiet" => 4.0,
        _ => 0.0,
    }
}

pub fn db_to_lin(db: f64) -> f32 {
    10f64.powf(db / 20.0) as f32
}

impl Track {
    pub fn gain(&self) -> f64 {
        self.gain_db.unwrap_or(0.0)
    }
    pub fn muted(&self) -> bool {
        self.mute.unwrap_or(false)
    }
    pub fn soloed(&self) -> bool {
        self.solo.unwrap_or(false)
    }
}

impl Region {
    pub fn gain(&self) -> f64 {
        self.gain_db.unwrap_or(0.0)
    }
    pub fn factor(&self) -> f32 {
        match self.kind.as_str() {
            "mute" => 0.0,
            "bad" | "regenerate" => 1.0,
            _ => db_to_lin(self.gain()),
        }
    }
}

impl Project {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut p = Project::deserialize_json(&text).map_err(|e| format!("parse {}: {e:?}", path.display()))?;
        p.sections.get_or_insert_with(Vec::new);
        p.regions.get_or_insert_with(Vec::new);
        Ok(p)
    }

    /// Temp file then rename, so a crash never leaves a half-written project.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.serialize_json()).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    pub fn sr(&self) -> u32 {
        self.sample_rate.unwrap_or(48000)
    }
    pub fn sections(&self) -> &Vec<Section> {
        self.sections.as_ref().expect("filled on load")
    }
    pub fn sections_mut(&mut self) -> &mut Vec<Section> {
        self.sections.get_or_insert_with(Vec::new)
    }
    pub fn regions(&self) -> &Vec<Region> {
        self.regions.as_ref().expect("filled on load")
    }
    pub fn regions_mut(&mut self) -> &mut Vec<Region> {
        self.regions.get_or_insert_with(Vec::new)
    }

    pub fn track_path(&self, project_path: &Path, i: usize) -> PathBuf {
        project_path.parent().unwrap_or(Path::new(".")).join(&self.tracks[i].file)
    }

    pub fn track_index(&self, id: &str) -> Option<usize> {
        self.tracks.iter().position(|t| t.id == id)
    }

    /// Linear gain per track after mute/solo.
    pub fn track_gains(&self) -> Vec<f32> {
        let any_solo = self.tracks.iter().any(|t| t.soloed());
        self.tracks
            .iter()
            .map(|t| if t.muted() || (any_solo && !t.soloed()) { 0.0 } else { db_to_lin(t.gain()) })
            .collect()
    }
}
