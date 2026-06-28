// src/config.rs
//
// Tiny persistent app config that survives across launches — currently just the
// last-opened texture folder, so the brush browser reopens where you left off.
// Stored as RON (matching the project format) under the OS config dir. Every
// operation is best-effort: a missing or unreadable file falls back to defaults
// and a failed write is logged, never fatal.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Default)]
pub struct Config {
    /// The texture folder the user last opened in the brush browser.
    #[serde(default)]
    pub last_texture_folder: Option<PathBuf>,
    /// The alpha-tip folder the user last opened in the brush-tip browser.
    #[serde(default)]
    pub last_alpha_folder: Option<PathBuf>,
}

impl Config {
    /// `~/.config/lowtex/config.ron` on Linux/macOS, `%APPDATA%\lowtex\config.ron`
    /// on Windows. `None` only if the home/appdata env var is unset.
    fn path() -> Option<PathBuf> {
        let base = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
        }?;
        Some(base.join("lowtex").join("config.ron"))
    }

    /// Load the saved config, or a default (empty) one if none exists or it's
    /// unreadable/corrupt.
    pub fn load() -> Self {
        let Some(path) = Self::path() else {
            return Self::default();
        };
        let Ok(s) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        ron::from_str(&s).unwrap_or_default()
    }

    /// Persist the config, best-effort (errors are logged, never fatal).
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()) {
            Ok(s) => {
                if let Err(e) = std::fs::write(&path, s) {
                    log::warn!("could not save config: {e}");
                }
            }
            Err(e) => log::warn!("could not serialize config: {e}"),
        }
    }
}
