// src/history.rs
//
// Undo/redo for the layer stack — the "document". An entry is a [`Snapshot`]: either a
// whole-stack clone (structural edits — add/remove/reorder a layer, a bake, a PNG load, a
// resolution change) or, for the overwhelmingly common case of a paint stroke, just the
// **one layer** the stroke touched. At 4K a whole-stack clone is hundreds of MB and stalls
// the start of every stroke for ~quarter-second; snapshotting only the active layer cuts
// that by the layer count, and apply is a `mem::replace` swap, so undo/redo never clone at
// all (the prior value is banked for the inverse).
//
// Camera and palette settings are *view* state, not document state, so they are
// deliberately outside undo: stepping back never moves the camera or swaps the
// palette, only the painted result changes.

use crate::layers::Layers;
use crate::paint::Texture;

/// One restorable point. `Stack` is a full clone (correct for any structural change);
/// `Layer` is the cheap stroke case — the pre-edit colour + mask of a single layer.
pub enum Snapshot {
    Stack(Layers),
    Layer {
        index: usize,
        tex: Texture,
        mask: Texture,
    },
}

impl Snapshot {
    /// Apply this snapshot to `layers`, returning the *inverse* snapshot (the state that
    /// was there before) for the opposite stack. A swap, never a clone. A `Layer` whose
    /// index no longer exists (a structural change slipped past) degrades to a no-op
    /// inverse so undo/redo can't panic.
    fn apply_into(self, layers: &mut Layers) -> Snapshot {
        match self {
            Snapshot::Stack(s) => Snapshot::Stack(std::mem::replace(layers, s)),
            Snapshot::Layer { index, tex, mask } => {
                if let Some(l) = layers.layers.get_mut(index) {
                    let inv = Snapshot::Layer {
                        index,
                        tex: std::mem::replace(&mut l.tex, tex),
                        mask: std::mem::replace(&mut l.mask, mask),
                    };
                    l.invalidate();
                    inv
                } else {
                    Snapshot::Layer { index, tex, mask }
                }
            }
        }
    }
}

/// Two stacks of snapshots. The most recent state is at the end of each. `undo`
/// holds states the user can step back to; `redo` holds states they stepped
/// back *from*. A fresh edit clears `redo` (you can't redo into an abandoned
/// future).
pub struct History {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    cap: usize,
}

impl History {
    pub fn new(cap: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            cap: cap.max(1),
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Record a restorable point (the state *before* an edit) and drop the redo future.
    /// Oldest entries are evicted past `cap`.
    pub fn record(&mut self, before: Snapshot) {
        self.undo.push(before);
        if self.undo.len() > self.cap {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Step back: apply the most recent undo snapshot to `layers` in place, banking the
    /// inverse for redo. Returns whether anything changed.
    pub fn undo(&mut self, layers: &mut Layers) -> bool {
        match self.undo.pop() {
            Some(snap) => {
                self.redo.push(snap.apply_into(layers));
                true
            }
            None => false,
        }
    }

    /// Step forward: apply the most recent redo snapshot to `layers` in place, banking the
    /// inverse for undo. Returns whether anything changed.
    pub fn redo(&mut self, layers: &mut Layers) -> bool {
        match self.redo.pop() {
            Some(snap) => {
                self.undo.push(snap.apply_into(layers));
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::Texture;

    /// A one-pixel layer stack whose red channel tags the "state".
    fn state(tag: u8) -> Layers {
        Layers::new(Texture::new(1, 1, [tag, 0, 0, 255]))
    }

    fn tag(l: &Layers) -> u8 {
        l.active_tex().pixels[0]
    }

    #[test]
    fn undo_then_redo_round_trips() {
        let mut h = History::new(8);
        assert!(!h.can_undo() && !h.can_redo());

        // Edit from state 1 to state 2: record the "before" (1), live is now 2.
        let mut live = state(2);
        h.record(Snapshot::Stack(state(1)));
        assert!(h.can_undo());

        // Undo → live becomes 1, 2 banked for redo.
        assert!(h.undo(&mut live));
        assert_eq!(tag(&live), 1);
        assert!(h.can_redo() && !h.can_undo());

        // Redo → live becomes 2 again.
        assert!(h.redo(&mut live));
        assert_eq!(tag(&live), 2);
        assert!(h.can_undo() && !h.can_redo());
    }

    #[test]
    fn a_new_edit_clears_the_redo_future() {
        let mut h = History::new(8);
        let mut live = state(2);
        h.record(Snapshot::Stack(state(1)));
        h.undo(&mut live); // redo now holds 2
        assert!(h.can_redo());

        // A fresh edit invalidates the abandoned future.
        h.record(Snapshot::Stack(state(1)));
        assert!(!h.can_redo());
    }

    #[test]
    fn empty_undo_and_redo_are_noops() {
        let mut h = History::new(8);
        let mut live = state(9);
        assert!(!h.undo(&mut live));
        assert!(!h.redo(&mut live));
    }

    #[test]
    fn capacity_evicts_the_oldest() {
        let mut h = History::new(2);
        h.record(Snapshot::Stack(state(1)));
        h.record(Snapshot::Stack(state(2)));
        h.record(Snapshot::Stack(state(3))); // evicts state(1)

        let mut live = state(4);
        assert!(h.undo(&mut live));
        assert_eq!(tag(&live), 3);
        assert!(h.undo(&mut live));
        assert_eq!(tag(&live), 2);
        assert!(!h.can_undo());
    }

    /// A per-layer (stroke) snapshot restores just that layer, leaving the rest untouched,
    /// and round-trips through redo — the cheap stroke-undo path.
    #[test]
    fn layer_snapshot_restores_one_layer() {
        let mut live = Layers::new(Texture::new(1, 1, [10, 0, 0, 255]));
        live.add_layer(); // index 1, starts transparent [0,0,0,0]
        let before_tex = live.layers[1].tex.clone();
        let before_mask = live.layers[1].mask.clone();
        live.layers[1].tex.pixels[0] = 99; // "paint" the layer

        let mut h = History::new(8);
        h.record(Snapshot::Layer {
            index: 1,
            tex: before_tex,
            mask: before_mask,
        });
        assert!(h.undo(&mut live));
        assert_eq!(live.layers[1].tex.pixels[0], 0, "stroke undone");
        assert_eq!(live.layers[0].tex.pixels[0], 10, "base untouched");
        assert!(h.redo(&mut live));
        assert_eq!(live.layers[1].tex.pixels[0], 99, "stroke re-applied");
    }
}
