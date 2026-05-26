// src/history.rs
//
// Undo/redo for the layer stack — the "document". Each history entry is a full
// snapshot of `Layers`, which is the simplest model that stays correct no matter
// what changed the stack: a paint stroke, a layer add/remove/reorder, an AO bake,
// a PNG load, or a resolution change all collapse to "the layers looked like
// this, now they look like that". At PSX texture sizes a snapshot is a few hundred
// KB, so a capped ring of them costs only a few MB. If textures ever grow large
// (1024²+), this is the obvious thing to revisit — store per-stroke pixel deltas
// instead of whole-stack clones.
//
// Camera and palette settings are *view* state, not document state, so they are
// deliberately outside undo: stepping back never moves the camera or swaps the
// palette, only the painted result changes.

use crate::layers::Layers;

/// Two stacks of snapshots. The most recent state is at the end of each. `undo`
/// holds states the user can step back to; `redo` holds states they stepped
/// back *from*. A fresh edit clears `redo` (you can't redo into an abandoned
/// future).
pub struct History {
    undo: Vec<Layers>,
    redo: Vec<Layers>,
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

    /// Record `before` as a restorable point and drop the redo future. Call with
    /// the state as it was *before* applying an edit. Oldest entries are evicted
    /// past `cap`.
    pub fn record(&mut self, before: Layers) {
        self.undo.push(before);
        if self.undo.len() > self.cap {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Step back: return the previous state, banking `current` for a later redo.
    /// `None` if there is nothing to undo.
    pub fn undo(&mut self, current: Layers) -> Option<Layers> {
        let prev = self.undo.pop()?;
        self.redo.push(current);
        Some(prev)
    }

    /// Step forward: return the next state, banking `current` for a later undo.
    /// `None` if there is nothing to redo.
    pub fn redo(&mut self, current: Layers) -> Option<Layers> {
        let next = self.redo.pop()?;
        self.undo.push(current);
        Some(next)
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

        // Edit from state 1 to state 2: record the "before" (1), now live is 2.
        h.record(state(1));
        assert!(h.can_undo());

        // Undo from live=2 → get back 1, and 2 is banked for redo.
        let back = h.undo(state(2)).unwrap();
        assert_eq!(tag(&back), 1);
        assert!(h.can_redo());
        assert!(!h.can_undo());

        // Redo from live=1 → get 2 back.
        let fwd = h.redo(state(1)).unwrap();
        assert_eq!(tag(&fwd), 2);
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn a_new_edit_clears_the_redo_future() {
        let mut h = History::new(8);
        h.record(state(1));
        let _ = h.undo(state(2)); // redo now holds 2
        assert!(h.can_redo());

        // A fresh edit invalidates the abandoned future.
        h.record(state(1));
        assert!(!h.can_redo());
    }

    #[test]
    fn empty_undo_and_redo_are_noops() {
        let mut h = History::new(8);
        assert!(h.undo(state(9)).is_none());
        assert!(h.redo(state(9)).is_none());
    }

    #[test]
    fn capacity_evicts_the_oldest() {
        let mut h = History::new(2);
        h.record(state(1));
        h.record(state(2));
        h.record(state(3)); // evicts state(1)

        // Three undos available? No — capped at 2.
        assert_eq!(tag(&h.undo(state(4)).unwrap()), 3);
        assert_eq!(tag(&h.undo(state(3)).unwrap()), 2);
        assert!(!h.can_undo());
    }
}
