// src/tablet.rs
//
// Pen-tablet pressure source. winit 0.30 dropped the pen/pressure fields from its
// pointer events, so on macOS we tap NSEvent directly with an application-global
// local event monitor and stash the latest stylus pressure. Everything else (the
// brush, the stroke engine) stays winit-driven; this just makes the current
// pressure available to read when a dab is about to be laid.
//
// A plain mouse (or a non-Force-Touch trackpad) reports pressure 1.0 while a
// button is held, so painting with a mouse is unaffected — the pressure read is a
// constant 1.0 and `Brush::with_pressure` leaves the brush untouched. On non-macOS
// platforms the whole thing is a stub that always returns 1.0 (Windows/Linux pen
// support is future work), and none of the objc2 deps are pulled in.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Shared, latest-wins stylus pressure. The value is stored as `f32` bits in an
/// atomic so the macOS event-monitor block (which captures a clone of the `Arc`)
/// and the main loop can touch it without locking. Defaults to 1.0 (full
/// pressure) so reads are safe before any pen event arrives.
pub struct Tablet {
    current: Arc<AtomicU32>,
    /// The opaque monitor object returned by AppKit, held so the monitor stays
    /// registered for the app's lifetime. macOS only; unused elsewhere.
    #[cfg(target_os = "macos")]
    _monitor: Option<objc2::rc::Retained<objc2::runtime::AnyObject>>,
}

impl Default for Tablet {
    fn default() -> Self {
        Self::new()
    }
}

impl Tablet {
    pub fn new() -> Self {
        Self {
            current: Arc::new(AtomicU32::new(1.0f32.to_bits())),
            #[cfg(target_os = "macos")]
            _monitor: None,
        }
    }

    /// The most recent stylus pressure, clamped to `0.0..=1.0`. 1.0 when no pen is
    /// in use (mouse, or before the first pen event).
    pub fn latest(&self) -> f32 {
        f32::from_bits(self.current.load(Ordering::Relaxed)).clamp(0.0, 1.0)
    }

    /// Install the platform pressure tap. Call once, after the window exists.
    #[cfg(target_os = "macos")]
    pub fn install(&mut self) {
        use block2::RcBlock;
        use objc2_app_kit::{NSEvent, NSEventMask};
        use std::ptr::NonNull;

        if self._monitor.is_some() {
            return; // already installed
        }

        let cell = self.current.clone();
        // The handler fires on the main thread inside winit's NSApplication run
        // loop, just before the event is dispatched to the window — so the pressure
        // we record is the pressure for the very mouse event winit is about to
        // deliver. We return the event unchanged so winit still receives it.
        let block = RcBlock::new(move |event: NonNull<NSEvent>| -> *mut NSEvent {
            // SAFETY: AppKit guarantees the event is live for the call's duration.
            let pressure = unsafe { event.as_ref().pressure() };
            cell.store(pressure.to_bits(), Ordering::Relaxed);
            event.as_ptr()
        });

        let mask = NSEventMask::LeftMouseDown | NSEventMask::LeftMouseDragged;
        // SAFETY: a standard AppKit local event monitor. The returned object owns
        // the registration; dropping it (on App teardown) removes the monitor.
        self._monitor =
            unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &block) };
    }

    /// No pressure source off macOS yet — reads stay at the 1.0 default.
    #[cfg(not(target_os = "macos"))]
    pub fn install(&mut self) {}
}
