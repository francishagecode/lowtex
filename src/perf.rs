// src/perf.rs
//
// Lightweight, opt-in phase profiler for the CPU paint pipeline (Tier 0 of the
// performance work). It exists to answer "where does a frame of painting go?"
// without guessing — splat vs. composite vs. quantize vs. bleed vs. upload.
//
// Disabled and ~free unless `LOWTEX_PROFILE` is set in the environment (any
// non-empty value). When enabled, wrap a phase in `perf::time("name", || ...)`;
// per-phase call counts and total/last durations accumulate on the main thread
// and a one-line summary is printed roughly once a second. Single-threaded by
// design — painting runs on the main thread, so a thread-local map needs no lock
// and the rayon-parallel kernels are each timed as one phase from the caller.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Cached `LOWTEX_PROFILE` check (read once). Off → `time` is a thin passthrough.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LOWTEX_PROFILE").is_some_and(|v| !v.is_empty()))
}

struct Phase {
    calls: u64,
    total: Duration,
    last: Duration,
}

struct Perf {
    phases: BTreeMap<&'static str, Phase>,
    last_report: Instant,
}

thread_local! {
    static PERF: RefCell<Perf> = RefCell::new(Perf {
        phases: BTreeMap::new(),
        last_report: Instant::now(),
    });
}

/// Time `f` under `name` when profiling is on; otherwise just run it. Returns
/// whatever `f` returns, so it wraps an expression in place.
pub fn time<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let out = f();
    record(name, start.elapsed());
    out
}

fn record(name: &'static str, dur: Duration) {
    PERF.with(|p| {
        let mut p = p.borrow_mut();
        let e = p.phases.entry(name).or_insert(Phase {
            calls: 0,
            total: Duration::ZERO,
            last: Duration::ZERO,
        });
        e.calls += 1;
        e.total += dur;
        e.last = dur;
        // Throttle the printout so a fast paint loop doesn't flood the terminal.
        if p.last_report.elapsed() >= Duration::from_secs(1) {
            report(&p);
            p.last_report = Instant::now();
            for e in p.phases.values_mut() {
                *e = Phase {
                    calls: 0,
                    total: Duration::ZERO,
                    last: Duration::ZERO,
                };
            }
        }
    });
}

fn report(p: &Perf) {
    let mut parts = Vec::new();
    for (name, e) in &p.phases {
        if e.calls == 0 {
            continue;
        }
        let avg_us = e.total.as_secs_f64() * 1e6 / e.calls as f64;
        parts.push(format!(
            "{name}: {:.2}ms total / {} calls / {:.0}µs avg",
            e.total.as_secs_f64() * 1e3,
            e.calls,
            avg_us
        ));
    }
    if !parts.is_empty() {
        log::info!("[perf/1s] {}", parts.join("  |  "));
    }
}
