//! Process-wide Rust allocation counters and step-boundary CSV snapshots.
//!
//! This is gated by the `mem-profile` Cargo feature, because wrapping
//! the global allocator adds atomic operations to every Rust allocation and
//! that could affect results when performing benchmarks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::{self, Write as _};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::run_metadata::RunMetadata;

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);

const CSV_HEADER: &str = concat!(
    "elapsed_s,phase,step,circuit_index,",
    "allocated_bytes_total,deallocated_bytes_total,",
    "live_bytes,peak_live_bytes,",
    "allocation_count,deallocation_count,reallocation_count,",
    "current_rss_bytes,peak_rss_bytes,",
    "git_commit,git_branch,cpu_model,cpu_available_threads,",
    "gpu_model,gpu_memory_mib",
);

thread_local! {
    /// Profiling I/O and formatting are implementation details, not prover work.
    static RECORDING_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub struct TrackingAllocator;

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    allocated_bytes: u64,
    deallocated_bytes: u64,
    live_bytes: u64,
    peak_live_bytes: u64,
    allocations: u64,
    deallocations: u64,
    reallocations: u64,
}

fn add_live(bytes: u64) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

fn remove_live(bytes: u64) {
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

fn recording_enabled() -> bool {
    RECORDING_ENABLED.try_with(Cell::get).unwrap_or(true)
}

struct RecordingPaused(bool);

impl RecordingPaused {
    fn new() -> Self {
        Self(
            RECORDING_ENABLED
                .try_with(|enabled| enabled.replace(false))
                .unwrap_or(true),
        )
    }
}

impl Drop for RecordingPaused {
    fn drop(&mut self) {
        let _ = RECORDING_ENABLED.try_with(|enabled| enabled.set(self.0));
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && recording_enabled() {
            let bytes = layout.size() as u64;
            ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            add_live(bytes);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && recording_enabled() {
            let bytes = layout.size() as u64;
            ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            add_live(bytes);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if recording_enabled() {
            let bytes = layout.size() as u64;
            DEALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            remove_live(bytes);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && recording_enabled() {
            let old_size = layout.size() as u64;
            let new_size = new_size as u64;
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            // A successful realloc releases the old requested extent and
            // acquires the new one. Count both sides so the cumulative totals
            // describe allocator traffic, while live bytes change by the delta.
            DEALLOCATED_BYTES.fetch_add(old_size, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            if new_size >= old_size {
                let growth = new_size - old_size;
                add_live(growth);
            } else {
                let shrinkage = old_size - new_size;
                remove_live(shrinkage);
            }
        }
        new_ptr
    }
}

fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
    }
}

pub struct MemProfiler {
    started: Instant,
    output: PathBuf,
    writer: BufWriter<File>,
    metadata: RunMetadata,
}

impl MemProfiler {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        // Metadata discovery and writer setup are profiler implementation
        // details, so keep their allocations out of the measured workload.
        let _recording_paused = RecordingPaused::new();
        let metadata = RunMetadata::collect();
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{CSV_HEADER}")?;
        Ok(Self {
            started: Instant::now(),
            output: path.to_path_buf(),
            writer,
            metadata,
        })
    }

    pub fn record(
        &mut self,
        phase: &str,
        step: Option<usize>,
        circuit_index: Option<usize>,
    ) -> std::io::Result<()> {
        // Exclude temporary allocations made while sampling /proc and writing
        // the row. The guard and all suppressed temporaries die in this call.
        let _recording_paused = RecordingPaused::new();
        let allocation = allocation_snapshot();
        let (rss, peak_rss) = process_rss_bytes();
        writeln!(
            self.writer,
            concat!(
                "{:.9},{phase},{},{},{},{},{},{},{},{},{},",
                "{rss},{peak_rss},{git_commit},{git_branch},",
                "{cpu_model},{cpu_threads},{gpu_model},{gpu_memory}",
            ),
            self.started.elapsed().as_secs_f64(),
            CsvOptional(step),
            CsvOptional(circuit_index),
            allocation.allocated_bytes,
            allocation.deallocated_bytes,
            allocation.live_bytes,
            allocation.peak_live_bytes,
            allocation.allocations,
            allocation.deallocations,
            allocation.reallocations,
            phase = phase,
            rss = CsvOptional(rss),
            peak_rss = CsvOptional(peak_rss),
            git_commit = CsvText(&self.metadata.git_commit),
            git_branch = CsvText(&self.metadata.git_branch),
            cpu_model = CsvText(&self.metadata.cpu_model),
            cpu_threads = CsvText(&self.metadata.cpu_available_threads),
            gpu_model = CsvText(&self.metadata.gpu_model),
            gpu_memory = CsvText(&self.metadata.gpu_memory_mib),
        )?;
        self.writer.flush()
    }

    pub fn finish(mut self) -> std::io::Result<PathBuf> {
        self.writer.flush()?;
        Ok(self.output)
    }
}

struct CsvOptional<T>(Option<T>);

impl<T: fmt::Display> fmt::Display for CsvOptional<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_ref() {
            Some(value) => value.fmt(formatter),
            None => Ok(()),
        }
    }
}

struct CsvText<'a>(&'a str);

impl fmt::Display for CsvText<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("\"")?;
        for character in self.0.chars() {
            if character == '"' {
                formatter.write_str("\"\"")?;
            } else {
                formatter.write_char(character)?;
            }
        }
        formatter.write_str("\"")
    }
}

fn process_rss_bytes() -> (Option<u64>, Option<u64>) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    (
        status_bytes(&status, "VmRSS:"),
        status_bytes(&status, "VmHWM:"),
    )
}

fn status_bytes(status: &str, key: &str) -> Option<u64> {
    let kib = status.lines().find_map(|line| {
        let mut fields = line.strip_prefix(key)?.split_whitespace();
        let value = fields.next()?.parse::<u64>().ok()?;
        (fields.next() == Some("kB")).then_some(value)
    })?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_rss_fields() {
        let status = "VmRSS:\t  65536 kB\nVmHWM:\t 126464 kB\n";
        assert_eq!(status_bytes(status, "VmRSS:"), Some(67_108_864));
        assert_eq!(status_bytes(status, "VmHWM:"), Some(129_499_136));
        assert_eq!(status_bytes("VmRSS:\t10 MB\n", "VmRSS:"), None);
    }

    #[test]
    fn formats_missing_csv_values_as_empty_fields() {
        assert_eq!(CsvOptional(Some(12)).to_string(), "12");
        assert_eq!(CsvOptional::<u64>(None).to_string(), "");
        assert_eq!(CsvText("A, \"B\"").to_string(), "\"A, \"\"B\"\"\"");
    }

    #[test]
    fn successful_realloc_counts_full_old_and_new_extents() {
        let layout = Layout::from_size_align(16, 8).unwrap();
        let before = allocation_snapshot();
        let ptr = unsafe { TrackingAllocator.alloc(layout) };
        assert!(!ptr.is_null());
        let allocated = allocation_snapshot();
        assert!(allocated.allocated_bytes >= before.allocated_bytes + 16);
        assert!(allocated.allocations >= before.allocations + 1);

        let ptr = unsafe { TrackingAllocator.realloc(ptr, layout, 32) };
        assert!(!ptr.is_null());
        let reallocated = allocation_snapshot();
        assert!(reallocated.allocated_bytes >= allocated.allocated_bytes + 32);
        assert!(reallocated.deallocated_bytes >= allocated.deallocated_bytes + 16);
        assert!(reallocated.reallocations >= allocated.reallocations + 1);

        let new_layout = Layout::from_size_align(32, 8).unwrap();
        unsafe { TrackingAllocator.dealloc(ptr, new_layout) };
        let deallocated = allocation_snapshot();
        assert!(deallocated.deallocated_bytes >= reallocated.deallocated_bytes + 32);
        assert!(deallocated.deallocations >= reallocated.deallocations + 1);
    }

    #[test]
    fn writes_a_complete_csv_snapshot() {
        let path = std::env::temp_dir().join(format!(
            "supernova-mem-profile-test-{}.csv",
            std::process::id()
        ));
        let mut profiler = MemProfiler::new(&path).unwrap();
        profiler.record("test", Some(7), Some(2)).unwrap();
        profiler.finish().unwrap();

        let csv = std::fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap().split(',').count(), 19);
        let fields: Vec<_> = lines.next().unwrap().splitn(14, ',').collect();
        assert_eq!(fields.len(), 14);
        assert_eq!(&fields[1..4], &["test", "7", "2"]);
        assert!(fields[13].starts_with('"'));
        assert!(lines.next().is_none());
        std::fs::remove_file(path).unwrap();
    }
}

