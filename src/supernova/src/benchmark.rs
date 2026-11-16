use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::run_metadata::RunMetadata;

const COLUMNS: &[&str] = &[
    "size_pixels",
    "width_pixels",
    "height_pixels",
    "steps",
    "batch_size",
    "dpc_constraints",
    "gain_constraints",
    "demosaic_constraints",
    "wbal_constraints",
    "dpc_num_vars",
    "gain_num_vars",
    "demosaic_num_vars",
    "wbal_num_vars",
    "preparation_time_s",
    "setup_time_s",
    "proving_time_s",
    "dpc_proving_time_s",
    "gain_proving_time_s",
    "demosaic_proving_time_s",
    "wbal_proving_time_s",
    "verification_time_s",
    "evidence_check_time_s",
    "total_time_s",
    "peak_rss_mib",
    "timestamp_utc",
    "git_commit",
    "git_branch",
    "git_worktree",
    "cpu_model",
    "cpu_available_threads",
    "system_os",
    "system_arch",
    "gpu_model",
    "gpu_memory_mib",
];


pub struct BenchmarkResult {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub batch_size: usize,
    pub application_constraints: [usize; 4],
    pub stage_num_vars: [usize; 4],
    pub preparation: Duration,
    pub setup: Duration,
    pub proving: Duration,
    pub stage_proving: [Duration; 4],
    pub verification: Duration,
    /// The verifier's checks on (z_0, z_N), dominated by the native hash of y.
    pub evidence_check: Duration,
    pub total: Duration,
    pub peak_rss_mib: Option<f64>,
}

impl BenchmarkResult {
    pub fn print(&self) {
        println!("--- Benchmark (wall clock) ---");
        println!("preparation:  {:.6} s", self.preparation.as_secs_f64());
        println!(
            "application constraints per step: DPC {}, gain {}, demosaic {}, wbal {}",
            self.application_constraints[0],
            self.application_constraints[1],
            self.application_constraints[2],
            self.application_constraints[3]
        );
        println!(
            "augmented R1CS variables per step: DPC {}, gain {}, demosaic {}, wbal {}",
            self.stage_num_vars[0], self.stage_num_vars[1], self.stage_num_vars[2], self.stage_num_vars[3]
        );
        println!("setup:        {:.6} s", self.setup.as_secs_f64());
        println!("proving:      {:.6} s", self.proving.as_secs_f64());
        println!("  DPC:        {:.6} s", self.stage_proving[0].as_secs_f64());
        println!("  gain:       {:.6} s", self.stage_proving[1].as_secs_f64());
        println!("  demosaic:   {:.6} s", self.stage_proving[2].as_secs_f64());
        println!("  wbal:       {:.6} s", self.stage_proving[3].as_secs_f64());
        println!(
            "recursive verification:   {:.6} s",
            self.verification.as_secs_f64()
        );
        println!(
            "evidence check (H(y)):    {:.6} s",
            self.evidence_check.as_secs_f64()
        );
        println!("total:        {:.6} s", self.total.as_secs_f64());
        match self.peak_rss_mib {
            Some(value) => println!("peak RSS:     {value:.3} MiB"),
            None => println!("peak RSS:     unavailable"),
        }
    }

    pub fn write_csv(&self, path: &Path) -> std::io::Result<PathBuf> {
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let header = COLUMNS.join(",");
        let write_header = !path.exists() || path.metadata()?.len() == 0;
        if !write_header {
            let existing = std::fs::read_to_string(path)?;
            if existing.lines().next() != Some(header.as_str()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "benchmark CSV uses a different schema; choose a new output path",
                ));
            }
        }

        let mut cells = vec![
            (self.width * self.height).to_string(),
            self.width.to_string(),
            self.height.to_string(),
            self.steps.to_string(),
            self.batch_size.to_string(),
            self.application_constraints[0].to_string(),
            self.application_constraints[1].to_string(),
            self.application_constraints[2].to_string(),
            self.application_constraints[3].to_string(),
            self.stage_num_vars[0].to_string(),
            self.stage_num_vars[1].to_string(),
            self.stage_num_vars[2].to_string(),
            self.stage_num_vars[3].to_string(),
            seconds(self.preparation),
            seconds(self.setup),
            seconds(self.proving),
            seconds(self.stage_proving[0]),
            seconds(self.stage_proving[1]),
            seconds(self.stage_proving[2]),
            seconds(self.stage_proving[3]),
            seconds(self.verification),
            seconds(self.evidence_check),
            seconds(self.total),
            self.peak_rss_mib
                .map(|value| format!("{value:.3}"))
                .unwrap_or_default(),
        ];
        cells.extend(RunMetadata::collect().csv_cells());
        debug_assert_eq!(cells.len(), COLUMNS.len());

        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if write_header {
            writeln!(file, "{header}")?;
        }
        writeln!(file, "{}", cells.join(","))?;
        Ok(path.to_path_buf())
    }
}

pub fn default_output_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/supernova.csv")
}

pub fn peak_rss_mib() -> Option<f64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| parse_peak_rss_mib(&status))
}

fn seconds(duration: Duration) -> String {
    format!("{:.9}", duration.as_secs_f64())
}

fn parse_peak_rss_mib(status: &str) -> Option<f64> {
    let kib = status.lines().find_map(|line| {
        let mut fields = line.strip_prefix("VmHWM:")?.split_whitespace();
        let value = fields.next()?.parse::<u64>().ok()?;
        (fields.next() == Some("kB")).then_some(value)
    })?;
    Some(kib as f64 / 1024.0)
}
