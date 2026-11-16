use std::env;
use std::path::Path;
use std::process::Command;

pub(crate) struct RunMetadata {
    pub(crate) timestamp_utc: String,
    pub(crate) git_commit: String,
    pub(crate) git_branch: String,
    pub(crate) git_worktree: String,
    pub(crate) cpu_model: String,
    pub(crate) cpu_available_threads: String,
    pub(crate) system_os: String,
    pub(crate) system_arch: String,
    pub(crate) gpu_model: String,
    pub(crate) gpu_memory_mib: String,
}

impl RunMetadata {
    pub(crate) fn collect() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = root.to_string_lossy();

        let git = |args: &[&str]| {
            let mut full = vec!["-C", root.as_ref()];
            full.extend_from_slice(args);
            command_output("git", &full).unwrap_or_else(|| "unknown".to_owned())
        };

        let git_worktree = Command::new("git")
            .args([
                "-C",
                root.as_ref(),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| {
                if output.stdout.is_empty() {
                    "clean"
                } else {
                    "dirty"
                }
            })
            .unwrap_or("unknown")
            .to_owned();

        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();

        let cpu_model = cpuinfo
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim().to_owned())
            })
            .or_else(|| command_output("sysctl", &["-n", "machdep.cpu.brand_string"]))
            .unwrap_or_else(|| "unknown".to_owned());

        let system_os = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|info| {
                info.lines().find_map(|line| {
                    line.strip_prefix("PRETTY_NAME=")
                        .map(|value| value.trim_matches('"').to_owned())
                })
            })
            .or_else(|| command_output("uname", &["-s", "-r"]))
            .unwrap_or_else(|| "unknown".to_owned());

        let gpu = command_output(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        )
        .or_else(|| {
            env::var("BENCHMARK_GPU_QUERY")
                .ok()
                .filter(|value| !value.is_empty())
        });

        let (gpu_model, gpu_memory_mib) = gpu
            .as_deref()
            .and_then(|value| value.lines().next())
            .and_then(|value| value.rsplit_once(','))
            .map(|(name, memory)| (name.trim().to_owned(), memory.trim().to_owned()))
            .unwrap_or_else(|| ("unavailable".to_owned(), "unavailable".to_owned()));

        Self {
            timestamp_utc: command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
                .unwrap_or_else(|| "unknown".to_owned()),
            git_commit: git(&["rev-parse", "HEAD"]),
            git_branch: git(&["branch", "--show-current"]),
            git_worktree,
            cpu_model,
            cpu_available_threads: std::thread::available_parallelism()
                .map(|count| count.get().to_string())
                .unwrap_or_else(|_| "unknown".to_owned()),
            system_os,
            system_arch: std::env::consts::ARCH.to_owned(),
            gpu_model,
            gpu_memory_mib,
        }
    }

    pub(crate) fn csv_cells(self) -> Vec<String> {
        vec![
            csv_cell(&self.timestamp_utc),
            csv_cell(&self.git_commit),
            csv_cell(&self.git_branch),
            csv_cell(&self.git_worktree),
            csv_cell(&self.cpu_model),
            self.cpu_available_threads,
            csv_cell(&self.system_os),
            csv_cell(&self.system_arch),
            csv_cell(&self.gpu_model),
            self.gpu_memory_mib,
        ]
    }
}

fn csv_cell(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

pub(crate) fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}
