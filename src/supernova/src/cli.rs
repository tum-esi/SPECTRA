use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Run,
    Check,
    Benchmark,
    CpuProfile,
    MemoryProfile,
}

pub struct RunConfig {
    mode: Mode,
    batch_size: usize,
    benchmark_output: Option<PathBuf>,
    #[cfg(feature = "mem-profile")]
    memory_profile_output: Option<PathBuf>,
}

impl RunConfig {
    pub fn from_env() -> Self {
        Self::parse(std::env::args().skip(1).collect()).unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(2);
        })
    }

    fn parse(args: Vec<String>) -> Result<Self, String> {
        let selected_modes = [
            ("--check", Mode::Check),
            ("--benchmark", Mode::Benchmark),
            ("--profile", Mode::CpuProfile),
            ("--mem-profile", Mode::MemoryProfile),
        ]
        .into_iter()
        .filter_map(|(flag, mode)| args.iter().any(|arg| arg == flag).then_some(mode))
        .collect::<Vec<_>>();

        // Fail if user selects more than one mode
        if selected_modes.len() > 1 {
            return Err(
                "--check, --benchmark, --profile and --mem-profile are separate modes"
                    .to_string(),
            );
        }
        let mode = selected_modes.first().copied().unwrap_or(Mode::Run);

        if mode == Mode::MemoryProfile && !cfg!(feature = "mem-profile") {
            return Err("--mem-profile requires building with --features mem-profile".to_string());
        }
        if cfg!(feature = "mem-profile") && mode != Mode::MemoryProfile {
            return Err(
                "a binary built with --features mem-profile must run in --mem-profile mode"
                    .to_string(),
            );
        }

        let batch_size = match option_value(&args, "--batch-size")? {
            Some(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| "--batch-size requires a positive integer".to_string())?,
            None => 1,
        };

        let benchmark_output_arg = option_value(&args, "--benchmark-output")?;
        if benchmark_output_arg.is_some() && mode != Mode::Benchmark {
            return Err("--benchmark-output requires --benchmark".to_string());
        }
        let benchmark_output = (mode == Mode::Benchmark).then(|| {
            benchmark_output_arg
                .map(PathBuf::from)
                .unwrap_or_else(crate::benchmark::default_output_path)
        });


        let memory_profile_output_arg = option_value(&args, "--mem-profile-output")?;
        if memory_profile_output_arg.is_some() && mode != Mode::MemoryProfile {
            return Err("--mem-profile-output requires --mem-profile".to_string());
        }
        #[cfg(feature = "mem-profile")]
        let memory_profile_output = (mode == Mode::MemoryProfile).then(|| {
            memory_profile_output_arg
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../benchmarks/supernova_mem_profile.csv")
                })
        });
        

        Ok(Self {
            mode,
            batch_size,
            benchmark_output,
            #[cfg(feature = "mem-profile")]
            memory_profile_output,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn is_check(&self) -> bool {
        self.mode == Mode::Check
    }

    pub fn is_benchmark(&self) -> bool {
        self.mode == Mode::Benchmark
    }

    pub fn is_cpu_profile(&self) -> bool {
        self.mode == Mode::CpuProfile
    }

    pub fn uses_fresh_setup(&self) -> bool {
        matches!(self.mode, Mode::Benchmark | Mode::MemoryProfile)
    }

    pub fn hides_progress(&self) -> bool {
        matches!(self.mode, Mode::Benchmark | Mode::MemoryProfile)
    }

    pub fn benchmark_output(&self) -> Option<&Path> {
        self.benchmark_output.as_deref()
    }

    #[cfg(feature = "mem-profile")]
    pub fn memory_profile_output(&self) -> Option<&Path> {
        self.memory_profile_output.as_deref()
    }
}

fn option_value<'a>(args: &'a [String], option: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = args.iter().position(|arg| arg == option) else {
        return Ok(None);
    };
    match args.get(index + 1) {
        Some(value) if !value.starts_with('-') => Ok(Some(value)),
        Some(_) => Err(format!("{option} requires a value, not another option")),
        None => Err(format!("{option} requires a value")),
    }
}
