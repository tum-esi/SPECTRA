use std::fs::File;
use std::io::Write;

use pprof::protos::Message;

use crate::cli::RunConfig;

#[cfg(feature = "mem-profile")]
#[global_allocator]
static GLOBAL_ALLOCATOR: crate::mem_profile::TrackingAllocator =
    crate::mem_profile::TrackingAllocator;

pub struct Profilers {
    cpu_profile_requested: bool,
    cpu_guard: Option<pprof::ProfilerGuard<'static>>,
    #[cfg(feature = "mem-profile")]
    memory: Option<crate::mem_profile::MemProfiler>,
}

impl Profilers {
    pub fn init(config: &RunConfig) -> Self {
        #[cfg(feature = "mem-profile")]
        let memory = config.memory_profile_output().map(|path| {
            crate::mem_profile::MemProfiler::new(path).unwrap_or_else(|err| {
                eprintln!("could not create memory profile: {err}");
                std::process::exit(1);
            })
        });

        let mut profilers = Self {
            cpu_profile_requested: config.is_cpu_profile(),
            cpu_guard: None,
            #[cfg(feature = "mem-profile")]
            memory,
        };
        profilers.memory_snapshot("start", None, None);
        profilers
    }

    pub fn start_proving(&mut self) {
        if self.cpu_profile_requested {
            self.cpu_guard = Some(
                pprof::ProfilerGuardBuilder::default()
                    .frequency(100)
                    .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                    .build()
                    .expect("could not start profiler"),
            );
        }
    }

    pub fn finish_proving(&mut self) {
        self.memory_snapshot("proving complete", None, None);
        let Some(guard) = self.cpu_guard.take() else {
            return;
        };

        let report = guard.report().build().expect("could not build profile");
        let profile = report.pprof().expect("could not convert profile");

        let mut encoded = Vec::new();
        profile.encode(&mut encoded).expect("could not encode profile");

        let mut file = File::create("profile.pb").expect("could not create profile.pb");
        file.write_all(&encoded).expect("could not write profile.pb");

        println!("Profile written to profile.pb");
    }

    pub fn memory_snapshot(
        &mut self,
        phase: &str,
        step: Option<usize>,
        circuit_index: Option<usize>,
    ) {
        #[cfg(not(feature = "mem-profile"))]
        let _ = (phase, step, circuit_index);

        #[cfg(feature = "mem-profile")]
        if let Some(profiler) = self.memory.as_mut()
            && let Err(err) = profiler.record(phase, step, circuit_index)
        {
            eprintln!("could not write memory profile during {phase}: {err}");
            std::process::exit(1);
        }
    }

    #[cfg(feature = "mem-profile")]
    pub fn finish(mut self) {
        if let Some(mut profiler) = self.memory.take() {
            if let Err(err) = profiler.record("complete", None, None) {
                eprintln!("could not write memory profile during complete: {err}");
                std::process::exit(1);
            }
            let path = profiler.finish().unwrap_or_else(|err| {
                eprintln!("could not finish memory profile: {err}");
                std::process::exit(1);
            });
            println!("Memory profile written to {}", path.display());
        }
    }
}
