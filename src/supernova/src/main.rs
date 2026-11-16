// main program where we run the circuits

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bellpepper_core::test_cs::TestConstraintSystem;
use indicatif::{ProgressBar, ProgressStyle};
use supernova::circuits::spectra::{ProgramSteps, SPECTRA};
use supernova::evidence::{check_evidence, initial_state, readout_digest, Spec};
use supernova::helpers::{load_frame, load_mask, load_rgb_planes};
use supernova::pp_cache::{self, Params, Source};
use supernova::prelude::*;

mod benchmark;
mod cli;
mod profiling;
mod run_metadata;
#[cfg(feature = "mem-profile")]
mod mem_profile;

use benchmark::BenchmarkResult;
use cli::RunConfig;
use profiling::Profilers;

/// Relative to the crate root (where `cargo run` is invoked from).
///
/// Files are keyed on frame geometry plus `pp_cache::fingerprint`, a hash
/// of the step circuits' constraints, so a circuit edit lands on a new
/// file instead of loading the old matrices (see the `stale_cache_*` tests
/// in `pp_cache` for what that would do). Old files are not cleaned up:
/// `rm pp/*` now and then.
static PP_CACHE_DIR: &str = "pp";
const GAIN: u8 = 2;
/// White-balance gains, as pipeline.py calls `wbal.wbal(demosaic_image, 2, 4)`.
const R_GAIN: u8 = 2;
const B_GAIN: u8 = 4;

/// Terminal progress bar over `total` steps: elapsed, ETA, and the current
/// pc as the message so the DPC -> gain -> demosaic -> wbal handovers are visible.
fn progress_bar(total: usize, label: &str) -> ProgressBar {
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{prefix:>9} [{bar:40.cyan/blue}] {pos:>5}/{len} ({percent}%) {elapsed_precise} eta {eta_precise} {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    bar.set_prefix(label.to_string());
    bar
}

/// A prover-side failure at step `k`, explained as far as we can and then
/// aborted. arecibo's errors are coarse (`new` even folds a witness-length
/// mismatch into `SynthesisError`), so re-run steps `0..=k` in the naming
/// constraint system: if one of *our* constraints fails, name it; if all of
/// them pass, the circuit is fine and the parameters are the suspect.
fn diagnose(
    what: &str,
    k: usize,
    err: arecibo::supernova::error::SuperNovaError,
    program: &SPECTRA,
    z0: &[F],
) -> ! {
    eprintln!("\n{what} failed at step {k}: {err:?}");
    match dry_run(program, z0, k + 1) {
        Err((j, name)) => eprintln!("step {j} does not satisfy our circuit: {name}"),
        Ok(_) => eprintln!(
            "steps 0..={k} satisfy our circuits in a test constraint system, so the \
             witness is fine and the R1CS shape in the public params is not the one \
             this binary synthesizes. Delete pp/ and rerun."
        ),
    }
    std::process::exit(1)
}

/// Run one step in a *naming* constraint system and return the state it
/// produces, or the name of the first constraint it fails.
///
/// arecibo's `UnSatIndex` carries an index into the augmented circuit, and
/// in 0.1.1 that circuit (and the `TestShapeCS` that could name its rows)
/// is private, so the index cannot be mapped back. What we can do is
/// re-synthesize *our* step circuit with the same inputs: every constraint
/// of ours has a namespace path, and `which_is_unsatisfied` prints it.
/// The one arecibo-side constraint we care about, `pc == circuit_index`,
/// is checked natively here.
fn check_step(step: &ProgramSteps, pc: usize, z_in: &[F]) -> Result<(usize, Vec<F>), String> {
    if pc != step.circuit_index() {
        return Err(format!(
            "pc = {pc} but the circuit claims index {}",
            step.circuit_index()
        ));
    }

    let mut cs = TestConstraintSystem::<F>::new();
    let z: Vec<AllocatedNum<F>> = z_in
        .iter()
        .enumerate()
        .map(|(i, v)| AllocatedNum::alloc_input(cs.namespace(|| format!("z{i}")), || Ok(*v)).unwrap())
        .collect();
    let pc_alloc =
        AllocatedNum::alloc_input(cs.namespace(|| "pc"), || Ok(F::from(pc as u64))).unwrap();

    let (pc_next, z_next) = step
        .synthesize(&mut cs.namespace(|| "F"), Some(&pc_alloc), &z)
        .map_err(|e| format!("synthesis error: {e:?}"))?;

    if !cs.is_satisfied() {
        return Err(cs.which_is_unsatisfied().unwrap_or("<unnamed>").to_string());
    }

    let pc_next = pc_next
        .and_then(|p| p.get_value())
        .map(|p| {
            // pc is a small integer; read it back from the low limb.
            let repr = p.to_repr();
            repr.as_ref()[0] as usize
        })
        .unwrap_or(pc);
    let z_next = z_next.iter().map(|v| v.get_value().unwrap()).collect();
    Ok((pc_next, z_next))
}

/// Walk steps `0..steps` through `check_step`, threading pc and z the way
/// the folding would. Milliseconds per step instead of seconds, and a
/// constraint *name* on failure instead of an index. Returns the last z.
fn dry_run(program: &SPECTRA, z0: &[F], steps: usize) -> Result<Vec<F>, (usize, String)> {
    let mut pc = program.initial_circuit_index();
    let mut z = z0.to_vec();
    let bar = progress_bar(steps, "checking");
    for k in 0..steps {
        let (pc_next, z_next) = check_step(&program.step(k), pc, &z).map_err(|e| (k, e))?;
        pc = pc_next;
        z = z_next;
        bar.set_message(format!("pc = {pc}"));
        bar.inc(1);
    }
    bar.finish_with_message("done");
    Ok(z)
}

/// Count the four raw application shapes. These exclude Arecibo's recursive
/// wrapper, but provide a stable machine-readable comparison for circuit edits.
fn application_constraints(program: &SPECTRA, z0: &[F]) -> [usize; 4] {
    let mut counts = [0; 4];
    for (stage, count) in counts.iter_mut().enumerate() {
        let step = program.primary_circuit(stage);
        let mut cs = TestConstraintSystem::<F>::new();
        let z: Vec<_> = z0
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                // Each shape is synthesized on its own, so make it its turn.
                let value = if i == Z_PC { F::from(stage as u64) } else { value };
                AllocatedNum::alloc_input(cs.namespace(|| format!("z{i}")), || Ok(value)).unwrap()
            })
            .collect();
        let pc =
            AllocatedNum::alloc_input(cs.namespace(|| "pc"), || Ok(F::from(stage as u64))).unwrap();
        step.synthesize(&mut cs.namespace(|| "F"), Some(&pc), &z)
            .expect("shape synthesis failed");
        // Not asserted satisfied: a stage that fits in one step checks its
        // link to the previous stage here, and z_0 has no previous digest.
        *count = cs.num_constraints();
    }
    counts
}

fn main() {
    let config = RunConfig::from_env();
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/python/out");

    let total_started = config.is_benchmark().then(Instant::now);
    let preparation_started = config.is_benchmark().then(Instant::now);
    let mut profilers = Profilers::init(&config);

    // pipeline.py saves the LUT as uint8 0/1, hence load_mask, not load_lut.
    let lut = load_mask(out.join("0_acquisition/lut.npy")).unwrap();
    let defective = load_frame(out.join("0_acquisition/uniform_cfa_defect.npy")).unwrap();
    let corrected = load_frame(out.join("1_dpc/dpc_cfa.npy")).unwrap();
    let gained = load_frame(out.join("2_gain/gained_image.npy")).unwrap();
    // (H, W, 3) on disk; the circuit wants one H x W plane per channel.
    let rgb = load_rgb_planes(out.join("3_demosaic/demosaic_rgb.npy")).unwrap();
    let white_balanced = load_rgb_planes(out.join("4_wbal/white_balanced_image.npy")).unwrap();
    // Frame geometry comes from the data, not from constants that can drift
    // from pipeline.py's DIMENSIONS.
    let height = defective.len();
    let width = defective[0].len();
    assert_eq!(rgb.r.len(), height, "rgb planes must match the frame height");
    assert_eq!(rgb.r[0].len(), width, "rgb planes must match the frame width");
    assert_eq!(white_balanced.g, rgb.g, "white balance must pass G through");

    // What the verifier holds: the specified parameters and the delivered
    // output y. d_1 is the anchor's digest of the readout (signature not
    // implemented), sent along in z_0.
    let constants = H2::<F>::new();
    let spec_mask = lut.clone();
    let spec = Spec { gain: GAIN, r_gain: R_GAIN, b_gain: B_GAIN, mask: &spec_mask };
    let d1 = readout_digest(&constants, &defective);
    let y = (white_balanced.r.clone(), rgb.g.clone(), white_balanced.b.clone());

    let nivc_circuit = SPECTRA::new(
        GAIN, lut, width, height, defective, corrected, gained, rgb.r, rgb.g, rgb.b,
        R_GAIN, B_GAIN, white_balanced.r, white_balanced.b,
    )
    .with_batch_size(config.batch_size())
    .unwrap_or_else(|err| {
        eprintln!("invalid batch size: {err}");
        std::process::exit(1);
    });

    println!(
        "Loaded {}x{} frame, {} proving steps (batch size {})",
        width,
        height,
        nivc_circuit.num_steps(),
        nivc_circuit.batch_size
    );
    profilers.memory_snapshot("preparation_complete", None, None);

    println!("--- Preparing the state vector z ---");
    // Layout in prelude.rs: parameters, mask digest and d_1 set, the rest zero.
    let z0_primary = initial_state(&constants, &spec, d1);

    // trivial secondary circuit, arity 1
    let z0_secondary = vec![<E2 as Engine>::Scalar::ONE];
    let application_constraints = config
        .is_benchmark()
        .then(|| application_constraints(&nivc_circuit, &z0_primary));
    let preparation_elapsed = preparation_started.map(|start| start.elapsed());

    if config.is_check() {
        println!("--- Dry run (no folding) ---");
        match dry_run(&nivc_circuit, &z0_primary, nivc_circuit.num_steps()) {
            Ok(z) => match check_evidence(&constants, &z0_primary, &z, &spec, (&y.0, &y.1, &y.2)) {
                Ok(()) => println!("every step satisfied, evidence checks hold, final z = {z:?}"),
                Err(why) => println!("every step satisfied, but evidence rejected: {why}"),
            },
            Err((k, name)) => println!("step {k} unsatisfied: {name}"),
        }
        return;
    }
    println!("--- Public Parameters ---");
    let setup_started = config.is_benchmark().then(Instant::now);

    let pp: Params<ProgramSteps> = if config.uses_fresh_setup() {
        println!("Measurement mode: running fresh setup without parameter cache");
        Params::<ProgramSteps>::setup(&nivc_circuit, &*default_ck_hint(), &*default_ck_hint())
    } else {
        let fingerprint = pp_cache::fingerprint(&nivc_circuit);
        let pp_path =
            Path::new(PP_CACHE_DIR).join(format!("spectra_{width}x{height}_{fingerprint}.pp"));
        let (pp, source): (Params<ProgramSteps>, _) = pp_cache::load_or_setup(&pp_path, &nivc_circuit);
        match source {
            Source::Cache => println!("Loaded public params from {}", pp_path.display()),
            Source::Setup => println!("Ran setup, cached public params to {}", pp_path.display()),
        }
        pp
    };

    let setup_elapsed = setup_started.map(|start| start.elapsed());
    let stage_num_vars = config.is_benchmark().then(|| {
        let shapes = pp.primary_r1cs_shapes();
        assert_eq!(
            shapes.len(),
            4,
            "benchmark schema expects DPC, gain, demosaic, and wbal shapes"
        );
        std::array::from_fn(|stage| shapes[stage].num_variables())
    });
    println!("Public params ready, {} circuits", pp.circuit_shapes.len());
    profilers.memory_snapshot("setup_complete", None, None);

    println!("--- Preparing the Recursive Snark ---");
    let secondary = nivc_circuit.secondary_circuit();
    let mut recursive_snark_option: Option<RecursiveSNARK<E1, E2>> = None;

    println!("--- Starting the iteration ---");
    let proving_started = config.is_benchmark().then(Instant::now);
    let bar = if config.hides_progress() {
        ProgressBar::hidden()
    } else {
        progress_bar(nivc_circuit.num_steps(), "folding")
    };
    let mut stage_proving = [Duration::ZERO; 4];

    profilers.start_proving();

    for s in 0..nivc_circuit.num_steps() {
        let step_started = config.is_benchmark().then(Instant::now);
        let primary = nivc_circuit.step(s);
        let stage = primary.circuit_index();

        if !config.is_benchmark() {
            bar.set_message(format!("pc = {}", stage));
        }

        // `take()` moves the snark out of the Option so it can be put back
        // below; without it the Option is consumed on the first iteration.
        let mut recursive_snark = recursive_snark_option.take().unwrap_or_else(|| {
            RecursiveSNARK::new(
                &pp,
                &nivc_circuit,
                &primary,
                &secondary,
                &z0_primary,
                &z0_secondary,
            )
            .unwrap_or_else(|e| diagnose("RecursiveSNARK::new", s, e, &nivc_circuit, &z0_primary))
        });

        recursive_snark
            .prove_step(&pp, &primary, &secondary)
            .unwrap_or_else(|e| diagnose("prove_step", s, e, &nivc_circuit, &z0_primary));
        recursive_snark_option = Some(recursive_snark);
        profilers.memory_snapshot("proving_step", Some(s), Some(stage));
        bar.inc(1);

        if let Some(start) = step_started {
            stage_proving[stage] += start.elapsed();
        }
    }

    bar.finish_with_message("done");
    let recursive_snark = recursive_snark_option.expect("no steps");
    let proving_elapsed = proving_started.map(|start| start.elapsed());
    profilers.finish_proving();

    println!("--- Verifying ---");
    let verification_started = config.is_benchmark().then(Instant::now);
    let verification_result = recursive_snark.verify(&pp, &z0_primary, &z0_secondary);
    let verification_elapsed = verification_started.map(|start| start.elapsed());
    profilers.memory_snapshot(
        "recursive_verification_complete", 
        None,
        None
    );

    // Timed apart from the proof check: this is the verifier hashing y.
    let mut evidence_elapsed = Duration::ZERO;
    let verified = match verification_result {
        Ok((zi, _)) => match {
            let started = Instant::now();
            let checked = check_evidence(&constants, &z0_primary, &zi, &spec, (&y.0, &y.1, &y.2));
            evidence_elapsed = started.elapsed();
            checked
        } {
            Ok(()) => {
                println!("verified, final z = {zi:?}");
                true
            }
            Err(why) => {
                println!("proof verified, but evidence rejected: {why}");
                false
            }
        },
        Err(err) => {
            println!("verify failed: {err:?}");
            // Locate the culprit with names, since the index cannot be mapped.
            if let Err((k, name)) = dry_run(&nivc_circuit, &z0_primary, nivc_circuit.num_steps()) {
                println!("step {k} unsatisfied: {name}");
            }
            false
        }
    };

    // Construct benchmark object after everything has run
    let mut benchmark_saved = true;
    if config.is_benchmark() {
        let result = BenchmarkResult {
            width,
            height,
            steps: nivc_circuit.num_steps(),
            batch_size: nivc_circuit.batch_size,
            application_constraints: application_constraints.unwrap(),
            stage_num_vars: stage_num_vars.unwrap(),
            preparation: preparation_elapsed.unwrap(),
            setup: setup_elapsed.unwrap(),
            proving: proving_elapsed.unwrap(),
            stage_proving,
            verification: verification_elapsed.unwrap(),
            evidence_check: evidence_elapsed,
            total: total_started.unwrap().elapsed(),
            peak_rss_mib: benchmark::peak_rss_mib(),
        };
        result.print();

        if verified {
            match result.write_csv(config.benchmark_output().unwrap()) {
                Ok(path) => println!("Benchmark written to {}", path.display()),
                Err(err) => {
                    eprintln!("could not write benchmark CSV: {err}");
                    benchmark_saved = false;
                }
            }
        } else {
            eprintln!("benchmark not written because verification failed");
            benchmark_saved = false;
        }
    }

    #[cfg(feature = "mem-profile")]
    profilers.finish();
    if !verified || !benchmark_saved {
        std::process::exit(1);
    }
}
