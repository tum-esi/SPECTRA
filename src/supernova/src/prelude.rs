//! Common imports and type aliases shared by every circuit.
//!
//! Bring everything into scope with `use crate::prelude::*;`.

pub use bellpepper_core::{
    ConstraintSystem, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
pub use ff::{Field, PrimeField};

/// Bit-packing helpers. `pack_bits` is the in-circuit side (1 constraint),
/// `compute_multipacking` / `bytes_to_bits_le` the native side the verifier
/// replays with. Same file upstream, so the two halves cannot drift.
pub use bellpepper::gadgets::multipack;

pub use arecibo::{
    provider::{PallasEngine, VestaEngine},
    supernova::{NonUniformCircuit, PublicParams, RecursiveSNARK},
    traits::{
        Engine,
        circuit::TrivialCircuit,
        circuit_supernova::{StepCircuit, TrivialSecondaryCircuit},
        snark::default_ck_hint,
    },
};

pub use generic_array::typenum::U2;
pub use neptune::circuit2::poseidon_hash_allocated;
pub use neptune::poseidon::{Poseidon, PoseidonConstants};
use std::path::PathBuf;
pub use std::sync::Arc;

/// Custom types
pub type H2<F> = PoseidonConstants<F, U2>;
pub type E1 = PallasEngine;
pub type E2 = VestaEngine;
pub type F = <E1 as Engine>::Scalar;
pub type Secondary = TrivialSecondaryCircuit<<E2 as Engine>::Scalar>;


/// Where the fixtures for one stage live, written by
/// `scripts/python/on-chip-processing/pipeline.py`.
///
/// Paths are relative to this crate's manifest dir, which is where cargo runs
/// tests from — so `../../scripts/python/data/...`.
pub const DATA: &str = "../../scripts/python/data";

/// One stage's fixture pair: the honest input, the honest output, and the
/// dishonest outputs forged from the *same* input.
///
/// A tampered frame only means something next to the benign frame it
/// contradicts: the test is "same input, wrong output → the digest must
/// differ", so both halves have to come from one place.
pub struct TestingPaths {
    /// Input frame the stage reads.
    pub input: PathBuf,
    /// Ground-truth output for that input.
    pub benign: PathBuf,
    /// Forged outputs, keyed by tamper mode (`wrong_gain_3x`, …).
    pub tampered: Vec<(String, PathBuf)>,
}

impl TestingPaths {
    /// Fixtures for the digital-gain stage run on its own.
    ///
    /// Input is `00_raw`, not a DPC output: with no defect in the frame, a
    /// failing gain test is the gain circuit's fault and nothing else's.
    ///
    /// `pattern` is an acquisition pattern name, e.g. `gradient_horizontal`.
    /// Not every tamper mode produces a file for every pattern — a uniform
    /// frame already saturated at 255 gains to itself under any factor, so
    /// pipeline.py skips writing a "tampered" frame identical to the honest
    /// one. [`Self::tampered`] therefore lists only the modes that exist.
    pub fn gain_single(pattern: &str) -> TestingPaths {
        let frame = format!("{pattern}.npy");
        let tampered_root = PathBuf::from(DATA).join("single/01_gain_from_raw_tampered");

        let mut tampered: Vec<(String, PathBuf)> = std::fs::read_dir(&tampered_root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let path = e.path().join(&frame);
                let mode = e.file_name().to_string_lossy().into_owned();
                path.exists().then_some((mode, path))
            })
            .collect();
        // read_dir order is filesystem-dependent; sort so a test that indexes
        // into this list gets the same mode on every machine.
        tampered.sort();

        TestingPaths {
            input: PathBuf::from(DATA).join("00_raw").join(&frame),
            benign: PathBuf::from(DATA).join("single/01_gain_from_raw").join(&frame),
            tampered,
        }
    }

    /// Fixtures for the dead-pixel-correction stage run on its own.
    pub fn dpc_single(pattern: &str) -> TestingPaths {
        let frame = format!("{pattern}.npy");
        TestingPaths {
            input: PathBuf::from(DATA).join("01_defective").join(&frame),
            benign: PathBuf::from(DATA)
                .join("single/01_dpc_from_defective")
                .join(&frame),
            tampered: Vec::new(),
        }
    }

    /// The shared defect map. A property of the sensor, so it is not per-stage.
    pub fn lut() -> PathBuf {
        PathBuf::from(DATA).join("01_defective/LUT_DP.npy")
    }
}

// global constants
//
// z = [GAIN, LUT_D, H_IN_GAIN, H_OUT_GAIN, H_IN_DPC, H_OUT_DPC, ROW]
//   z[0] GAIN       gain factor, read in-circuit by the gain stage
//   z[1] LUT_D      running digest of mask rows (DPC absorbs one per step)
//   z[2] H_IN_GAIN  running digest of the gain stage's input rows
//   z[3] H_OUT_GAIN running digest of the gain stage's output rows
//   z[4] H_IN_DPC   running digest of DPC's (above, cur, below) row triples
//   z[5] H_OUT_DPC  running digest of DPC's corrected rows
//   z[6] ROW        per-stage progress counter (see `advance_stage`): every
//                   stage increments it once per proving step and zeroes it
//                   on handover, so each stage counts its own steps from 0
//   z[7] H_IN_DEMOSAIC running digest of the demosaic stage's input bands
//   z[8] H_OUT_DEMOSAIC running digest of the demosaic stage's output rows
//   z[9] H_IN_WBAL running digest for the white balancing (each color channel contributes)
//   z[10]H_OUT_WBAL running digest of the white balancing output rows, i.e. of y
//   z[11] R_GAIN    white-balance gain on R, read in-circuit
//   z[12] B_GAIN    white-balance gain on B, read in-circuit
//   z[13] LUT_SPEC  digest of the specified mask; DPC checks LUT_D against
//                   it on its last step
//   z[14] D1        digest of the readout, to be signed by the anchor; the
//                   verifier compares H_IN_DPC with it
//   z[15] PC        index of the next stage. arecibo keeps its own pc
//                   outside z and leaves the first one free, so the state
//                   carries it too: every step checks it is its turn, the
//                   verifier fixes it in z_0 and expects DONE_IDX in z_N
//
// The verifier replays every chain natively over the known image + mask and
// checks the final z.
//
// TODO: This can be updated by having a single running digest shared by every
// stage — DPC absorbs (above, cur, below, mask, out) per row, then gain keeps
// absorbing (in, out) on the same d. Room for optimization.

pub const ARITY: usize = 16;

/// Named z slots. A circuit that touches slot `S` writes `z_next[S]`; every
/// other slot threads through untouched via `z.to_vec()`.
pub const Z_GAIN: usize = 0;
pub const Z_LUT_D: usize = 1;
pub const Z_H_IN_GAIN: usize = 2;
pub const Z_H_OUT_GAIN: usize = 3;
pub const Z_H_IN_DPC: usize = 4;
pub const Z_H_OUT_DPC: usize = 5;
pub const Z_ROW: usize = 6;
pub const Z_H_IN_DEMOSAIC: usize = 7;
pub const Z_H_OUT_DEMOSAIC: usize = 8;
pub const Z_H_IN_WBAL: usize = 9;
pub const Z_H_OUT_WBAL: usize = 10;
pub const Z_R_GAIN: usize = 11;
pub const Z_B_GAIN: usize = 12;
pub const Z_LUT_SPEC: usize = 13;
pub const Z_D1: usize = 14;
pub const Z_PC: usize = 15;

/// Circuit indices, numbered in pipeline order: DPC -> gain -> demosaic ->white Balancing.
/// Keeping the schedule aligned with the indices means pc starts at 0 —
/// arecibo's `initial_circuit_index` default — so no override is needed.
/// Each stage's `advance_stage` tail flips pc to the next index after that
/// stage's last step (white balancing, being last, keeps its own).
pub const DPC_IDX: usize = 0;
pub const GAIN_IDX: usize = 1;
pub const DEMOSAIC_IDX: usize = 2;
pub const WBAL_IDX: usize = 3;
pub const INDECES: &[usize] = &[DPC_IDX, GAIN_IDX, DEMOSAIC_IDX, WBAL_IDX];
/// Value of z[Z_PC] once the last stage has finished: one past the last index.
pub const DONE_IDX: usize = 4;

