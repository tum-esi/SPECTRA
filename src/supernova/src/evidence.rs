//! Native side of the evidence: the digests the anchor and the verifier
//! compute outside the circuit, the initial state z_0, and the checks the
//! verifier runs on (z_0, z_N) once the NIVC proof has verified.
//!
//! Every layout here mirrors a circuit: same packing, same absorb order.
//! `evidence_matches_the_circuits` in provenance_test.rs fails if they drift.

use crate::prelude::*;

/// The parameters the verifier expects the pipeline to run with.
pub struct Spec<'a> {
    pub gain: u8,
    pub r_gain: u8,
    pub b_gain: u8,
    pub mask: &'a [Vec<bool>],
}

fn absorb(constants: &H2<F>, d: F, elts: &[F]) -> F {
    elts.iter()
        .fold(d, |d, e| Poseidon::new_with_preimage(&[d, *e], constants).hash())
}

/// `gadgets::absorb_bytes`, natively: 31 bytes per element, little-endian.
fn absorb_bytes(constants: &H2<F>, d: F, row: &[u8]) -> F {
    let packed: Vec<F> = row
        .chunks((F::CAPACITY / 8) as usize)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(F::ZERO, |acc, &p| acc * F::from(256u64) + F::from(p as u64))
        })
        .collect();
    absorb(constants, d, &packed)
}

/// Digest of the defect mask, row by row, as DPC accumulates it in LUT_D.
pub fn mask_digest(constants: &H2<F>, mask: &[Vec<bool>]) -> F {
    mask.iter().fold(F::ZERO, |d, row| {
        absorb(constants, d, &multipack::compute_multipacking::<F>(row))
    })
}

/// Digest d_1 of the readout, in the layout DPC reads it: per row the
/// (above, cur, below) triple, rows two apart, clamped at the borders.
pub fn readout_digest(constants: &H2<F>, frame: &[Vec<u8>]) -> F {
    let h = frame.len();
    (0..h).fold(F::ZERO, |d, t| {
        let above = if t < 2 { t } else { t - 2 };
        let below = if t + 2 >= h { t } else { t + 2 };
        [above, t, below]
            .iter()
            .fold(d, |d, &row| absorb_bytes(constants, d, &frame[row]))
    })
}

/// Digest of the delivered output y: per row the (R, G, B) channel rows.
pub fn output_digest(constants: &H2<F>, r: &[Vec<u8>], g: &[Vec<u8>], b: &[Vec<u8>]) -> F {
    (0..r.len()).fold(F::ZERO, |d, t| {
        [&r[t], &g[t], &b[t]]
            .iter()
            .fold(d, |d, row| absorb_bytes(constants, d, row))
    })
}

/// z_0: the first stage as program counter, the parameters, the digest of
/// the specified mask, the digest d_1 of the readout, and every running
/// digest and counter at zero.
pub fn initial_state(constants: &H2<F>, spec: &Spec, d1: F) -> Vec<F> {
    let mut z = vec![F::ZERO; ARITY];
    z[Z_GAIN] = F::from(spec.gain as u64);
    z[Z_R_GAIN] = F::from(spec.r_gain as u64);
    z[Z_B_GAIN] = F::from(spec.b_gain as u64);
    z[Z_LUT_SPEC] = mask_digest(constants, spec.mask);
    z[Z_D1] = d1;
    z[Z_PC] = F::from(DPC_IDX as u64);
    z
}

/// The verifier's checks on the states of an accepted proof. `d1` comes with
/// z_0 (the anchor's signature on it is not implemented yet).
///
/// Only the two ends of the digest chain are checked here. The links
/// between stages are constraints of the circuits: each stage, on its last
/// step, requires its input digest to equal its predecessor's output digest,
/// and reaching DONE_IDX means every stage took that step.
pub fn check_evidence(
    constants: &H2<F>,
    z0: &[F],
    zn: &[F],
    spec: &Spec,
    (r, g, b): (&[Vec<u8>], &[Vec<u8>], &[Vec<u8>]),
) -> Result<(), String> {
    let checks = [
        (z0 == initial_state(constants, spec, z0[Z_D1]), "z_0 does not hold the specified parameters"),
        (zn[Z_H_IN_DPC] == z0[Z_D1], "first stage did not consume the readout d_1"),
        (zn[Z_H_OUT_WBAL] == output_digest(constants, r, g, b), "last stage did not produce y"),
        (zn[Z_PC] == F::from(DONE_IDX as u64), "run stopped before the last stage finished"),
    ];
    match checks.iter().find(|(ok, _)| !ok) {
        Some((_, why)) => Err(why.to_string()),
        None => Ok(()),
    }
}
