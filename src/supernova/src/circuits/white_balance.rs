// White balance: per-channel digital gain on R and B, G passes through.
//
//   R' = min(R * r_gain, 255)
//   G' = G
//   B' = min(B * b_gain, 255)
//
// This circuit touches:
//   z[9]  H_IN_WBAL  absorbs (R, G, B) per row -- the same layout demosaic's
//                    H_OUT absorbs, so H_IN_WBAL == H_OUT_DEMOSAIC links
//                    the two stages
//   z[10] H_OUT_WBAL absorbs (R', G, B') per row, so it is the digest of y
// and threads everything else. Last stage: pc stays on WBAL_IDX.
//
// The gains are parameters in the state, z[11] and z[12], read in-circuit
// like z[0] for the gain stage: the verifier fixes them in z_0.

use crate::{
    gadgets::{
        absorb_bytes, advance_stage, enforce_equal_on_last, stage_turn, alloc_bytes, alloc_bytes_unchecked, byte_hint,
        enforce_saturating_mul, num_to_bits_le,
    },
    prelude::*,
};

#[derive(Clone)]
pub struct WhiteBalancingCircuit {
    /// Current row of each input plane.
    pub in_r: Vec<u8>,
    pub in_g: Vec<u8>,
    pub in_b: Vec<u8>,

    /// Current row of the balanced R and B planes. No G output is supplied:
    /// the circuit absorbs the *input* G row into H_OUT_WBAL, which is the
    /// passthrough constraint with zero extra allocations.
    pub out_r: Vec<u8>,
    pub out_b: Vec<u8>,

    pub width: usize,
    pub height: usize,

    pub circuit_index: usize,
    pub constants: Arc<H2<F>>,
}

impl StepCircuit<F> for WhiteBalancingCircuit {
    fn arity(&self) -> usize {
        ARITY
    }

    fn circuit_index(&self) -> usize {
        self.circuit_index
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        _pc: Option<&AllocatedNum<F>>,
        z: &[AllocatedNum<F>],
    ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
        // Gains: 8-bit range check is load-bearing, the saturating multiply
        // is only sound over bytes (see `enforce_saturating_mul`).
        // Read from the state, never re-allocated: the read is the binding.
        let (r_gain, b_gain) = (&z[Z_R_GAIN], &z[Z_B_GAIN]);
        for (name, gain) in [("r gain", r_gain), ("b gain", b_gain)] {
            num_to_bits_le(cs.namespace(|| format!("range check {name}")), gain, 8, byte_hint(gain))?;
        }

        let in_r = alloc_bytes(cs.namespace(|| "in R"), &self.in_r)?;
        let in_g = alloc_bytes(cs.namespace(|| "in G"), &self.in_g)?;
        let in_b = alloc_bytes(cs.namespace(|| "in B"), &self.in_b)?;

        // Outputs are pinned to `low` or 255 by the saturating select, so
        // their range is inherited; no decomposition needed.
        let out_r = alloc_bytes_unchecked(cs.namespace(|| "out R"), &self.out_r)?;
        let out_b = alloc_bytes_unchecked(cs.namespace(|| "out B"), &self.out_b)?;

        for (name, gain, input, output) in [("R", r_gain, &in_r, &out_r), ("B", b_gain, &in_b, &out_b)] {
            for (i, (px, out)) in input.iter().zip(output).enumerate() {
                enforce_saturating_mul(cs.namespace(|| format!("{name} px {i}")), px, gain, out)?;
            }
        }

        // Absorb order per step is (R, G, B) into H_IN and (R', G, B') into
        // H_OUT; `evidence::output_digest` must match exactly.
        let mut h_in = z[Z_H_IN_WBAL].clone();
        for (name, row) in [("R", &in_r), ("G", &in_g), ("B", &in_b)] {
            h_in = absorb_bytes(cs.namespace(|| format!("absorb in {name}")), &self.constants, h_in, row)?;
        }

        let mut h_out = z[Z_H_OUT_WBAL].clone();
        for (name, row) in [("R", &out_r), ("G", &in_g), ("B", &out_b)] {
            h_out = absorb_bytes(cs.namespace(|| format!("absorb out {name}")), &self.constants, h_out, row)?;
        }

        // Last stage: pc stays put, ROW still counts and wraps to 0.
        let (pc_next, row_next, is_last) = advance_stage(
            cs.namespace(|| "stage handover"),
            &z[Z_ROW],
            self.height,
            WBAL_IDX,
            WBAL_IDX,
        )?;

        // Link to the previous stage: once every row is absorbed, the input
        // digest must equal the output digest demosaic recorded.
        enforce_equal_on_last(
            cs.namespace(|| "input is the previous stage's output"),
            &is_last,
            &h_in,
            &z[Z_H_OUT_DEMOSAIC],
        );
        let z_pc_next = stage_turn(cs.namespace(|| "state pc"), &z[Z_PC], &is_last, WBAL_IDX)?;
        let mut z_next = z.to_vec();
        z_next[Z_PC] = z_pc_next;
        z_next[Z_ROW] = row_next;
        z_next[Z_H_IN_WBAL] = h_in;
        z_next[Z_H_OUT_WBAL] = h_out;
        Ok((Some(pc_next), z_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;

    const W: usize = 40; // > 31, so a row packs into two field elements

    fn balance(px: &[u8], gain: u8) -> Vec<u8> {
        px.iter().map(|&p| (p as u16 * gain as u16).min(255) as u8).collect()
    }

    fn circuit(r_gain: u8, b_gain: u8) -> WhiteBalancingCircuit {
        let in_r: Vec<u8> = (0..W).map(|i| (i * 6) as u8).collect();
        let in_g: Vec<u8> = (0..W).map(|i| (i * 3) as u8).collect();
        let in_b: Vec<u8> = (0..W).map(|i| (255 - i * 5) as u8).collect();
        WhiteBalancingCircuit {
            out_r: balance(&in_r, r_gain),
            out_b: balance(&in_b, b_gain),
            in_r,
            in_g,
            in_b,
            width: W,
            height: 4,
            circuit_index: WBAL_IDX,
            constants: Arc::new(H2::<F>::new()),
        }
    }

    fn synth(c: &WhiteBalancingCircuit, r_gain: u8, b_gain: u8) -> TestConstraintSystem<F> {
        let mut cs = TestConstraintSystem::<F>::new();
        let z: Vec<AllocatedNum<F>> = (0..ARITY)
            .map(|i| {
                let v = match i {
                    Z_R_GAIN => r_gain as u64,
                    Z_B_GAIN => b_gain as u64,
                    Z_PC => WBAL_IDX as u64,
                    _ => i as u64,
                };
                AllocatedNum::alloc(cs.namespace(|| format!("z {i}")), || Ok(F::from(v))).unwrap()
            })
            .collect();
        c.synthesize(&mut cs.namespace(|| "wb"), None, &z).unwrap();
        cs
    }

    #[test]
    fn honest_row_satisfies() {
        // Gain 2 saturates the top of the R ramp and gain 1 leaves B alone.
        let cs = synth(&circuit(2, 1), 2, 1);
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        println!("wb constraints for width {W}: {}", cs.num_constraints());
    }

    #[test]
    fn tampered_output_rejected() {
        let mut c = circuit(2, 3);
        c.out_b[7] ^= 1;
        let cs = synth(&c, 2, 3);
        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("wb/B px 7/ov * (255 - low) = out - low"));
    }

    #[test]
    fn unsaturated_output_rejected() {
        // Claim the raw product instead of the clamp where it overflows.
        let mut c = circuit(2, 1);
        assert_eq!(c.in_r[W - 1], 234);
        c.out_r[W - 1] = (234u16 * 2 % 256) as u8;
        let cs = synth(&c, 2, 1);
        assert!(!cs.is_satisfied());
    }

    #[test]
    fn gain_other_than_the_one_in_the_state_rejected() {
        // Outputs computed with gain 3 on R, state says 2.
        let cs = synth(&circuit(3, 1), 2, 1);
        assert!(!cs.is_satisfied());
    }
}
