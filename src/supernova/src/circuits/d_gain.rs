// Example program that wants to build from simple to complicated
// z = [GAIN, LUT_D, H_IN_GAIN, H_OUT_GAIN, H_IN_DPC, H_OUT_DPC, ROW]
// This circuit touches:
// z[0] = gain factor (read in-circuit, the read is the binding)
// z[2] = running digest of the gain stage's input rows
// z[3] = running digest of the gain stage's output rows
// and threads the DPC slots (z[1], z[4], z[5]) and the row counter (z[6]).

use crate::{
    gadgets::{
        absorb_bytes, advance_stage, enforce_equal_on_last, stage_turn, alloc_bytes, alloc_bytes_unchecked, byte_hint,
        enforce_saturating_mul, num_to_bits_le,
    },
    prelude::*,
};

#[derive(Clone)]
pub struct DGainCircuit {
    pub in_advice: Vec<u8>, // Now we have as an input only a single row, same as output.
    pub gain: u8,
    pub width: usize,
    /// Rows in the frame = proving steps of this stage. Read by
    /// `advance_stage` to know when to hand pc to the demosaic circuit.
    pub height: usize,
    pub out_advice: Vec<u8>,
    pub circuit_index: usize,
    pub constants: Arc<H2<F>>,
}

impl StepCircuit<F> for DGainCircuit {
    fn arity(&self) -> usize {
        ARITY
    }

    fn circuit_index(&self) -> usize {
        // Must be the index the driver handed out: `EnforcingStepCircuit`
        // constrains `pc == circuit_index()`, so a hardcoded 0 would make every
        // step that is not pc = 0 unsatisfiable once a second circuit exists.
        self.circuit_index
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        _pc: Option<&AllocatedNum<F>>,
        z: &[AllocatedNum<F>],
    ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
        // We allocate things only if we need them to be constrained. We allocate each pixel per row and we constrain their value.
        // The fixed-gain shape uses the checked top bit of px to select
        // doubled or saturated output. Other gains retain the generic shape:
        //   px*g decomposed into 16 bits (low byte `low`, high byte `high`)
        //   ov = (high != 0)            -- overflow flag, inverse trick
        //   ov * (255 - low) = out - low -- select: out = ov ? 255 : low

        // read, never re-allocate — the read IS the binding
        let gain_allocated = &z[Z_GAIN];

        if self.gain == 2 {
            // The optimized shape is only sound for exactly two, so bind the
            // public state to the constant rather than merely trusting the
            // circuit struct's Rust value.
            cs.enforce(
                || "fixed gain is two",
                |lc| lc + gain_allocated.get_variable() - (F::from(2u64), CS::one()),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            // The generic product relation assumes that gain is a byte. Hint
            // from z[0] itself so an out-of-range public gain is rejected.
            num_to_bits_le(
                cs.namespace(|| "range check gain"),
                gain_allocated,
                8,
                byte_hint(gain_allocated),
            )?;
        }

        // Inputs range-checked to 8 bits (the packing below is only
        // injective over bytes, and the saturating multiply only sound over
        // them). Outputs are pinned to `low` or 255 by the select, so their
        // range is inherited.
        let out_row = alloc_bytes_unchecked(cs.namespace(|| "out"), &self.out_advice)?;
        let in_row = if self.gain == 2 {
            let mut in_row = Vec::with_capacity(self.in_advice.len());
            for (i, (&value, out)) in self.in_advice.iter().zip(&out_row).enumerate() {
                let mut pixel_cs = cs.namespace(|| format!("px {i}"));
                let px = AllocatedNum::alloc(pixel_cs.namespace(|| "input"), || {
                    Ok(F::from(value as u64))
                })?;
                let bits = num_to_bits_le(
                    pixel_cs.namespace(|| "decompose input"),
                    &px,
                    8,
                    Some(value as u64),
                )?;

                // For an 8-bit px, bit 7 is one exactly when 2*px exceeds
                // 255. Enforce out = 2*px below 128 and out = 255 otherwise:
                //   bit7 * (255 - 2*px) = out - 2*px.
                pixel_cs.enforce(
                    || "fixed gain",
                    |lc| lc + bits[7].get_variable(),
                    |lc| lc + (F::from(255u64), CS::one()) - (F::from(2u64), px.get_variable()),
                    |lc| lc + out.get_variable() - (F::from(2u64), px.get_variable()),
                );
                in_row.push(px);
            }
            in_row
        } else {
            let in_row = alloc_bytes(cs.namespace(|| "in"), &self.in_advice)?;
            for (i, (px, out)) in in_row.iter().zip(&out_row).enumerate() {
                enforce_saturating_mul(
                    cs.namespace(|| format!("px {i}")),
                    px,
                    gain_allocated,
                    out,
                )?;
            }
            in_row
        };

        // A pixel is 8 bits, a field element ~255: packing 31 pixels per
        // element before hashing saves ~31x the Poseidon calls.
        let h_in_next = absorb_bytes(
            cs.namespace(|| "absorb in row"),
            &self.constants,
            z[Z_H_IN_GAIN].clone(),
            &in_row,
        )?;
        let h_out_next = absorb_bytes(
            cs.namespace(|| "absorb out row"),
            &self.constants,
            z[Z_H_OUT_GAIN].clone(),
            &out_row,
        )?;
        // Count this row (ROW was zeroed by DPC's handover), and after the
        // last one flip pc to the demosaic circuit and zero ROW again.
        let (pc_next, row_next, is_last) = advance_stage(
            cs.namespace(|| "stage handover"),
            &z[Z_ROW],
            self.height,
            GAIN_IDX,
            DEMOSAIC_IDX,
        )?;

        // Link to the previous stage: once every row is absorbed, the input
        // digest must equal the output digest DPC recorded.
        enforce_equal_on_last(
            cs.namespace(|| "input is the previous stage's output"),
            &is_last,
            &h_in_next,
            &z[Z_H_OUT_DPC],
        );
        let z_pc_next = stage_turn(cs.namespace(|| "state pc"), &z[Z_PC], &is_last, GAIN_IDX)?;
        let mut z_next = z.to_vec();
        z_next[Z_PC] = z_pc_next;
        z_next[Z_H_IN_GAIN] = h_in_next;
        z_next[Z_H_OUT_GAIN] = h_out_next;
        z_next[Z_ROW] = row_next;
        Ok((Some(pc_next), z_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;

    fn check(
        inputs: Vec<u8>,
        outputs: Vec<u8>,
        circuit_gain: u8,
        public_gain: u64,
    ) -> (bool, usize) {
        let mut cs = TestConstraintSystem::<F>::new();
        let z = (0..ARITY)
            .map(|i| {
                let value = match i {
                    Z_GAIN => F::from(public_gain),
                    Z_PC => F::from(GAIN_IDX as u64),
                    _ => F::ZERO,
                };
                AllocatedNum::alloc_input(cs.namespace(|| format!("z{i}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        let circuit = DGainCircuit {
            width: inputs.len(),
            height: 2,
            in_advice: inputs,
            out_advice: outputs,
            gain: circuit_gain,
            circuit_index: GAIN_IDX,
            constants: Arc::new(H2::<F>::new()),
        };
        circuit.synthesize(&mut cs, None, &z).unwrap();
        (cs.is_satisfied(), cs.num_constraints())
    }

    #[test]
    fn fixed_two_covers_every_byte_and_rejects_wrong_statements() {
        let inputs: Vec<u8> = (0..=255).collect();
        let outputs: Vec<u8> = inputs
            .iter()
            .map(|&x| (u16::from(x) * 2).min(255) as u8)
            .collect();
        let (valid, fixed_constraints) = check(inputs.clone(), outputs.clone(), 2, 2);
        assert!(valid);

        for &index in &[0, 1, 127, 128, 255] {
            let mut wrong = outputs.clone();
            wrong[index] = wrong[index].wrapping_add(1);
            assert!(
                !check(inputs.clone(), wrong, 2, 2).0,
                "wrong output at {index}"
            );
        }
        assert!(
            !check(inputs.clone(), outputs, 2, 3).0,
            "public gain must be two"
        );

        let generic_outputs = inputs
            .iter()
            .map(|&x| (u16::from(x) * 3).min(255) as u8)
            .collect();
        let (generic_valid, generic_constraints) = check(inputs, generic_outputs, 3, 3);
        assert!(generic_valid);
        assert!(fixed_constraints < generic_constraints);
    }

    #[test]
    fn other_gains_keep_the_generic_relation() {
        let inputs = vec![0, 1, 85, 86, 127, 128, 255];
        let outputs: Vec<_> = inputs
            .iter()
            .map(|&x| (u16::from(x) * 3).min(255) as u8)
            .collect();
        assert!(check(inputs.clone(), outputs.clone(), 3, 3).0);
        let mut wrong = outputs;
        wrong[2] = wrong[2].wrapping_add(1);
        assert!(!check(inputs, wrong, 3, 3).0);
    }
}
