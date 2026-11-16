// DPC circuit
//
// z = [GAIN, LUT_D, H_IN_GAIN, H_OUT_GAIN, H_IN_DPC, H_OUT_DPC, ROW]
// This circuit runs first in the pipeline (pc = DPC_IDX = 0): it absorbs
// its own (above, cur, below) rows into H_IN_DPC, the mask row into LUT_D
// and the corrected row into H_OUT_DPC, then flips pc to the gain circuit
// once ROW reaches `height`. The verifier replays every one of those chains
// natively over the known image + mask and checks the final z.

use crate::{
    gadgets::{absorb, advance_stage, enforce_equal_on_last, stage_turn, num_to_bits_le, pack_pixels_into_field_elements},
    prelude::*,
};

#[derive(Clone)]
pub struct DPCCircuit {
    pub in_advice: Vec<u8>,           // current row of the defective image
    pub below_row_in_advice: Vec<u8>, // row below it
    pub above_in_advice: Vec<u8>,     // row above it
    pub out_advice: Vec<u8>,          // current row after the DPC
    pub lut: Vec<bool>,                 // this row of the dead-pixel map
    pub circuit_index: usize,         // circuit index
    pub constants: Arc<H2<F>>,        // things for the poseidon hashes
    pub width: usize,
    pub height: usize,
}

impl StepCircuit<F> for DPCCircuit {
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
        // allocate the input
        let mut in_advice_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);
        let mut in_advice_row_below: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);
        let mut in_advice_row_above: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);
        // now we allocate the output
        let mut out_advice_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);

        // The mask row. Boolean by allocation (AllocatedBit carries its own
        // b*(1-b) = 0), which is also what makes the bit-packing below
        // injective. Bound to the verifier's mask through the LUT_D chain,
        // not by re-reading z -- z[1] is a digest, not the table itself.
        let mut lut_allocated: Vec<AllocatedBit> = Vec::with_capacity(self.width);

        for i in 0..self.width {
            let lut_elem = AllocatedBit::alloc(
                cs.namespace(|| format!("Lut elem {i}")),
                Some(self.lut[i]), // The signature expects an Option<bool>
            )?;

            let cur_px_value =
                AllocatedNum::alloc(cs.namespace(|| format!("pixel value curr row {i}")), || {
                    Ok(F::from(self.in_advice[i] as u64))
                })?;

            let below_px_value =
                AllocatedNum::alloc(cs.namespace(|| format!("pixel value prev row {i}")), || {
                    Ok(F::from(self.below_row_in_advice[i] as u64))
                })?;
            let above_px_value = AllocatedNum::alloc(
                cs.namespace(|| format!("pixel value above row {i}")),
                || Ok(F::from(self.above_in_advice[i] as u64)),
            )?;

            let out_value =
                AllocatedNum::alloc(cs.namespace(|| format!("out value row {i}")), || {
                    Ok(F::from(self.out_advice[i] as u64))
                })?;

            // Range-check every input pixel to 8 bits. Load-bearing twice:
            //   - the euclidean-division constraint sums eight of these into C;
            //     unbounded field elements could wrap the modulus and make a
            //     forged quotient satisfiable,
            //   - `pack_pixels_into_field_elements` is only injective over
            //     bytes (its documented caller obligation).
            // `out` is deliberately NOT decomposed: the select below pins it to
            // either `cur` (checked here) or `quotient` (checked to 8 bits in
            // the second pass), and the border constraint pins it to `cur`, so
            // its range is inherited everywhere.
            num_to_bits_le(
                cs.namespace(|| format!("range check cur {i}")),
                &cur_px_value,
                8,
                Some(self.in_advice[i] as u64),
            )?;
            num_to_bits_le(
                cs.namespace(|| format!("range check below {i}")),
                &below_px_value,
                8,
                Some(self.below_row_in_advice[i] as u64),
            )?;
            num_to_bits_le(
                cs.namespace(|| format!("range check above {i}")),
                &above_px_value,
                8,
                Some(self.above_in_advice[i] as u64),
            )?;

            lut_allocated.push(lut_elem);
            in_advice_allocated.push(cur_px_value);
            in_advice_row_below.push(below_px_value);
            in_advice_row_above.push(above_px_value);
            out_advice_allocated.push(out_value);
        }

        // Second pass: the correction. It cannot live in the loop above --
        // at iteration i that loop has only pushed 0..i, so a neighbour at
        // i+1 does not exist yet.
        //
        // Bayer neighbourhood: the eight same-colour neighbours are two
        // columns away in this row and in the rows two above / two below
        // (NN_OFFSETS_BAYER in deadpixel.py; `above`/`below` advice are rows
        // t-2 / t+2). Hence 2..width-2: generateDefectMap never flags a
        // pixel within two of the border, so leaving those four columns
        // uncorrected is the reference behaviour, not an approximation.
        for i in 2..self.width - 2 {
            let mut cs = cs.namespace(|| format!("correction for pixel {i}"));

            // Now we need to compute the average, to do that we need to implement the euclidian division algorithm:
            // for x , y in N: Exists q,r such that x = qy + r
            // div(x,y) = [x/y]
            // x mod y = x-y[x/y]
            //
            // Native u16, not u8: 8 * 255 = 2040 overflows a byte.
            let sum_above = self.above_in_advice[i - 2] as u16
                + self.above_in_advice[i] as u16
                + self.above_in_advice[i + 2] as u16;

            let sum_below = self.below_row_in_advice[i - 2] as u16
                + self.below_row_in_advice[i] as u16
                + self.below_row_in_advice[i + 2] as u16;

            // The centre pixel is excluded -- it is the dead one being
            // replaced. Eight neighbours, matching NN_OFFSETS_BAYER in
            // deadpixel.py.
            let sum_cur = self.in_advice[i - 2] as u16 + self.in_advice[i + 2] as u16;

            let sum: u16 = sum_above + sum_below + sum_cur;
            let denominator: u16 = 8; // we have 8 pixel values that we want to average

            // Native / and % give the witness. On their own they prove nothing;
            // the constraints below are what makes them binding.
            let quotient_value = sum / denominator;
            let reminder_value = sum % denominator;

            let quotient = AllocatedNum::alloc(cs.namespace(|| "quotient"), || {
                Ok(F::from(quotient_value as u64))
            })?;
            let reminder = AllocatedNum::alloc(cs.namespace(|| "reminder"), || {
                Ok(F::from(reminder_value as u64))
            })?;

            // 8*quotient + reminder = sum of the 8 neighbours, in one row.
            // Both sides are linear, so no intermediate `sum` variable is
            // needed -- the neighbour total goes straight into C.
            cs.enforce(
                || "euclidean division over the neighbour sum",
                |lc| lc + (F::from(8u64), quotient.get_variable()) + reminder.get_variable(),
                |lc| lc + CS::one(),
                |lc| {
                    lc + in_advice_row_above[i - 2].get_variable()
                        + in_advice_row_above[i].get_variable()
                        + in_advice_row_above[i + 2].get_variable()
                        + in_advice_allocated[i - 2].get_variable()
                        + in_advice_allocated[i + 2].get_variable()
                        + in_advice_row_below[i - 2].get_variable()
                        + in_advice_row_below[i].get_variable()
                        + in_advice_row_below[i + 2].get_variable()
                },
            );

            // reminder < 8. Without it (q-1, r+8) satisfies the identity just
            // as well, and the quotient stops being unique.
            num_to_bits_le(
                cs.namespace(|| "reminder < 8"),
                &reminder,
                3,
                Some(reminder_value as u64),
            )?;

            // quotient < 256. Free for an honest prover since sum <= 2040, but
            // it is what stops a dishonest one solving the identity by wrapping
            // the modulus.
            num_to_bits_le(
                cs.namespace(|| "quotient < 256"),
                &quotient,
                8,
                Some(quotient_value as u64),
            )?;

            // out[i] = lut[i] ? quotient : in[i], via the select
            // lut * (quotient - in) = out - in.
            cs.enforce(
                || "lut * (q - in) = out - in",
                |lc| lc + (F::ONE, lut_allocated[i].get_variable()),
                |lc| lc + quotient.get_variable() - in_advice_allocated[i].get_variable(),
                |lc| {
                    lc + out_advice_allocated[i].get_variable()
                        - in_advice_allocated[i].get_variable()
                },
            );
        }

        // Border passthrough: the correction loop never touches the two
        // columns at each edge, so without these constraints those outputs
        // are free advice that still gets absorbed into H_OUT_DPC. out = in
        // is the reference behaviour (generateDefectMap never flags a pixel
        // within two of the border), here made binding instead of assumed.
        for i in [0, 1, self.width - 2, self.width - 1] {
            cs.enforce(
                || format!("border passthrough {i}"),
                |lc| {
                    lc + out_advice_allocated[i].get_variable()
                        - in_advice_allocated[i].get_variable()
                },
                |lc| lc + CS::one(),
                |lc| lc,
            );
        }

        // Bind the mask row: pack the (boolean) lut bits and absorb them into
        // the LUT_D chain at z[1]. The verifier replays this natively with
        // `compute_multipacking` over the known defect map, so a prover who
        // flips any lut bit -- "this healthy pixel is dead, replace it" --
        // lands on a different final z[1] and is rejected.
        let lut_bools: Vec<Boolean> = lut_allocated.iter().cloned().map(Boolean::from).collect();
        let mut packed_lut: Vec<AllocatedNum<F>> = Vec::new();
        for (ci, chunk) in lut_bools.chunks(F::CAPACITY as usize).enumerate() {
            packed_lut.push(multipack::pack_bits(
                cs.namespace(|| format!("pack lut chunk {ci}")),
                chunk,
            )?);
        }
        let lut_d_next = absorb(
            cs.namespace(|| "absorb lut row"),
            &self.constants,
            z[1].clone(),
            &packed_lut,
        )?;

        // Update the digests. H_IN_DPC absorbs all three input rows, not just
        // the current one: above/below are fresh allocations in this step, and
        // nothing else ties them to the `cur` row of the neighbouring steps.
        // Absorbing them makes the verifier's native replay that tie -- fake
        // neighbours produce a different final digest. Absorb order per row is
        // (above, cur, below); the replay must match it exactly.
        let packed_above =
            pack_pixels_into_field_elements(cs.namespace(|| "pack above"), &in_advice_row_above)?;
        let packed_cur =
            pack_pixels_into_field_elements(cs.namespace(|| "pack input"), &in_advice_allocated)?;
        let packed_below =
            pack_pixels_into_field_elements(cs.namespace(|| "pack below"), &in_advice_row_below)?;
        let packed_out =
            pack_pixels_into_field_elements(cs.namespace(|| "pack output"), &out_advice_allocated)?;

        let mut h_in_dpc = z[4].clone();
        h_in_dpc = absorb(
            cs.namespace(|| "absorb above row"),
            &self.constants,
            h_in_dpc,
            &packed_above,
        )?;
        h_in_dpc = absorb(
            cs.namespace(|| "absorb cur row"),
            &self.constants,
            h_in_dpc,
            &packed_cur,
        )?;
        h_in_dpc = absorb(
            cs.namespace(|| "absorb below row"),
            &self.constants,
            h_in_dpc,
            &packed_below,
        )?;
        let h_out_dpc = absorb(
            cs.namespace(|| "absorb out row"),
            &self.constants,
            z[5].clone(),
            &packed_out,
        )?;

        // Stage bookkeeping: count this row, and on the last one flip pc to
        // the gain circuit and zero ROW so the gain stage starts counting
        // its own rows from 0.
        let (pc_next, row_next, is_last) = advance_stage(
            cs.namespace(|| "stage handover"),
            &z[6],
            self.height,
            DPC_IDX,
            GAIN_IDX,
        )?;

        // The mask is a parameter too large for the state: z_0 holds its
        // digest, and once the last row is absorbed the accumulated LUT_D
        // must equal it.
        enforce_equal_on_last(
            cs.namespace(|| "mask digest matches the specified one"),
            &is_last,
            &lut_d_next,
            &z[Z_LUT_SPEC],
        );

        let z_pc_next = stage_turn(cs.namespace(|| "state pc"), &z[Z_PC], &is_last, DPC_IDX)?;
        let mut z_next = z.to_vec();
        z_next[Z_PC] = z_pc_next;
        z_next[Z_LUT_D] = lut_d_next;
        z_next[Z_H_IN_DPC] = h_in_dpc;
        z_next[Z_H_OUT_DPC] = h_out_dpc;
        z_next[Z_ROW] = row_next;
        Ok((Some(pc_next), z_next))
    }
}
