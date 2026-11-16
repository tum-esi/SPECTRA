use crate::{
    gadgets::{absorb_bytes, advance_stage, enforce_equal_on_last, stage_turn, num_to_bits_le},
    prelude::*,
};



#[derive(Clone)]
pub struct DemosaicCircuit {
    pub r1_advice: Vec<u8>, // first row
    pub r2_advice: Vec<u8>, // second row
    pub width: usize,
    /// Rows in the frame. This stage eats two rows per step, so its step
    /// count -- what `advance_stage` counts against -- is `height / 2`.
    pub height: usize,
    pub circuit_index: usize,
    pub constants: Arc<H2<F>>,
}

impl StepCircuit<F> for DemosaicCircuit {
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
        // Allocate the input samples. The output channels reuse these
        // variable handles, so no separate RGB witness is needed.
        let mut r1_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);
        let mut r2_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(self.width);
        // Two output rows per channel, concatenated row-major.
        let mut out_r_channel_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(2 * self.width);
        let mut out_g_channel_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(2 * self.width);
        let mut out_b_channel_allocated: Vec<AllocatedNum<F>> = Vec::with_capacity(2 * self.width);

        for i in 0..self.width {
            // Allocate the R - G Sequences
            let r1_pixel = AllocatedNum::alloc(cs.namespace(|| format!("r1_pixel_{}", i)), || {
                Ok(F::from(self.r1_advice[i] as u64))
            })?;

            // Allocate the G - B Sequences
            let r2_pixel = AllocatedNum::alloc(cs.namespace(|| format!("r2_pixel_{}", i)), || {
                Ok(F::from(self.r2_advice[i] as u64))
            })?;

            // Range-check the input pixels to 8 bits:
            // `pack_pixels_into_field_elements` (the digest absorbs later)
            // is only injective over bytes. Output handles reuse these
            // checked inputs, so their range is inherited.
            num_to_bits_le(
                cs.namespace(|| format!("range check r1 {i}")),
                &r1_pixel,
                8,
                Some(self.r1_advice[i] as u64),
            )?;
            num_to_bits_le(
                cs.namespace(|| format!("range check r2 {i}")),
                &r2_pixel,
                8,
                Some(self.r2_advice[i] as u64),
            )?;

            r1_allocated.push(r1_pixel);
            r2_allocated.push(r2_pixel);
        }

        // Wire every output pixel to its source sample. The
        // Bayer pair looks like
        //
        //     r1: R  G1 R  G1 ...
        //     r2: G2 B  G2 B  ...
        //
        // and each 2x2 superpixel expands to (stages/demosaic.py):
        //
        //     R G1 B    R G1 B      <- output row 1
        //     R G2 B    R G2 B      <- output row 2
        //
        // Which sample feeds column k depends only on the parity of k: on
        // an even column R and G2 sit on the column itself and G1/B one to
        // the right; on an odd column G1 and B sit on the column and R/G2
        // one to the left. Both arms of a pair therefore name the same
        // four samples -- exactly the expansion above. What goes into the
        // single G channel is G1 for output row 1 and G2 for output row 2;
        // R and B are the same in both rows. k and the row index are
        // synthesis-time constants, so all of this is static wiring: no
        // selector gadget, the parity never touches the constraints.
        //
        // The even arm reads k + 1, hence the whole-superpixel width
        // requirement (the python reference raises on odd dims too).
        assert_eq!(self.width % 2, 0, "width must be even");
        for row in 0..2 {
            for k in 0..self.width {
                let (r_src, g1_src, g2_src, b_src) = if k % 2 == 0 {
                    (
                        &r1_allocated[k],     // R on this column
                        &r1_allocated[k + 1], // G1 to the right
                        &r2_allocated[k],     // G2 below
                        &r2_allocated[k + 1], // B below-right
                    )
                } else {
                    (
                        &r1_allocated[k - 1], // R to the left
                        &r1_allocated[k],     // G1 on this column
                        &r2_allocated[k - 1], // G2 below-left
                        &r2_allocated[k],     // B below
                    )
                };

                let g_src = if row == 0 { g1_src } else { g2_src };
                out_r_channel_allocated.push(r_src.clone());
                out_g_channel_allocated.push(g_src.clone());
                out_b_channel_allocated.push(b_src.clone());
            }
        }

        // Digest layout. Input: row 1 then row 2, each packed on its own --
        // exactly what the gain stage's H_OUT absorbs per row, so
        // H_IN_DEMOSAIC == H_OUT_GAIN. Output: row-major, (R, G, B) of
        // output row 1 then of output row 2, each channel row packed on its
        // own -- exactly what white balance's H_IN absorbs per row, so
        // H_OUT_DEMOSAIC == H_IN_WBAL.
        //
        // Chunk boundaries (31 bytes) are part of the layout: packing a
        // 2*width band is NOT packing its two rows separately unless width
        // is a multiple of 31. Both sides of each link must pack one row
        // per call, and so must the verifier's native replay.
        let mut h_in_demosaic = z[7].clone();
        for (name, row) in [("1", &r1_allocated), ("2", &r2_allocated)] {
            h_in_demosaic = absorb_bytes(
                cs.namespace(|| format!("absorb in row {name}")),
                &self.constants,
                h_in_demosaic,
                row,
            )?;
        }

        let mut h_out_demosaic = z[8].clone();
        for row in 0..2 {
            let cols = row * self.width..(row + 1) * self.width;
            for (name, channel) in [
                ("R", &out_r_channel_allocated),
                ("G", &out_g_channel_allocated),
                ("B", &out_b_channel_allocated),
            ] {
                h_out_demosaic = absorb_bytes(
                    cs.namespace(|| format!("absorb out row {row} {name}")),
                    &self.constants,
                    h_out_demosaic,
                    &channel[cols.clone()],
                )?;
            }
        }

        // ROW counts the bands (height/2 steps); after the last one pc
        // hands over to white balance and ROW wraps to 0.
        let (pc_next, row_next, is_last) = advance_stage(
            cs.namespace(|| "stage handover"),
            &z[6],
            self.height / 2,
            DEMOSAIC_IDX,
            WBAL_IDX,
        )?;

        // Link to the previous stage: once every row is absorbed, the input
        // digest must equal the output digest the gain stage recorded.
        enforce_equal_on_last(
            cs.namespace(|| "input is the previous stage's output"),
            &is_last,
            &h_in_demosaic,
            &z[Z_H_OUT_GAIN],
        );
        let z_pc_next = stage_turn(cs.namespace(|| "state pc"), &z[Z_PC], &is_last, DEMOSAIC_IDX)?;
        let mut z_next = z.to_vec();
        z_next[Z_PC] = z_pc_next;
        z_next[Z_ROW] = row_next;
        z_next[Z_H_IN_DEMOSAIC] = h_in_demosaic;
        z_next[Z_H_OUT_DEMOSAIC] = h_out_demosaic;
        Ok((Some(pc_next), z_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;

    fn synthesize(circuit: &DemosaicCircuit) -> (TestConstraintSystem<F>, Vec<AllocatedNum<F>>) {
        let mut cs = TestConstraintSystem::<F>::new();
        let z: Vec<_> = (0..ARITY)
            .map(|i| {
                let v = if i == Z_PC { F::from(DEMOSAIC_IDX as u64) } else { F::ZERO };
                AllocatedNum::alloc_input(cs.namespace(|| format!("z{i}")), || Ok(v)).unwrap()
            })
            .collect();
        let (_, next) = circuit.synthesize(&mut cs, None, &z).unwrap();
        (cs, next)
    }

    fn circuit(width: usize) -> DemosaicCircuit {
        DemosaicCircuit {
            r1_advice: (0..width).map(|i| (3 * i + 1) as u8).collect(),
            r2_advice: (0..width).map(|i| (5 * i + 2) as u8).collect(),
            width,
            height: 4,
            circuit_index: DEMOSAIC_IDX,
            constants: Arc::new(H2::<F>::new()),
        }
    }

    #[test]
    fn wired_outputs_have_no_copy_constraints() {
        let (cs, _) = synthesize(&circuit(8));
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        // Row-major out absorb: 6 one-chunk rows at width 8, not 3 one-chunk
        // bands. Equal chunk count at width 1920 (2 * 62 = ceil(3840 / 31)).
        // Plus the 2 constraints of the pc carried in the state (`stage_turn`)
        // and 1 for the link to the gain stage.
        assert_eq!(cs.num_constraints(), 2064);
    }

    #[test]
    fn wired_output_digest_matches_row_major_rgb_layout_across_pack_chunks() {
        // Width 32 crosses the 31-byte packing boundary inside each channel row.
        let width = 32;
        let r1: Vec<u8> = (0..width).map(|i| (3 * i + 1) as u8).collect();
        let r2: Vec<u8> = (0..width).map(|i| (5 * i + 2) as u8).collect();
        let constants = Arc::new(H2::<F>::new());
        let circuit = DemosaicCircuit {
            r1_advice: r1.clone(),
            r2_advice: r2.clone(),
            width,
            height: 4,
            circuit_index: DEMOSAIC_IDX,
            constants: constants.clone(),
        };
        let (cs, next) = synthesize(&circuit);
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());

        let mut channels = [
            vec![0u8; 2 * width],
            vec![0u8; 2 * width],
            vec![0u8; 2 * width],
        ];
        for row in 0..2 {
            for k in 0..width {
                let even = k & !1;
                let i = row * width + k;
                channels[0][i] = r1[even];
                channels[1][i] = if row == 0 { r1[even + 1] } else { r2[even] };
                channels[2][i] = r2[even + 1];
            }
        }
        let mut expected = F::ZERO;
        for row in 0..2 {
            for channel in &channels {
                let channel_row = &channel[row * width..(row + 1) * width];
                for chunk in channel_row.chunks((F::CAPACITY / 8) as usize) {
                    let packed = chunk.iter().rev().fold(F::ZERO, |acc, &byte| {
                        acc * F::from(256u64) + F::from(byte as u64)
                    });
                    expected = Poseidon::new_with_preimage(&[expected, packed], &constants).hash();
                }
            }
        }
        assert_eq!(next[8].get_value(), Some(expected));
    }
}
