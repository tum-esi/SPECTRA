#[cfg(test)]
mod tests {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;

    const W: usize = 8;

    // Fixture: constant 10s except cur[3] = 50 (the dead pixel) and
    // cur[4] = 11 (to make the j=3 neighbour sum 81, NOT divisible by 8:
    // corr = 10, r = 1 — exercises the remainder path).
    const ABOVE: [u8; W] = [10; W];
    const BELOW: [u8; W] = [10; W];
    const CUR: [u8; W] = [10, 10, 10, 50, 11, 10, 10, 10];
    const MASK: [bool; W] = [false, false, false, true, false, false, false, false];
    /// What the python pipeline would produce: dead pixel 50 -> floor(81/8) = 10.
    const EXPECTED_OUT: [u8; W] = [10, 10, 10, 10, 11, 10, 10, 10];

    /// Synthesize one DPC step over arbitrary rows.
    fn synth_rows(
        above: &[u8],
        cur: &[u8],
        below: &[u8],
        mask: &[bool],
        out: Option<Vec<u8>>,
    ) -> (
        TestConstraintSystem<F>,
        Vec<AllocatedNum<F>>,
        AllocatedNum<F>,
        Arc<H2<F>>,
    ) {
        let constants: Arc<H2<F>> = Arc::new(H2::<F>::new());
        let mut cs = TestConstraintSystem::<F>::new();

        let z = vec![
            AllocatedNum::alloc_input(cs.namespace(|| "step_in"), || Ok(F::ZERO)).unwrap(),
            AllocatedNum::alloc_input(cs.namespace(|| "d_in"), || Ok(F::ZERO)).unwrap(),
        ];

        let circuit = DPCCircuit {
            advice_in: Some(cur.to_vec()),
            row_above: Some(above.to_vec()),
            row_below: Some(below.to_vec()),
            mask: Some(mask.to_vec()),
            out_advice: out,
            constants: constants.clone(),
            width: cur.len(),
        };

        let (pc_next, z_next) = circuit
            .synthesize(&mut cs.namespace(|| "step 0"), None, &z)
            .unwrap();

        (cs, z_next, pc_next.unwrap(), constants)
    }

    /// Synthesize one DPC step over the fixture (or overridden mask/out).
    fn synth(
        mask: [bool; W],
        out: Option<Vec<u8>>,
    ) -> (
        TestConstraintSystem<F>,
        Vec<AllocatedNum<F>>,
        AllocatedNum<F>,
        Arc<H2<F>>,
    ) {
        synth_rows(&ABOVE, &CUR, &BELOW, &mask, out)
    }

    /// The verifier's native digest replay: same schedule as the circuit —
    /// above, cur, below, mask, out.
    fn replay_rows(
        constants: &H2<F>,
        above: &[u8],
        cur: &[u8],
        below: &[u8],
        mask: &[bool],
        out: &[u8],
    ) -> F {
        let mut d = F::ZERO;
        d = absorb_row_native(constants, d, above);
        d = absorb_row_native(constants, d, cur);
        d = absorb_row_native(constants, d, below);
        d = absorb_mask_native(constants, d, mask);
        d = absorb_row_native(constants, d, out);
        d
    }

    fn replay(constants: &H2<F>, mask: &[bool], out: &[u8]) -> F {
        replay_rows(constants, &ABOVE, &CUR, &BELOW, mask, out)
    }

    /// Happy path: correct claimed output satisfies the system, the digest
    /// matches the native replay, and the bookkeeping advances.
    #[test]
    fn corrects_dead_pixel() {
        let (cs, z_next, pc_next, constants) = synth(MASK, Some(EXPECTED_OUT.to_vec()));

        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        assert_eq!(
            z_next[1].get_value().unwrap(),
            replay(&constants, &MASK, &EXPECTED_OUT),
            "digest drift vs native replay"
        );
        assert_eq!(z_next[0].get_value().unwrap(), F::ONE);
        // step 0 is not the last row: pc stays on DPC (1).
        assert_eq!(pc_next.get_value().unwrap(), F::ONE);
    }

    /// Prover-derived output (out_advice = None) must equal the python
    /// pipeline's answer — checked through the digest.
    #[test]
    fn derived_output_matches_pipeline() {
        let (cs, z_next, _, constants) = synth(MASK, None);

        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        assert_eq!(
            z_next[1].get_value().unwrap(),
            replay(&constants, &MASK, &EXPECTED_OUT)
        );
    }

    // --- negative tests -----------------------------------------------------

    /// Wrong corrected value on the dead pixel: the select constraint must
    /// fail on exactly that pixel.
    #[test]
    fn wrong_correction_rejected() {
        let mut out = EXPECTED_OUT.to_vec();
        out[3] = 11; // truth is 10
        let (cs, _, _, _) = synth(MASK, Some(out));

        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("step 0/dpc select 3"));
    }

    /// A live pixel is NOT free to change: pass-through is enforced, not
    /// assumed.
    #[test]
    fn tampered_live_pixel_rejected() {
        let mut out = EXPECTED_OUT.to_vec();
        out[5] = 99; // live pixel, must stay 10
        let (cs, _, _, _) = synth(MASK, Some(out));

        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("step 0/dpc select 5"));
    }

    /// The mask may never flag a border pixel (python guarantees it; the
    /// circuit enforces it).
    #[test]
    fn dead_border_pixel_rejected() {
        let mut mask = MASK;
        mask[0] = true;
        let (cs, _, _, _) = synth(mask, Some(EXPECTED_OUT.to_vec()));

        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("step 0/border flag 0"));
    }

    /// THE core soundness test for the division. Field division is exact:
    /// corr' = 81 * 8^-1 with r' = 0 satisfies the Euclid constraint
    /// perfectly — the ONLY thing standing between the prover and that fake
    /// quotient is corr's 8-bit range check (the repack constraint). Swap
    /// both witnesses and verify the failure lands exactly there.
    #[test]
    fn field_inverse_quotient_rejected() {
        let (mut cs, _, _, _) = synth(MASK, Some(EXPECTED_OUT.to_vec()));
        assert!(cs.is_satisfied());

        let fake_quotient = F::from(81u64) * F::from(8u64).invert().unwrap();
        cs.set("step 0/corr/px 2/Alloc value/num", fake_quotient); // px 2 = j 3
        cs.set("step 0/r 3 bit 0/boolean", F::ZERO); // r: 1 -> 0

        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("step 0/corr/px 2 repack"));
    }

    /// Tampering with the remainder alone breaks the Euclid identity.
    #[test]
    fn tampered_remainder_rejected() {
        let (mut cs, _, _, _) = synth(MASK, Some(EXPECTED_OUT.to_vec()));
        assert!(cs.is_satisfied());

        cs.set("step 0/r 3 bit 0/boolean", F::ZERO); // 81 = 8*10 + 1, claim r = 0
        assert!(!cs.is_satisfied());
        assert_eq!(cs.which_is_unsatisfied(), Some("step 0/euclid 3"));
    }

    /// A remainder "bit" outside {0,1} dies on booleanity — that is the half
    /// of the r < 8 range check that stops r from absorbing arbitrary field
    /// values.
    #[test]
    fn non_boolean_remainder_rejected() {
        let (mut cs, _, _, _) = synth(MASK, Some(EXPECTED_OUT.to_vec()));
        assert!(cs.is_satisfied());

        cs.set("step 0/r 3 bit 0/boolean", F::from(2u64));
        assert!(!cs.is_satisfied());
        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("step 0/r 3 bit 0/boolean constraint")
        );
    }

    /// A forged mask (extra dead pixel) can satisfy the constraints — the
    /// circuit alone cannot know the true mask. What catches it is the
    /// digest: the final z no longer matches the verifier's replay over the
    /// REAL mask. This is exactly why the mask is absorbed.
    #[test]
    fn forged_mask_caught_by_digest() {
        let mut forged = MASK;
        forged[5] = true; // interior, so in-circuit constraints stay happy
        let (cs, z_next, _, constants) = synth(forged, None);

        assert!(cs.is_satisfied(), "forged mask should still satisfy R1CS");
        assert_ne!(
            z_next[1].get_value().unwrap(),
            replay(&constants, &MASK, &EXPECTED_OUT),
            "digest must expose the forged mask"
        );
    }

    // --- real pipeline data -------------------------------------------------
    //
    // Frames straight from the python pipeline: the defective capture
    // (01_defective) as circuit input, the DPC stage's own output (01_dpc) as
    // the claimed output row, and the sensor's defect table (LUT_DP). Row 2
    // holds a dead pixel (stuck black) at column 214.

    use crate::helpers::{load_frame, load_mask};

    /// (above, cur, below, mask row, pipeline out row, dead pixel column)
    fn load_pipeline_row() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<bool>, Vec<u8>, usize) {
        let f_in =
            load_frame("../../scripts/python/data/01_defective/gradient_horizontal.npy").unwrap();
        let f_out =
            load_frame("../../scripts/python/data/dpc_then_gain/01_dpc/gradient_horizontal.npy")
                .unwrap();
        let mask = load_mask("../../scripts/python/data/01_defective/LUT_DP.npy").unwrap();

        const R: usize = 2; // interior row with a defect (stride walk)
        let j = mask[R]
            .iter()
            .position(|f| *f)
            .expect("row 2 should hold a dead pixel");

        (
            f_in[R - 1].clone(),
            f_in[R].clone(),
            f_in[R + 1].clone(),
            mask[R].clone(),
            f_out[R].clone(),
            j,
        )
    }

    /// The pipeline's own output for a real defective row satisfies the
    /// circuit, and the digest matches the verifier's replay. Cross-checks
    /// the circuit against the python implementation end to end.
    #[test]
    fn real_frame_row_verifies() {
        let (above, cur, below, mask, out, _) = load_pipeline_row();
        let (cs, z_next, _, constants) =
            synth_rows(&above, &cur, &below, &mask, Some(out.clone()));

        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        assert_eq!(
            z_next[1].get_value().unwrap(),
            replay_rows(&constants, &above, &cur, &below, &mask, &out),
            "digest drift vs native replay"
        );
    }

    /// Same row with the prover deriving the output itself: the digest must
    /// land on the pipeline file's digest, i.e. circuit arithmetic ==
    /// python arithmetic on real data.
    #[test]
    fn real_frame_derived_output_matches_pipeline() {
        let (above, cur, below, mask, out, _) = load_pipeline_row();
        let (cs, z_next, _, constants) = synth_rows(&above, &cur, &below, &mask, None);

        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        assert_eq!(
            z_next[1].get_value().unwrap(),
            replay_rows(&constants, &above, &cur, &below, &mask, &out),
            "derived output diverges from the python pipeline"
        );
    }

    /// Tamper the real corrected value at the dead pixel: must fail on that
    /// pixel's select constraint and no other.
    #[test]
    fn real_frame_tampered_dead_pixel_rejected() {
        let (above, cur, below, mask, mut out, j) = load_pipeline_row();
        out[j] = out[j].wrapping_add(1);
        let (cs, _, _, _) = synth_rows(&above, &cur, &below, &mask, Some(out));

        assert!(!cs.is_satisfied());
        assert_eq!(
            cs.which_is_unsatisfied().as_deref(),
            Some(format!("step 0/dpc select {j}").as_str())
        );
    }

    /// Wrong stage as output: claim the *defective* row itself (skipping the
    /// correction). Every live pixel passes through fine — the one dead
    /// pixel betrays it.
    #[test]
    fn real_frame_uncorrected_output_rejected() {
        let (above, cur, below, mask, _, j) = load_pipeline_row();
        let (cs, _, _, _) = synth_rows(&above, &cur, &below, &mask, Some(cur.clone()));

        assert!(!cs.is_satisfied());
        assert_eq!(
            cs.which_is_unsatisfied().as_deref(),
            Some(format!("step 0/dpc select {j}").as_str())
        );
    }

    /// Forged mask on real data: hide the true defect (flag it off) and let
    /// the prover derive a consistent witness. R1CS stays satisfied — the
    /// stuck-black pixel simply passes through — but the digest no longer
    /// matches the verifier's replay over the sensor's real defect table.
    #[test]
    fn real_frame_forged_mask_caught_by_digest() {
        let (above, cur, below, mask, out, j) = load_pipeline_row();
        let mut forged = mask.clone();
        forged[j] = false; // pretend the dead pixel is fine
        let (cs, z_next, _, constants) = synth_rows(&above, &cur, &below, &forged, None);

        assert!(cs.is_satisfied(), "forged mask should still satisfy R1CS");
        assert_ne!(
            z_next[1].get_value().unwrap(),
            replay_rows(&constants, &above, &cur, &below, &mask, &out),
            "digest must expose the forged mask"
        );
    }
}
