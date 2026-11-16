//! Negative tests for provenance under real folding.
//!
//! The question: is pc binding enough for the proof to attest *which*
//! pipeline produced the output, or do the circuits need more constraints?
//! Every test is one adversarial prover deviating from the honest
//! DPC -> gain -> demosaic -> wbal run over the same frame, folded with arecibo for
//! real, and records exactly what stops it — and what does not.
//!
//!   adversary                                 stopped by                         where
//!   circuit lies about its index              pp shape that pc selects           prove_step: InvalidWitnessLength
//!   wrong circuit through the honest API      prover-side assert                 prove_step: panic
//!   gain claims height 1, ends stage early    `total` is a constant in the shape verify: UnSatIndex
//!   starts at gain, never runs DPC            pc in the state (z[Z_PC])          no accepted proof
//!   stops one row early                       nothing in the proof               verify OK; z[Z_PC] != DONE_IDX
//!   gain rows fed in a different order        link check on gain's last step     no accepted proof
//!   gain fed the defective frame              link check on gain's last step     no accepted proof
//!
//! Reading: pc plus the pp shapes bind the *schedule* — which circuit's
//! constraints every step satisfied, and that a stage can only hand over on
//! its last step. The *data* is bound by the digest slots of z: every stage
//! records the digest of what it consumed and of what it produced, and on
//! its last step requires its input digest to equal its predecessor's output
//! digest (`enforce_equal_on_last`). The verifier is left with the two ends
//! of that chain and the pc having reached DONE_IDX (evidence.rs). A
//! truncated run is the one valid proof in the table; that check rejects it.
//!
//! One arecibo fact drives the "starts at gain" row: in the base case the
//! augmented circuit allocates pc as a free witness and only enforces
//! `pc == circuit_index` of whatever circuit runs first
//! (supernova/circuit.rs, `enforcing_synthesize`); nothing pins the initial
//! pc to a public value, and `verify` never takes one.
//!
//! Honest frames come from the same native spec test2.rs used, so the
//! circuits are checked against the python semantics, not against
//! themselves. `assert_eq!` against the honest run's z stands in for the
//! verifier's replay: the per-stage tests already establish that the
//! circuit digests equal the native ones.

#[cfg(test)]
mod tests {
    use crate::circuits::spectra::{ProgramSteps, SPECTRA};
    use crate::circuits::d_gain::DGainCircuit;
    use crate::evidence::{check_evidence, initial_state, readout_digest, Spec};
    use crate::prelude::*;
    use arecibo::errors::NovaError;
    use arecibo::supernova::error::SuperNovaError;

    /// Frame geometry: h + h + h/2 + h = 21 proving steps. W even (whole
    /// superpixels), H even (whole bands), both > 4 so a defect fits
    /// inside the 2-pixel Bayer margin the DPC neighbourhood needs.
    const H: usize = 6;
    const W: usize = 8;
    const GAIN: u8 = 2;
    const R_GAIN: u8 = 2;
    const B_GAIN: u8 = 4;

    // z slots, see prelude.rs.
    const LUT_D: usize = 1;
    const H_IN_GAIN: usize = 2;
    const H_OUT_GAIN: usize = 3;
    const H_IN_DPC: usize = 4;
    const H_OUT_DPC: usize = 5;
    const ROW: usize = 6;
    const H_IN_DEMOSAIC: usize = 7;
    const H_OUT_DEMOSAIC: usize = 8;
    const H_IN_WBAL: usize = 9;
    const H_OUT_WBAL: usize = 10;

    // ------------------------------------------------------------------
    // Native spec: what the honest frames look like (mirrors
    // scripts/python/stages/{deadpixel,gain,demosaic,wbal}.py).
    // ------------------------------------------------------------------

    /// Deterministic input frame spanning the gain saturation boundary,
    /// every row distinct, one stuck-white pixel at (2, 3) for DPC to fix.
    fn defective_frame() -> Vec<Vec<u8>> {
        let mut f: Vec<Vec<u8>> = (0..H)
            .map(|y| (0..W).map(|x| ((y * 30 + x * 20) % 200 + 40) as u8).collect())
            .collect();
        f[2][3] = 250;
        f
    }

    /// Defect map: only (2, 3). Two away from every border, as the LUT
    /// generator guarantees for a Bayer frame.
    fn mask() -> Vec<Vec<bool>> {
        let mut m = vec![vec![false; W]; H];
        m[2][3] = true;
        m
    }

    /// deadpixel.py with `isBayer=True`: a flagged pixel becomes the
    /// floor-average of its 8 same-colour neighbours (offsets ±2) in the
    /// *defective* frame.
    fn dpc_spec(defective: &[Vec<u8>], mask: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut out = defective.to_vec();
        for y in 0..H {
            for x in 0..W {
                if mask[y][x] != 0 {
                    let mut acc: u16 = 0;
                    for dy in [-2i32, 0, 2] {
                        for dx in [-2i32, 0, 2] {
                            if (dy, dx) != (0, 0) {
                                acc += defective[(y as i32 + dy) as usize]
                                    [(x as i32 + dx) as usize]
                                    as u16;
                            }
                        }
                    }
                    out[y][x] = (acc / 8) as u8;
                }
            }
        }
        out
    }

    /// Saturating digital gain: out = min(px * g, 255).
    fn gain_spec(frame: &[Vec<u8>], g: u8) -> Vec<Vec<u8>> {
        frame
            .iter()
            .map(|row| row.iter().map(|px| ((*px as u16) * (g as u16)).min(255) as u8).collect())
            .collect()
    }

    /// demosaic.py NearestNeighbour: each 2x2 superpixel [R G1; G2 B]
    /// expands to two RGB rows sharing R and B, row 1 keeping G1, row 2
    /// keeping G2. Three full-height channel planes.
    #[allow(clippy::type_complexity)]
    fn demosaic_spec(bayer: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut r = vec![vec![0u8; W]; H];
        let mut g = vec![vec![0u8; W]; H];
        let mut b = vec![vec![0u8; W]; H];
        for t in 0..H / 2 {
            let (y1, y2) = (2 * t, 2 * t + 1);
            for k in (0..W).step_by(2) {
                let (rv, g1, g2, bv) =
                    (bayer[y1][k], bayer[y1][k + 1], bayer[y2][k], bayer[y2][k + 1]);
                for x in [k, k + 1] {
                    r[y1][x] = rv;
                    r[y2][x] = rv;
                    g[y1][x] = g1;
                    g[y2][x] = g2;
                    b[y1][x] = bv;
                    b[y2][x] = bv;
                }
            }
        }
        (r, g, b)
    }

    /// The program whose gain stage reads `corrected`, everything after
    /// it chained from the spec. wbal.py is the gain spec per plane: R and
    /// B scaled, G passed through.
    fn chained(defective: Vec<Vec<u8>>, corrected: Vec<Vec<u8>>) -> SPECTRA {
        let gained = gain_spec(&corrected, GAIN);
        let (r, g, b) = demosaic_spec(&gained);
        let (wb_r, wb_b) = (gain_spec(&r, R_GAIN), gain_spec(&b, B_GAIN));
        SPECTRA::new(
            GAIN, mask(), W, H, defective, corrected, gained, r, g, b, R_GAIN, B_GAIN, wb_r, wb_b,
        )
    }

    /// The honest program: every frame chained from the spec.
    fn honest() -> SPECTRA {
        let defective = defective_frame();
        let mask_u8: Vec<Vec<u8>> =
            mask().iter().map(|r| r.iter().map(|&b| b as u8).collect()).collect();
        let corrected = dpc_spec(&defective, &mask_u8);
        chained(defective, corrected)
    }

    // ------------------------------------------------------------------
    // The prover's view: circuits carry the index they *claim*, and the
    // program decides where pc starts. Honest values reproduce SPECTRA.
    // ------------------------------------------------------------------

    /// A step circuit with a claimed index. `circuit_index()` is a plain
    /// Rust method, so a dishonest prover can return whatever pc wants to
    /// hear — that is exactly what the impostor test does.
    #[derive(Clone)]
    struct Claimed {
        inner: ProgramSteps,
        index: usize,
    }

    impl Claimed {
        fn honest(inner: ProgramSteps) -> Self {
            let index = inner.circuit_index();
            Self { inner, index }
        }
    }

    impl StepCircuit<F> for Claimed {
        fn arity(&self) -> usize {
            self.inner.arity()
        }
        fn circuit_index(&self) -> usize {
            self.index
        }
        fn synthesize<CS: ConstraintSystem<F>>(
            &self,
            cs: &mut CS,
            pc: Option<&AllocatedNum<F>>,
            z: &[AllocatedNum<F>],
        ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
            self.inner.synthesize(cs, pc, z)
        }
    }

    struct Program {
        base: SPECTRA,
        /// Where the prover says pc starts. arecibo takes this on trust.
        start: usize,
    }

    impl NonUniformCircuit<E1, E2, Claimed, Secondary> for Program {
        fn num_circuits(&self) -> usize {
            INDECES.len()
        }
        fn primary_circuit(&self, circuit_index: usize) -> Claimed {
            Claimed::honest(self.base.primary_circuit(circuit_index))
        }
        fn secondary_circuit(&self) -> Secondary {
            Default::default()
        }
        fn initial_circuit_index(&self) -> usize {
            self.start
        }
    }

    impl Program {
        fn honest() -> Self {
            Self { base: honest(), start: DPC_IDX }
        }
        fn dpc(&self, t: usize) -> Claimed {
            Claimed::honest(self.base.dpc_step(t))
        }
        fn gain(&self, t: usize) -> Claimed {
            Claimed::honest(self.base.gain_step(t))
        }
        fn demosaic(&self, t: usize) -> Claimed {
            Claimed::honest(self.base.demosaic_step(t))
        }
        fn wbal(&self, t: usize) -> Claimed {
            Claimed::honest(self.base.wbal_step(t))
        }
        /// The honest schedule: DPC over every row, gain over every row,
        /// demosaic over every band, white balance over every row.
        fn schedule(&self) -> Vec<Claimed> {
            (0..H)
                .map(|t| self.dpc(t))
                .chain((0..H).map(|t| self.gain(t)))
                .chain((0..H / 2).map(|t| self.demosaic(t)))
                .chain((0..H).map(|t| self.wbal(t)))
                .collect()
        }
    }

    fn setup(program: &Program) -> PublicParams<E1, E2, Claimed, Secondary> {
        PublicParams::setup(program, &*default_ck_hint(), &*default_ck_hint())
    }

    fn spec_mask() -> Vec<Vec<bool>> {
        mask()
    }

    /// z_0 as the honest prover sets it: parameters, mask digest, d_1.
    fn z0() -> Vec<F> {
        let constants = H2::<F>::new();
        let mask = spec_mask();
        let spec = Spec { gain: GAIN, r_gain: R_GAIN, b_gain: B_GAIN, mask: &mask };
        initial_state(&constants, &spec, readout_digest(&constants, &defective_frame()))
    }

    /// Thread z through `steps` in a test constraint system (no folding).
    /// Returns the final state, or the index of the first unsatisfied step.
    fn dry_run(steps: &[ProgramSteps], z0: &[F]) -> Result<Vec<F>, usize> {
        use bellpepper_core::test_cs::TestConstraintSystem;
        let mut z = z0.to_vec();
        for (k, step) in steps.iter().enumerate() {
            let mut cs = TestConstraintSystem::<F>::new();
            let zs: Vec<_> = z
                .iter()
                .enumerate()
                .map(|(i, v)| AllocatedNum::alloc_input(cs.namespace(|| format!("z{i}")), || Ok(*v)).unwrap())
                .collect();
            let (_, z_next) = step.synthesize(&mut cs.namespace(|| "F"), None, &zs).unwrap();
            if !cs.is_satisfied() {
                return Err(k);
            }
            z = z_next.iter().map(|v| v.get_value().unwrap()).collect();
        }
        Ok(z)
    }

    fn schedule_of(program: &SPECTRA) -> Vec<ProgramSteps> {
        (0..program.num_steps()).map(|k| program.step(k)).collect()
    }

    /// The native digests and checks of evidence.rs agree with what the
    /// circuits write, and each check rejects the deviation it is there for.
    #[test]
    fn evidence_matches_the_circuits() {
        let constants = H2::<F>::new();
        let program = honest();
        let mask = spec_mask();
        let spec = Spec { gain: GAIN, r_gain: R_GAIN, b_gain: B_GAIN, mask: &mask };
        let y = (&program.wbal_r[..], &program.rgb_g[..], &program.wbal_b[..]);
        let zn = dry_run(&schedule_of(&program), &z0()).expect("honest run is satisfiable");
        assert_eq!(check_evidence(&constants, &z0(), &zn, &spec, y), Ok(()));

        // z_0 with a parameter the verifier did not specify.
        let other = Spec { gain: GAIN, r_gain: R_GAIN + 1, b_gain: B_GAIN, mask: &mask };
        assert!(check_evidence(&constants, &z0(), &zn, &other, y).is_err());

        // A delivered output the pipeline did not produce.
        let mut forged = program.wbal_r.clone();
        forged[0][0] ^= 1;
        assert!(check_evidence(&constants, &z0(), &zn, &spec, (&forged[..], y.1, y.2)).is_err());

        // A readout digest other than the one the first stage consumed.
        let mut z0_bad = z0();
        z0_bad[Z_D1] += F::ONE;
        assert!(check_evidence(&constants, &z0_bad, &zn, &spec, y).is_err());
    }

    /// DPC run with a mask other than the specified one fails in-circuit on
    /// its last row, where LUT_D is compared with the digest in z_0.
    #[test]
    fn mask_other_than_the_specified_one_is_unsatisfiable() {
        let mut program = honest();
        program.mask[0][0] = true; // border pixel: DPC passes it through, frames stay honest
        assert_eq!(dry_run(&schedule_of(&program), &z0()), Err(program.dpc_steps() - 1));
    }

    /// A stage fed anything but its predecessor's output is unsatisfiable on
    /// its last step, where its input digest is compared with the output
    /// digest the predecessor recorded. One fork per link; each fork is
    /// consistent downstream of the frame it replaces, so nothing else fails.
    #[test]
    fn stage_not_fed_its_predecessors_output_is_unsatisfiable() {
        let a = honest();
        let flip = |frame: &[Vec<u8>]| {
            let mut f = frame.to_vec();
            f[0][0] ^= 1;
            f
        };
        let fork_gain = chained(a.defective.clone(), flip(&a.corrected));
        let fork_demosaic = {
            let gained = flip(&a.gained);
            let (r, g, b) = demosaic_spec(&gained);
            let (wb_r, wb_b) = (gain_spec(&r, R_GAIN), gain_spec(&b, B_GAIN));
            SPECTRA::new(
                GAIN, mask(), W, H, a.defective.clone(), a.corrected.clone(), gained, r, g, b,
                R_GAIN, B_GAIN, wb_r, wb_b,
            )
        };
        let fork_wbal = {
            let r = flip(&a.rgb_r);
            let wb_r = gain_spec(&r, R_GAIN);
            SPECTRA::new(
                GAIN, mask(), W, H, a.defective.clone(), a.corrected.clone(), a.gained.clone(), r,
                a.rgb_g.clone(), a.rgb_b.clone(), R_GAIN, B_GAIN, wb_r, a.wbal_b.clone(),
            )
        };

        // First step of gain, demosaic, white balance; the stage's last step
        // is one before the next stage's first.
        let gain = a.dpc_steps();
        let demosaic = gain + a.gain_steps();
        let wbal = demosaic + a.demosaic_steps();
        for (fork, first, last) in [
            (&fork_gain, gain, demosaic - 1),
            (&fork_demosaic, demosaic, wbal - 1),
            (&fork_wbal, wbal, a.num_steps() - 1),
        ] {
            let steps: Vec<ProgramSteps> = (0..a.num_steps())
                .map(|k| if k < first { a.step(k) } else { fork.step(k) })
                .collect();
            assert_eq!(dry_run(&steps, &z0()), Err(last));
        }
    }

    fn z0_s() -> Vec<<E2 as Engine>::Scalar> {
        vec![<E2 as Engine>::Scalar::ONE]
    }

    /// Fold `steps` in order under `pp`; on failure, say at which step.
    fn fold(
        pp: &PublicParams<E1, E2, Claimed, Secondary>,
        program: &Program,
        steps: &[Claimed],
    ) -> Result<RecursiveSNARK<E1, E2>, (usize, SuperNovaError)> {
        let secondary = program.secondary_circuit();
        let (z0_p, z0_s) = (z0(), z0_s());
        let mut rs = RecursiveSNARK::new(pp, program, &steps[0], &secondary, &z0_p, &z0_s)
            .map_err(|e| (0, e))?;
        for (k, step) in steps.iter().enumerate() {
            rs.prove_step(pp, step, &secondary).map_err(|e| (k, e))?;
        }
        Ok(rs)
    }

    fn fold_ok(
        pp: &PublicParams<E1, E2, Claimed, Secondary>,
        program: &Program,
        steps: &[Claimed],
    ) -> RecursiveSNARK<E1, E2> {
        fold(pp, program, steps).unwrap_or_else(|(k, e)| panic!("prove_step {k}: {e:?}"))
    }

    fn verify(
        pp: &PublicParams<E1, E2, Claimed, Secondary>,
        rs: &RecursiveSNARK<E1, E2>,
    ) -> Result<Vec<F>, SuperNovaError> {
        rs.verify(pp, &z0(), &z0_s()).map(|(zi, _)| zi)
    }

    /// Whether an adversarial run fails to yield an accepted proof. Folding
    /// does not check satisfiability, so the failure may only surface at
    /// verification.
    fn rejected(
        pp: &PublicParams<E1, E2, Claimed, Secondary>,
        program: &Program,
        steps: &[Claimed],
    ) -> bool {
        match fold(pp, program, steps) {
            Err(_) => true,
            Ok(rs) => verify(pp, &rs).is_err(),
        }
    }

    /// The honest run's final state: what the verifier's replay yields.
    fn honest_zi(pp: &PublicParams<E1, E2, Claimed, Secondary>, program: &Program) -> Vec<F> {
        let rs = fold_ok(pp, program, &program.schedule());
        verify(pp, &rs).expect("honest run verifies")
    }

    /// Slow in debug: every test runs `setup` and folds ten real steps.
    macro_rules! fold_test {
        ($(#[$m:meta])* fn $name:ident() $body:block) => {
            $(#[$m])*
            #[cfg_attr(
                debug_assertions,
                ignore = "real setup + folding, slow in debug: run with `cargo test --release`"
            )]
            #[test]
            fn $name() $body
        };
    }

    // ------------------------------------------------------------------
    // Baseline
    // ------------------------------------------------------------------

    fold_test! {
        /// The honest schedule verifies, and the verifier's slot checks all
        /// hold: this is the bar every adversary below is measured against.
        fn honest_schedule_verifies_and_slots_agree() {
            let program = Program::honest();
            let pp = setup(&program);
            let zi = honest_zi(&pp, &program);

            assert_eq!(zi[0], F::from(GAIN as u64), "gain drifted");
            assert_eq!(zi[ROW], F::ZERO, "ROW did not wrap");
            assert_eq!(zi[H_IN_GAIN], zi[H_OUT_DPC], "DPC out != gain in");
            assert_eq!(zi[H_IN_DEMOSAIC], zi[H_OUT_GAIN], "gain out != demosaic in");
            assert_eq!(zi[H_IN_WBAL], zi[H_OUT_DEMOSAIC], "demosaic out != wbal in");
            for (slot, name) in [
                (LUT_D, "LUT_D"),
                (H_IN_DPC, "H_IN_DPC"),
                (H_OUT_DEMOSAIC, "H_OUT_DEMOSAIC"),
                (H_IN_WBAL, "H_IN_WBAL"),
                (H_OUT_WBAL, "H_OUT_WBAL"),
            ] {
                assert_ne!(zi[slot], F::ZERO, "{name} never absorbed anything");
            }
        }
    }

    // ------------------------------------------------------------------
    // Schedule: what pc + the pp shapes stop on their own
    // ------------------------------------------------------------------

    fold_test! {
        /// A circuit that lies about its index. `circuit_index()` is not a
        /// constraint, so the prover can say "gain" while synthesising DPC
        /// demosaic or wbal; pc == index then holds in-circuit. What it buys
        /// is nothing: pc selects the *pp shape* the witness must satisfy,
        /// and no other stage's witness fits the gain shape.
        fn impostor_index_is_stopped_by_the_shape_pc_selects() {
            let program = Program::honest();
            let pp = setup(&program);

            for (what, inner) in [
                ("DPC", program.base.dpc_step(0)),
                ("demosaic", program.base.demosaic_step(0)),
                ("wbal", program.base.wbal_step(0)),
            ] {
                let mut steps = program.schedule();
                steps[H] = Claimed { inner, index: GAIN_IDX };
                let err = fold(&pp, &program, &steps)
                    .err()
                    .unwrap_or_else(|| panic!("{what} passed off as gain must not fold"));
                assert_eq!(
                    err,
                    (H, SuperNovaError::NovaError(NovaError::InvalidWitnessLength)),
                    "{what} as gain: expected the gain shape to reject the witness at step H"
                );
            }
        }
    }

    fold_test! {
        /// The honest API's own guard: hand `prove_step` a circuit whose
        /// index is not the current pc and arecibo asserts before touching
        /// the constraint system. Prover-side only, so on its own it is
        /// not soundness — the impostor test above is.
        fn honest_api_refuses_a_circuit_the_pc_did_not_select() {
            let program = Program::honest();
            let pp = setup(&program);

            // DPC done, pc = gain; try to jump straight to demosaic.
            let mut steps: Vec<Claimed> = (0..H).map(|t| program.dpc(t)).collect();
            steps.push(program.demosaic(0));
            let outcome = std::thread::scope(|s| {
                s.spawn(|| fold(&pp, &program, &steps).map(|_| ())).join()
            });
            assert!(outcome.is_err(), "expected arecibo's pc == circuit_index assert to fire");
        }
    }

    fold_test! {
        /// Ending a stage early by lying about its length. `height` is
        /// only a Rust field, but `advance_stage` bakes it into the
        /// matrices as the constant `total`, so a witness computed with
        /// `total = 1` folds fine and is unsatisfiable under the pp shape.
        /// Everything up to the last `verify` succeeds.
        fn stage_length_is_a_shape_constant() {
            let program = Program::honest();
            let pp = setup(&program);

            let ProgramSteps::Gain(mut gain0) = program.base.gain_step(0) else { unreachable!() };
            let gain0 = gain0.pop().unwrap();
            let early = Claimed::honest(ProgramSteps::Gain(vec![DGainCircuit { height: 1, ..gain0 }]));

            let mut steps: Vec<Claimed> = (0..H).map(|t| program.dpc(t)).collect();
            steps.push(early); // pc flips to demosaic after one row
            steps.extend((0..H / 2).map(|t| program.demosaic(t)));

            let rs = fold_ok(&pp, &program, &steps);
            let err = verify(&pp, &rs).err().expect("a lied-about height must not verify");
            assert!(
                matches!(err, SuperNovaError::UnSatIndex("r_primary", _)),
                "expected the gain accumulator to be unsatisfiable, got {err:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Data: what the digests and the pc carried in the state stop
    // ------------------------------------------------------------------

    fold_test! {
        /// Skip DPC entirely by starting at gain. arecibo leaves the
        /// base-case pc free, so on its own it would accept "gain,
        /// demosaic, wbal". The pc carried in the state does not: z_0 says
        /// DPC, and the gain circuit's first constraint is that it is its
        /// turn. No accepted proof exists.
        fn starting_at_gain_is_rejected_by_the_pc_in_the_state() {
            let program = Program::honest();
            let pp = setup(&program);

            let skipping = Program { base: honest(), start: GAIN_IDX };
            let steps: Vec<Claimed> = (0..H)
                .map(|t| skipping.gain(t))
                .chain((0..H / 2).map(|t| skipping.demosaic(t)))
                .chain((0..H).map(|t| skipping.wbal(t)))
                .collect();
            assert!(rejected(&pp, &skipping, &steps), "a run that never entered DPC must not verify");
        }
    }

    fold_test! {
        /// Stop before the last white-balance row. A recursive SNARK is
        /// valid after any number of steps; the pc in the final state still
        /// names the last stage instead of DONE_IDX, and the output digest
        /// is one row short.
        fn truncated_schedule_is_a_valid_proof_that_never_reaches_done() {
            let program = Program::honest();
            let pp = setup(&program);
            let expected = honest_zi(&pp, &program);
            assert_eq!(expected[Z_PC], F::from(DONE_IDX as u64), "honest run ends at DONE");

            let mut steps = program.schedule();
            steps.pop();
            let rs = fold_ok(&pp, &program, &steps);
            let zi = verify(&pp, &rs).expect("arecibo accepts a run that stopped early");

            assert_eq!(zi[Z_PC], F::from(WBAL_IDX as u64), "still in the last stage");
            assert_ne!(zi[H_OUT_WBAL], expected[H_OUT_WBAL]);
        }
    }

    fold_test! {
        /// Gain over the right rows in the wrong order. Every step is
        /// honest for its own row and the schedule looks complete, but the
        /// sequential input digest no longer equals DPC's output digest, and
        /// gain's last step requires that equality. No accepted proof.
        fn reordered_gain_rows_are_rejected_by_the_link_check() {
            let program = Program::honest();
            let pp = setup(&program);

            let mut steps = program.schedule();
            steps.swap(H, H + 1); // gain rows 0 and 1
            assert!(rejected(&pp, &program, &steps), "reordered rows must not verify");
        }
    }

    fold_test! {
        /// Gain applied to the defective frame, i.e. DPC's output silently
        /// replaced by its input. DPC is proven honestly, and gain, demosaic
        /// and wbal are consistent with each other. Gain's input digest is
        /// not DPC's output digest, so its last step is unsatisfiable. No
        /// accepted proof.
        fn gain_over_the_defective_frame_is_rejected_by_the_link_check() {
            let program = Program::honest();
            let pp = setup(&program);

            // The adversary's downstream: defective -> gain -> demosaic -> wbal.
            let defective = defective_frame();
            let forked = Program { base: chained(defective.clone(), defective), start: DPC_IDX };
            let steps: Vec<Claimed> = (0..H)
                .map(|t| program.dpc(t)) // honest DPC, real correction
                .chain((0..H).map(|t| forked.gain(t)))
                .chain((0..H / 2).map(|t| forked.demosaic(t)))
                .chain((0..H).map(|t| forked.wbal(t)))
                .collect();
            assert!(rejected(&pp, &program, &steps), "a stage fed another frame must not verify");
        }
    }
}
