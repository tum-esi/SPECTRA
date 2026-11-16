use crate::circuits::d_gain::*;
use crate::circuits::demosaic::*;
use crate::circuits::dpc::*;
use crate::circuits::white_balance::*;

use crate::prelude::*;

type Img = Vec<Vec<u8>>;
type Lut = Vec<Vec<bool>>;

#[derive(Clone)]
pub enum ProgramSteps {
    // The stages of the NIVC circuit, possibility to extend
    DPC(Vec<DPCCircuit>),
    Gain(Vec<DGainCircuit>),
    Demosaic(Vec<DemosaicCircuit>),
    WhiteBalance(Vec<WhiteBalancingCircuit>),
}

fn synthesize_batch<C, CS>(
    circuits: &[C],
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
    circuit_index: usize,
) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError>
where
    C: StepCircuit<F>,
    CS: ConstraintSystem<F>,
{
    if circuits.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut state = z.to_vec();
    let mut final_pc = None;

    for (i, circuit) in circuits.iter().enumerate() {
        let (pc_next, state_next) = circuit.synthesize(
            &mut cs.namespace(|| format!("logical step {i}")),
            if i == 0 {
                pc
            } else {
                None
            },
            &state,
        )?;

        let pc_next = pc_next.ok_or(SynthesisError::Unsatisfiable)?;

        if (i + 1) < circuits.len() {
            cs.enforce(
                || format!("logical step {i} stays in circuit"),
                |lc| lc,
                |lc| lc,
                |lc| lc + pc_next.get_variable() - (F::from(circuit_index as u64), CS::one()),
            );
        }
        state = state_next;
        final_pc = Some(pc_next);
    }

    Ok((final_pc, state))
}

impl StepCircuit<F> for ProgramSteps {
    fn arity(&self) -> usize {
        ARITY
    }

    fn circuit_index(&self) -> usize {
        match self {
            Self::DPC(_) => DPC_IDX,
            Self::Gain(_) => GAIN_IDX,
            Self::Demosaic(_) => DEMOSAIC_IDX,
            Self::WhiteBalance(_) => WBAL_IDX,
        }
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        pc: Option<&AllocatedNum<F>>,
        z: &[AllocatedNum<F>],
    ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
        match self {
            Self::DPC(circuits) => synthesize_batch(circuits, cs, pc, z, DPC_IDX),
            Self::Gain(circuits) => synthesize_batch(circuits, cs, pc, z, GAIN_IDX),
            Self::Demosaic(circuits) => synthesize_batch(circuits, cs, pc, z, DEMOSAIC_IDX),
            Self::WhiteBalance(circuits) => synthesize_batch(circuits, cs, pc, z, WBAL_IDX),
        }
    }
}

pub struct SPECTRA {
    pub constants: Arc<H2<F>>,

    pub gain: u8,
    pub mask: Vec<Vec<bool>>, // LUT for the row
    pub width: usize,
    pub height:usize,
    // Batch size = number of rows per fold for DPC/Gain, or two-rwo bands per
    // fold for Demosaic. A frame must contain a whole number of batches.
    pub batch_size: usize,
    // Stage 1 input - Out of stage 0
    pub defective: Vec<Vec<u8>>,
    // Stage 2 input - Stage 1 output
    pub corrected: Vec<Vec<u8>>,
    //Stage 3
    pub gained: Vec<Vec<u8>>,
    // Stage 4: Demosaicing
    pub rgb_r: Vec<Vec<u8>>,
    pub rgb_g: Vec<Vec<u8>>,
    pub rgb_b: Vec<Vec<u8>>,
    // Stage 5: white balance. G passes through, so only R and B are outputs.
    pub r_gain: u8,
    pub b_gain: u8,
    pub wbal_r: Vec<Vec<u8>>,
    pub wbal_b: Vec<Vec<u8>>,
}



impl NonUniformCircuit<E1, E2, ProgramSteps, Secondary> for SPECTRA {
    fn num_circuits(&self) -> usize { INDECES.len() }
    fn primary_circuit(&self, circuit_index: usize) -> ProgramSteps {
        match circuit_index {
           0=>self.dpc_step(0),
           1=>self.gain_step(0),
           2=>self.demosaic_step(0),
           3=>self.wbal_step(0),
           _ => panic!("pipeline has {} circuits, asked for index {circuit_index}", INDECES.len()),
        }
    }
    fn secondary_circuit(&self) -> Secondary { Default::default() }
}

impl SPECTRA {
    
    #[allow(clippy::too_many_arguments)]
    pub fn new(gain:u8, lut: Lut, width:usize , height:usize, defective:Img, corrected:Img,gained:Img, rgb_r:Img, rgb_g:Img, rgb_b:Img, r_gain:u8, b_gain:u8, wbal_r:Img, wbal_b:Img ) ->SPECTRA{
        SPECTRA{
            constants: Arc::new(H2::<F>::new()),
            gain,
            mask: lut,
            width,
            height,
            batch_size: 1,
            defective,
            corrected,
            gained,
            rgb_r,
            rgb_g,
            rgb_b,
            r_gain,
            b_gain,
            wbal_r,
            wbal_b,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self, String> {
        if batch_size == 0 {
            return Err("batch size must be at least 1".into());
        }

        if self.height() % batch_size != 0 || (self.height / 2) % batch_size != 0 {
            return Err(format!(
                "height {} must be divisible by 2 * batch_size {}",
                self.height(),
                batch_size
            ));
        }

        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn width(&self) -> usize {
        self.defective[0].len()
    }

    
    pub fn height(&self) -> usize {
        self.defective.len()
    }

    /// Proving steps per stage: DPC and gain eat `batch_size` rows per step,
    /// demosaic `batch_size` 2-row bands.
    pub fn dpc_steps(&self) -> usize {
        self.height() / self.batch_size
    }
    pub fn gain_steps(&self) -> usize {
        self.height() / self.batch_size
    }
    pub fn demosaic_steps(&self) -> usize {
        (self.height() / 2) / self.batch_size
    }
    pub fn wbal_steps(&self) -> usize {
        self.height() / self.batch_size
    }

    /// Total steps of the whole pipeline: h/B + h/B + h/2B + h/B, where B is batch size.
    pub fn num_steps(&self) -> usize {
        self.dpc_steps() + self.gain_steps() + self.demosaic_steps() + self.wbal_steps()
    }

    /// The circuit for global step `k`, following the pc schedule the
    /// circuits themselves enforce: DPC over every row, then gain over every
    /// row, then demosaic over every band, then white balance over every
    /// row. The prover must hand
    /// `prove_step` circuits in exactly this order — a mismatch shows up as
    /// `circuit_index != pc` and an unsatisfiable step.
    pub fn step(&self, k: usize) -> ProgramSteps {
        let dpc_steps = self.dpc_steps();
        let gain_steps = self.gain_steps();
        let demosaic_steps = self.demosaic_steps();

        // Convert the global step counter `k` into a stage-local counter,
        // then call the appropriate stage with this local counter.
        if k < dpc_steps {
            self.dpc_step(k)
        } else if k < (dpc_steps + gain_steps) {
            self.gain_step(k - dpc_steps)
        } else if k < (dpc_steps + gain_steps + demosaic_steps) {
            self.demosaic_step(k - dpc_steps - gain_steps)
        } else {
            self.wbal_step(k - dpc_steps - gain_steps - demosaic_steps)
        }
    }

    /// DPC over batch `t` of the defective frame.
    ///
    /// Bayer neighbourhood (pipeline.py runs `dpc(..., isBayer=True)`): the
    /// eight same-colour neighbours sit two rows and two columns away, so
    /// `above` is row t-2 and `below` row t+2.
    ///
    /// The two rows at each border have no such neighbour and use the clamp
    /// convention: the row itself. The mask never flags a pixel within two
    /// of the border (generateDefectMap guarantees it, deadpixel.py asserts
    /// it), so those values can never reach a correction — they only have
    /// to be *some* fixed choice, because they are absorbed into H_IN_DPC
    /// and the verifier's native replay must pick the same rows.
    pub fn dpc_step(&self, t: usize) -> ProgramSteps {
        let first = t * self.batch_size;

        // Based on the stage-local counter `t`, we find the section of the
        // image frame we need to process now and for every single row in that
        // batch, we create each row-circuit and collect them as one step/fold.
        ProgramSteps::DPC(
            (first..(first + self.batch_size))
                .map(|row| self.dpc_row(row))
                .collect(),
        )
    }

    /// Private function dpc_row builds a circuit for a single row while
    /// the public-facing dpc_step collects the circuits from all rows within
    /// a batch into the ProgramSteps that then get returned for the proof.
    fn dpc_row(&self, t: usize) -> DPCCircuit {
        let h = self.height();
        let above = if t < 2 { t } else { t - 2 };
        let below = if t + 2 >= h { t } else { t + 2 };

        DPCCircuit {
            in_advice: self.defective[t].clone(),
            above_in_advice: self.defective[above].clone(),
            below_row_in_advice: self.defective[below].clone(),
            out_advice: self.corrected[t].clone(),
            lut: self.mask[t].clone(),
            width: self.width(),
            height: h,
            // The index arecibo assigned this circuit. `circuit_index()` is
            // constrained in-circuit against pc, so it is identity, not decor.
            circuit_index: DPC_IDX,
            constants: self.constants.clone(),
        }
    }

    /// Gain over batch `t`: input is DPC's output frame.
    pub fn gain_step(&self, t: usize) -> ProgramSteps {
        let first = t * self.batch_size;

        ProgramSteps::Gain(
            (first..(first + self.batch_size))
                 .map(|row| self.gain_row(row))
                 .collect(),
        )
    }

    fn gain_row(&self, t: usize) -> DGainCircuit {
        DGainCircuit {
            in_advice: self.corrected[t].clone(),
            out_advice: self.gained[t].clone(),
            gain: self.gain,
            width: self.width(),
            height: self.height(),
            circuit_index: GAIN_IDX,
            constants: self.constants.clone(),
        }
    }

    /// Demosaic over batch `t`, containing `batch_size` two-row bands,
    /// where each band is rows 2t and 2t+1 of the gained frame.
    /// The output channels are the matching two rows of each rgb plane,
    /// concatenated row-major — the layout DemosaicCircuit documents.
    pub fn demosaic_step(&self, t: usize) -> ProgramSteps {
        let first = t * self.batch_size;

        ProgramSteps::Demosaic(
            (first..(first + self.batch_size))
                .map(|band| self.demosaic_band(band))
                .collect(),
        )
    }

    fn demosaic_band(&self, t: usize) -> DemosaicCircuit {
        let (r1, r2) = (2 * t, 2 * t + 1);
        DemosaicCircuit {
            r1_advice: self.gained[r1].clone(),
            r2_advice: self.gained[r2].clone(),
            width: self.width(),
            height: self.height(),
            circuit_index: DEMOSAIC_IDX,
            constants: self.constants.clone(),
        }
    }

    /// White balance over batch `t`: input is demosaic's rgb planes, row by row.
    pub fn wbal_step(&self, t: usize) -> ProgramSteps {
        let first = t * self.batch_size;

        ProgramSteps::WhiteBalance(
            (first..(first + self.batch_size))
                .map(|row| self.wbal_row(row))
                .collect(),
        )
    }

    fn wbal_row(&self, t: usize) -> WhiteBalancingCircuit {
        WhiteBalancingCircuit {
            in_r: self.rgb_r[t].clone(),
            in_g: self.rgb_g[t].clone(),
            in_b: self.rgb_b[t].clone(),
            out_r: self.wbal_r[t].clone(),
            out_b: self.wbal_b[t].clone(),
            width: self.width(),
            height: self.height(),
            circuit_index: WBAL_IDX,
            constants: self.constants.clone(),
        }
    }
}

#[cfg(test)]
mod tmp_measure {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;
    #[test]
    fn tmp_constraints_1920() {
        let (w, h) = (1920, 4);
        let z = || vec![vec![0u8; w]; h];
        let p = SPECTRA::new(2, vec![vec![false; w]; h], w, h, z(), z(), z(), z(), z(), z(), 2, 4, z(), z());
        for i in 0..4 {
            let mut cs = TestConstraintSystem::<F>::new();
            let zs: Vec<_> = (0..ARITY).map(|k| AllocatedNum::alloc_input(cs.namespace(|| format!("z{k}")), || Ok(F::ZERO)).unwrap()).collect();
            let pc = AllocatedNum::alloc_input(cs.namespace(|| "pc"), || Ok(F::from(i as u64))).unwrap();
            p.primary_circuit(i).synthesize(&mut cs.namespace(|| "F"), Some(&pc), &zs).unwrap();
            println!("stage {i}: {} constraints", cs.num_constraints());
        }
    }
}
