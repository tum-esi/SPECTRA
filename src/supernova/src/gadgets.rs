use crate::prelude::*;
use bellpepper_core::LinearCombination;

pub type H2<F> = PoseidonConstants<F, U2>; // for the poseidon hash

/// Pack 8-bit pixels into as few field elements as possible.
///
/// Each chunk of `CAPACITY / 8` pixels becomes one element
/// `packed = sum 256^i * p_i`.
///
/// # Caller obligation
///
/// **Every pixel must already be constrained to 8 bits.** This function does
/// not check — the range check that used to live here is commented out below,
/// because in `DGainCircuit` it would be redundant and costs 8 constraints per
/// pixel. Without that guarantee the packing is not injective: a value >= 256
/// carries into its neighbour's byte slot, so two different rows can pack to
/// the same field element and therefore to the same digest.
///
/// How `DGainCircuit` discharges it:
///   - input pixels are explicitly decomposed to 8 bits;
///   - output pixels are *not*, but are forced into range by the saturating
///     select `ov * (255 - low) = out - low`, which pins `out` to either `low`
///     (8 bits, from the product decomposition) or 255. Change that constraint
///     and this obligation is silently broken.
pub fn pack_pixels_into_field_elements<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    pixels: &[AllocatedNum<F>],
) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let pixels_per_fe = (F::CAPACITY / 8) as usize;
    let shift = F::from(256u64);
    let mut packed = Vec::with_capacity(pixels.len().div_ceil(pixels_per_fe));

    for (chunk_idx, chunk) in pixels.chunks(pixels_per_fe).enumerate() {
        let mut cs = cs.namespace(|| format!("chunk {chunk_idx}"));

        // for (i, p) in chunk.iter().enumerate() {
        //     // The low byte of the repr is the pixel value (little-endian);
        //     // if the witness is >= 256 the packing constraint below fails.
        //     num_to_bits_le(
        //         cs.namespace(|| format!("range check pixel {i}")),
        //         p,
        //         8,
        //         p.get_value().map(|v| v.to_repr().as_ref()[0] as u64),
        //     )?;
        // }

        let fe = AllocatedNum::alloc(cs.namespace(|| "packed"), || {
            let mut acc = F::ZERO;
            let mut coeff = F::ONE;
            for p in chunk {
                let v = p.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                acc += v * coeff;
                coeff *= shift;
            }
            Ok(acc)
        })?;

        cs.enforce(
            || "packing",
            |mut lc| {
                let mut coeff = F::ONE;
                for p in chunk {
                    lc = lc + (coeff, p.get_variable());
                    coeff *= shift;
                }
                lc
            },
            |lc| lc + CS::one(),
            |lc| lc + fe.get_variable(),
        );

        packed.push(fe);
    }

    Ok(packed)
}


/// One link of a digest chain, in-circuit: `d' = H(d, e)` (2-ary Poseidon).
pub fn h2<F: PrimeField, CS: ConstraintSystem<F>>(
  cs: CS,
  constants: &H2<F>,
  d: AllocatedNum<F>,
  e: AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  poseidon_hash_allocated(cs, vec![d, e], constants)
}

/// Absorb a slice of elements into a running digest, in-circuit.

pub fn absorb<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  constants: &H2<F>,
  mut d: AllocatedNum<F>,
  elts: &[AllocatedNum<F>],
) -> Result<AllocatedNum<F>, SynthesisError> {
  for (i, e) in elts.iter().enumerate() {
    d = h2(cs.namespace(|| format!("h2 {i}")), constants, d, e.clone())?;
  }
  Ok(d)
}

/// Per-stage progress bookkeeping: advance the shared row counter and hand
/// the program counter to the next circuit when this stage completes.
///
/// Every stage's tail is the same four constraints, generalized from the
/// DPC circuit:
///
///   is_last <=> row + 1 == total          (inverse trick on diff)
///   row'  = row + 1 - total * is_last     (0 on handover, so the next
///                                          stage starts counting fresh)
///   pc'   = self_index + (next_index - self_index) * is_last
///
/// The reset is linear, no select needed: when `is_last` fires, row + 1
/// equals `total` exactly, so subtracting `total * is_last` lands on 0.
///
/// `total` is the number of proving steps in the stage, not necessarily the
/// image height — a stage eating two rows per step passes `height / 2`.
/// The last stage of the pipeline passes `next_index == self_index`: pc
/// stays put and the counter still wraps to 0.
///
/// Returns `(pc_next, row_next, is_last)`; `is_last` lets a stage attach
/// a check to its final step.
pub fn advance_stage<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    row: &AllocatedNum<F>,
    total: usize,
    self_index: usize,
    next_index: usize,
) -> Result<(AllocatedNum<F>, AllocatedNum<F>, AllocatedBit), SynthesisError> {
    let total_f = F::from(total as u64);
    let self_f = F::from(self_index as u64);
    let next_f = F::from(next_index as u64);

    // diff = row + 1 - total, as a value for the hints and as a linear
    // combination inside the constraints (no allocation needed).
    let diff_value = row.get_value().map(|v| v + F::ONE - total_f);

    // is_last <=> (diff == 0), via the standard inverse trick:
    //   diff * inv     = 1 - is_last   (diff != 0 forces is_last = 0)
    //   diff * is_last = 0             (diff == 0 permits is_last = 1,
    //                                   and the first line then forces it)
    let is_last = AllocatedBit::alloc(
        cs.namespace(|| "is_last"),
        diff_value.map(|d| bool::from(d.is_zero())),
    )?;
    let inv = AllocatedNum::alloc(cs.namespace(|| "diff inv"), || {
        let d = diff_value.ok_or(SynthesisError::AssignmentMissing)?;
        Ok(d.invert().unwrap_or(F::ZERO))
    })?;
    cs.enforce(
        || "diff * inv = 1 - is_last",
        |lc| lc + row.get_variable() + CS::one() - (total_f, CS::one()),
        |lc| lc + inv.get_variable(),
        |lc| lc + CS::one() - is_last.get_variable(),
    );
    cs.enforce(
        || "diff * is_last = 0",
        |lc| lc + row.get_variable() + CS::one() - (total_f, CS::one()),
        |lc| lc + is_last.get_variable(),
        |lc| lc,
    );

    // row' = row + 1 - total * is_last
    let row_next = AllocatedNum::alloc(cs.namespace(|| "row next"), || {
        let r = row.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let l = is_last
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
        Ok(r + F::ONE - if l { total_f } else { F::ZERO })
    })?;
    cs.enforce(
        || "row increment with reset",
        |lc| lc,
        |lc| lc,
        |lc| {
            lc + row.get_variable() + CS::one()
                - (total_f, is_last.get_variable())
                - row_next.get_variable()
        },
    );

    // pc' = self + (next - self) * is_last
    let pc_next = AllocatedNum::alloc(cs.namespace(|| "pc next"), || {
        is_last
            .get_value()
            .map(|b| if b { next_f } else { self_f })
            .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "pc handover",
        |lc| lc,
        |lc| lc,
        |lc| {
            lc + (self_f, CS::one()) + (next_f - self_f, is_last.get_variable())
                - pc_next.get_variable()
        },
    );

    Ok((pc_next, row_next, is_last))
}

/// Enforce `a == b` on the last step of a stage, and nothing before it:
/// `is_last * (a - b) = 0`. Used for values that are only complete once the
/// stage has seen every row, i.e. the running digests.
pub fn enforce_equal_on_last<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    is_last: &AllocatedBit,
    a: &AllocatedNum<F>,
    b: &AllocatedNum<F>,
) {
    cs.enforce(
        || "equal on the last step",
        |lc| lc + is_last.get_variable(),
        |lc| lc + a.get_variable() - b.get_variable(),
        |lc| lc,
    );
}

/// The program counter carried in the state (z[Z_PC]): enforce that it is
/// this stage's turn, and return its next value, `self_index` until the
/// stage's last step and `self_index + 1` after it.
pub fn stage_turn<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    z_pc: &AllocatedNum<F>,
    is_last: &AllocatedBit,
    self_index: usize,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let self_f = F::from(self_index as u64);
    cs.enforce(
        || "its turn",
        |lc| lc + z_pc.get_variable() - (self_f, CS::one()),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let next = AllocatedNum::alloc(cs.namespace(|| "next"), || {
        let l = is_last.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(self_f + if l { F::ONE } else { F::ZERO })
    })?;
    cs.enforce(
        || "next stage after the last step",
        |lc| lc + (self_f, CS::one()) + is_last.get_variable() - next.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(next)
}

/// Decompose `num` into `nbit` little-endian bits and enforce the packing
/// constraint `sum 2^i * b_i = num`. Without the packing constraint the bits
/// are free-floating advice, unrelated to `num` — so this is the only sound
/// way to reason about individual bits of an allocated number. Doubles as a
/// range check: `num < 2^nbit`.
pub fn num_to_bits_le<Scalar, CS>(
    mut cs: CS,
    num: &AllocatedNum<Scalar>,
    nbit: usize,
    value: Option<u64>,
) -> Result<Vec<AllocatedBit>, SynthesisError>
where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
{
    let bits = (0..nbit)
        .map(|i| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("bit {i}")),
                value.map(|v| (v >> i) & 1 == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    cs.enforce(
        || "packing",
        |mut lc| {
            let mut coeff = Scalar::ONE;
            for b in &bits {
                lc = lc + (coeff, b.get_variable());
                coeff = coeff.double();
            }
            lc
        },
        |lc| lc + CS::one(),
        |lc| lc + num.get_variable(),
    );

    Ok(bits)
}

/// Low byte of an allocated number's witness, as the `u64` hint the bit
/// gadgets want. Only meaningful for values the caller knows are bytes.
pub fn byte_hint<F: PrimeField>(n: &AllocatedNum<F>) -> Option<u64> {
    n.get_value().map(|v| v.to_repr().as_ref()[0] as u64)
}

/// Allocate one byte and range-check it to 8 bits.
pub fn alloc_byte<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    value: u8,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let n = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(F::from(value as u64)))?;
    num_to_bits_le(cs.namespace(|| "range check"), &n, 8, Some(value as u64))?;
    Ok(n)
}

/// Allocate a row of bytes, each range-checked to 8 bits. This is the form
/// every *input* row takes: it discharges the caller obligation of
/// [`pack_pixels_into_field_elements`] and makes any arithmetic on the
/// pixels sound.
pub fn alloc_bytes<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    bytes: &[u8],
) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| alloc_byte(cs.namespace(|| format!("px {i}")), b))
        .collect()
}

/// Allocate a row of bytes with NO range check. For *output* rows whose
/// every element is pinned by a constraint to something already in range
/// (a checked input, a `low` byte, the constant 255). The caller owns that
/// argument; if it stops holding, the digest stops being injective.
pub fn alloc_bytes_unchecked<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    bytes: &[u8],
) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            AllocatedNum::alloc(cs.namespace(|| format!("px {i}")), || Ok(F::from(b as u64)))
        })
        .collect()
}

/// Pack a byte row and absorb it into a running digest: `d' = absorb(d, pack(row))`.
/// Chunk boundaries are part of the layout; the verifier's native replay
/// must pack exactly this slice.
pub fn absorb_bytes<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    constants: &H2<F>,
    d: AllocatedNum<F>,
    row: &[AllocatedNum<F>],
) -> Result<AllocatedNum<F>, SynthesisError> {
    let packed = pack_pixels_into_field_elements(cs.namespace(|| "pack"), row)?;
    absorb(cs.namespace(|| "absorb"), constants, d, &packed)
}

/// Enforce `out = min(a * b, 255)` for two bytes `a`, `b`.
///
/// Soundness needs both factors already range-checked to 8 bits by the
/// caller: the 16-bit decomposition of `a*b` is only unique when the product
/// fits, so `low` is only the true low byte and `high != 0` only a genuine
/// overflow under that assumption. `out` may be unchecked advice: the final
/// select pins it to `low` (8 bits) or 255.
///
///   a*b decomposed into 16 bits (low byte `low`, high byte `high`)
///   ov = (high != 0)             -- overflow flag, inverse trick
///   ov * (255 - low) = out - low -- select: out = ov ? 255 : low
pub fn enforce_saturating_mul<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    a: &AllocatedNum<F>,
    b: &AllocatedNum<F>,
    out: &AllocatedNum<F>,
) -> Result<(), SynthesisError> {
    // a*b must be its own variable: an R1CS linear combination cannot
    // contain a product of two witnesses.
    let prod = a.mul(cs.namespace(|| "a*b"), b)?;
    let prod_val = byte_hint(a).zip(byte_hint(b)).map(|(x, y)| x * y);
    let prod_bits = num_to_bits_le(cs.namespace(|| "decompose a*b"), &prod, 16, prod_val)?;

    let weighted = |bits: &[AllocatedBit], sign: F| {
        let bits = bits.to_vec();
        move |mut lc: LinearCombination<F>| {
            let mut coeff = sign;
            for bit in &bits {
                lc = lc + (coeff, bit.get_variable());
                coeff = coeff.double();
            }
            lc
        }
    };
    // high = a*b div 256 (bits 8..16); overflow iff high != 0
    let high_lc = weighted(&prod_bits[8..], F::ONE);
    // -low = -(a*b mod 256) (bits 0..8), used inside the select
    let sub_low = weighted(&prod_bits[..8], -F::ONE);

    // ov = (high != 0), via the standard inverse trick:
    //   high * inv = ov
    //   high * (1 - ov) = 0
    // Together these force ov = 1 exactly when high != 0 and ov = 0 when
    // high = 0, so ov is boolean by construction.
    let high_val = prod_val.map(|p| p >> 8);
    let ov = AllocatedNum::alloc(cs.namespace(|| "overflow"), || {
        let h = high_val.ok_or(SynthesisError::AssignmentMissing)?;
        Ok(if h != 0 { F::ONE } else { F::ZERO })
    })?;
    let inv = AllocatedNum::alloc(cs.namespace(|| "high inv"), || {
        let h = high_val.ok_or(SynthesisError::AssignmentMissing)?;
        Ok(F::from(h).invert().unwrap_or(F::ZERO))
    })?;
    cs.enforce(
        || "high * inv = ov",
        &high_lc,
        |lc| lc + inv.get_variable(),
        |lc| lc + ov.get_variable(),
    );
    cs.enforce(
        || "high * (1 - ov) = 0",
        &high_lc,
        |lc| lc + CS::one() - ov.get_variable(),
        |lc| lc,
    );

    // Saturating select, one constraint: ov * (255 - low) = out - low
    cs.enforce(
        || "ov * (255 - low) = out - low",
        |lc| lc + ov.get_variable(),
        |lc| sub_low(lc + (F::from(255u64), CS::one())),
        |lc| sub_low(lc + out.get_variable()),
    );
    Ok(())
}