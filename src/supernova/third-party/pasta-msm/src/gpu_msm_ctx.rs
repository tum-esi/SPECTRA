#[cfg(feature = "cuda")]
use core::{ffi::c_void, ptr::NonNull};
use pasta_curves::pallas;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_pippenger_pallas_context_create(
        context: *mut *mut c_void,
        points: *const pallas::Affine,
        npoints: usize,
    ) -> sppark::Error;

    fn cuda_pippenger_pallas_context_invoke(
        context: *mut c_void,
        out: *mut pallas::Point,
        scalars: *const pallas::Scalar,
        npoints: usize,
    ) -> sppark::Error;

    fn cuda_pippenger_pallas_context_destroy(context: *mut c_void);
}

#[cfg(feature = "cuda")]
pub struct PallasGpuMsmContext {
    context: NonNull<c_void>,
    max_points: usize,
}

// SAFETY: the context is exclusively owned, its operations require `&mut self`,
// and the C++ entry points reselect the CUDA device recorded at construction.
// It may therefore move between host threads, but it must not be used by
// multiple threads concurrently.
#[cfg(feature = "cuda")]
unsafe impl Send for PallasGpuMsmContext {}

#[cfg(feature = "cuda")]
impl PallasGpuMsmContext {
    pub fn is_recommended_for(num_scalars: usize) -> bool {
        num_scalars >= crate::CUDA_MSM_MIN_POINTS
    }

    pub fn new(bases: &[pallas::Affine]) -> Result<Self, String> {
        if bases.is_empty() {
            return Err("MSM bases must not be empty".into());
        }

        let mut raw = core::ptr::null_mut();
        let status = unsafe {
            cuda_pippenger_pallas_context_create(
                &mut raw,
                bases.as_ptr(),
                bases.len(),
            )
        };
        if status.code != 0 {
            return Err(status.to_string());
        }

        let context =
            NonNull::new(raw).ok_or("CUDA returned a null MSM context")?;

        Ok(Self {
            context,
            max_points: bases.len(),
        })
    }

    pub fn msm(
        &mut self,
        scalars: &[pallas::Scalar],
    ) -> Result<pallas::Point, String> {
        if scalars.is_empty() || scalars.len() > self.max_points {
            return Err("invalid number of MSM scalars".into());
        }

        let mut out = pallas::Point::default();
        let status = unsafe {
            cuda_pippenger_pallas_context_invoke(
                self.context.as_ptr(),
                &mut out,
                scalars.as_ptr(),
                scalars.len(),
            )
        };
        if status.code != 0 {
            return Err(status.to_string());
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
impl Drop for PallasGpuMsmContext {
    fn drop(&mut self) {
        unsafe {
            cuda_pippenger_pallas_context_destroy(self.context.as_ptr());
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub struct PallasGpuMsmContext {
    _unavailable: (),
}

#[cfg(not(feature = "cuda"))]
impl PallasGpuMsmContext {
    pub fn is_recommended_for(_num_scalars: usize) -> bool {
        false
    }

    pub fn new(_bases: &[pallas::Affine]) -> Result<Self, String> {
        Err("CUDA support is not available".into())
    }

    pub fn msm(
        &mut self,
        _scalars: &[pallas::Scalar],
    ) -> Result<pallas::Point, String> {
        Err("CUDA support is not available".into())
    }
}
