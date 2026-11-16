

// load a .npy file as a Vec<Vec<u8>>

use std::path::Path;

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::{read_npy, ReadNpyError};

/// One demosaiced frame split into its three colour planes, each `H x W`
/// row-major, so a plane can be sliced per row exactly like a mono frame.
///
/// The circuit constrains one field element per sample, so the `(H, W, 3)`
/// array pipeline.py writes has to be unzipped: `r[y][x]` is what numpy calls
/// `rgb[y, x, 0]`.
pub struct RgbPlanes {
    pub r: Vec<Vec<u8>>,
    pub g: Vec<Vec<u8>>,
    pub b: Vec<Vec<u8>>,
}

/// Split an `(H, W, 3)` channel-last array into planes.
pub fn split_planes(rgb: &Array3<u8>) -> RgbPlanes {
    assert_eq!(rgb.shape()[2], 3, "expected an (H, W, 3) array, got {:?}", rgb.shape());
    let plane = |c: usize| -> Vec<Vec<u8>> {
        rgb.index_axis(Axis(2), c)
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect()
    };
    RgbPlanes { r: plane(0), g: plane(1), b: plane(2) }
}

/// Load the demosaic stage output (`3_demosaic/demosaic_cfa.npy`) as planes.
pub fn load_rgb_planes<P: AsRef<Path>>(path: P) -> Result<RgbPlanes, ReadNpyError> {
    let a: Array3<u8> = read_npy(path)?;
    Ok(split_planes(&a))
}

// this loads the frame in memory for the proof
pub fn load_frame<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<u8>>, ReadNpyError> {
    let a: Array2<u8> = read_npy(path)?;
    Ok(a.rows().into_iter().map(|r| r.to_vec()).collect())
}

pub fn load_lut<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<bool>>, ReadNpyError> {
    let a: Array2<bool> = read_npy(path)?;
    Ok(a.rows().into_iter().map(|r| r.to_vec()).collect())
}

// load the boolean defect mask (the python pipeline saves it as uint8 0/1)
pub fn load_mask<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<bool>>, ReadNpyError> {
    let a: Array2<u8> = read_npy(path)?;
    Ok(a.rows()
        .into_iter()
        .map(|r| r.iter().map(|v| *v != 0).collect())
        .collect())
}

// get_dims -> W , H
pub fn get_dims(arr: Vec<Vec<u8>>) -> (usize,usize) {
    (arr.len(),arr[0].len())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_planes_unzips_channel_last() {
        // rgb[y, x, c] = 100*c + 10*y + x, so every plane is recognisable.
        let a = Array3::from_shape_fn((2, 3, 3), |(y, x, c)| (100 * c + 10 * y + x) as u8);
        let p = split_planes(&a);
        assert_eq!(p.r, vec![vec![0, 1, 2], vec![10, 11, 12]]);
        assert_eq!(p.g, vec![vec![100, 101, 102], vec![110, 111, 112]]);
        assert_eq!(p.b, vec![vec![200, 201, 202], vec![210, 211, 212]]);
    }
    #[test]
    fn loads_npy() {
        let f = load_frame("../../scripts/python/data/01_defective/gradient_horizontal.npy").unwrap();
        println!("{}x{} first={:?}", f.len(), f[0].len(), &f[0][..4]);
    }
    #[test]
    fn check_dimensions() {
        let f = load_frame("../../scripts/python/data/01_defective/gradient_horizontal.npy").unwrap();
        println!("len of the columns {} , len of the rows {}", f[0].len(), f.len())
    }
}
