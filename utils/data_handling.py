import numpy as np
import os
import glob
from scipy.ndimage import map_coordinates
import nibabel as nib
from nibabel import streamlines
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from dipy.io import read_bvals_bvecs


class DataHandler(object):

    def __init__(self, **args):

        self.params = args['params']
        self.dwi_path = self.params['DWI_path']
        self.brain_mask_path =self.params['brain_mask_path']
        self.wm_mask_path = self.params['wm_mask_path']
        self.tractogram_path = self.params['tractogram_path']
        self.voxel_size = self.params['voxel_size']
        self.max_val = 255

        self.dwi = None
        self.bvals = None
        self.bvecs = None
        self.brain_mask = []
        self.wm_mask = []
        self.tractogram = None
        if self.dwi_path is not None:
            self.load_dwi()
            self.load_b_table()
        if self.brain_mask_path is not None:
            self.brain_mask = self.load_mask(self.brain_mask_path)
        if self.wm_mask_path is not None:
            self.wm_mask = self.load_mask(self.wm_mask_path)
        if self.tractogram_path is not None:
            self.load_tractogram()

    def load_dwi(self):
        dwi_file = get_file_path(os.getcwd(), self.dwi_path, "*.nii*")
        dwi_data = nib.load(dwi_file)
        self.dwi = dwi_data.get_data().astype("float32")

    def load_b_table(self):
        bval_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvals")
        bvec_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvecs")
        self.bvals, self.bvecs = read_bvals_bvecs(bval_file, bvec_file)

    def load_tractogram(self):
        tractogram_data = streamlines.load(self.tractogram_path)
        self.tractogram = tractogram_data.streamlines

    @staticmethod
    def load_mask(mask_path):
        dwi_data = nib.load(mask_path)
        return dwi_data.get_data().astype("float32")

    def normalize_dwi(self, b0):
        """ Normalize dwi by the first b0.
        Parameters:
        -----------
        weights : ndarray of shape (X, Y, Z, #gradients)
            Diffusion weighted images.
        b0 : ndarray of shape (X, Y, Z)
            B0 image.
        Returns
        -------
        ndarray
            Diffusion weights normalized by the B0.
        """
        weights = self.dwi
        b0 = b0[..., None]  # Easier to work if it is a 4D array.

        # Make sure in every voxels weights are lower than ones from the b0.
        nb_erroneous_voxels = np.sum(weights > b0)
        if nb_erroneous_voxels != 0:
            weights = np.minimum(weights, b0)

        # Normalize dwi using the b0.
        weights_normed = weights / b0
        weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

        return weights_normed

    def get_spherical_harmonics_coefficients(self, sh_order=8, smooth=0.006):
        """ Compute coefficients of the spherical harmonics basis.
        Parameters
        -----------
        dwi : `nibabel.NiftiImage` object
            Diffusion signal as weighted images (4D).
        bvals : ndarray shape (N,)
            B-values used with each direction.
        bvecs : ndarray shape (N, 3)
            Directions of the diffusion signal. Directions are
            assumed to be only on the hemisphere.
        sh_order : int, optional
            SH order. Default: 8
        smooth : float, optional
            Lambda-regularization in the SH fit. Default: 0.006.
        Returns
        -------
        sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
            Spherical harmonics coefficients at every voxel. The actual number of
            coeffs depends on `sh_order`.
        """

        bvals = np.asarray(self.bvals)
        bvecs = np.asarray(self.bvecs)
        dwi_weights = self.dwi

        # Exract the averaged b0.
        b0_idx = bvals == 0
        b0 = dwi_weights[..., b0_idx].mean(axis=3) + 1e-10

        # Extract diffusion weights and normalize by the b0.
        bvecs = bvecs[np.logical_not(b0_idx)]
        weights = dwi_weights[..., np.logical_not(b0_idx)]
        weights = self.normalize_dwi(weights, b0)

        # Assuming all directions are on the hemisphere.
        raw_sphere = HemiSphere(xyz=bvecs)

        # Fit SH to signal
        sph_harm_basis = sph_harm_lookup.get('mrtrix')
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
        data_sh = np.dot(weights, invB.T)
        return data_sh

    def resample_dwi(self, directions=None, sh_order=8, smooth=0.006):
        """ Resamples a diffusion signal according to a set of directions using spherical harmonics.
        Parameters
        -----------
        directions : `dipy.core.sphere.Sphere` object, optional
            Directions the diffusion signal will be resampled to. Directions are
            assumed to be on the whole sphere, not the hemisphere like bvecs.
            If omitted, 100 directions evenly distributed on the sphere will be used.
        sh_order : int, optional
            SH order. Default: 8
        smooth : float, optional
            Lambda-regularization in the SH fit. Default: 0.006.
        """
        data_sh = self.get_spherical_harmonics_coefficients(self.dwi, self.bvals, self.bvecs,
                                                            sh_order=sh_order, smooth=smooth)
        sphere = get_sphere('repulsion100')
        if directions is not None:
            sphere = Sphere(xyz=directions)

        sph_harm_basis = sph_harm_lookup.get('mrtrix')
        Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        data_resampled = np.dot(data_sh, Ba.T)

        return data_resampled

    def mask_dwi(self):
        dwi_vol = self.dwi
        mask_vol = self.brain_mask
        if mask_vol.ndim == 3:
            masked_dwi = dwi_vol * np.tile(mask_vol[..., None], (1, 1, 1, dwi_vol.shape[-1]))
        else:
            masked_dwi = dwi_vol * mask_vol

        return masked_dwi


def align_streamlines_to_grid(tract, dwi):
    dwi_vec = np.diag(dwi.affine)[0:3]
    tract_vec = np.diag(tract.affine)[0:3]
    ratio_vec = tract_vec / dwi_vec
    if not all(ratio_vec == 1.0):
        ratio_mat = np.diag(ratio_vec)
        all_streamlines = []
        for i in range(len(tract.streamlines)):
            all_streamlines.append(np.matmul(tract.streamlines[i], ratio_mat))
        return all_streamlines
    else:
        return tract.streamlines


def eval_volume_at_3d_coordinates(volume, coords):
    """ Evaluates the volume data at the given coordinates using trilinear interpolation.
    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    Returns
    -------
    output : 2D array
        Values from volume.
    """
    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode="nearest")

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i],
                                         coords.T, order=1, mode="nearest")
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def get_streamlines_lengths(streamlines_list):
    """
INPUT: fibers_list - python list of size N, where each element is a fiber represented by a Ln x 3 np array

OUTPUT: lengths - a np array (vector) of length N, holding the #points in each fiber
"""
    lengths = np.zeros(len(streamlines_list))

    for i in range(len(streamlines_list)):
        lengths[i] = streamlines_list[i].shape[0]

    return lengths


def get_file_path(curr_path, target_dir, extension):
    os.chdir(target_dir)
    for file in glob.glob(extension):
        file_path = os.path.join(target_dir, file)
    os.chdir(curr_path)

    return file_path
