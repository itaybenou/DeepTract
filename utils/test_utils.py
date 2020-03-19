import random
import numpy as np
from scipy.stats import rv_discrete
import scipy.ndimage as ndimage
from nibabel import streamlines


def normalize_logits(logits):
    """
INPUT: logits - tensor of size N x max_time_steps x #out_dirctions (N is the batch size)

OUTPUT: pdf - a normalized probability density function (same size as input)
"""
    b = np.amax(logits, axis=2)
    pdf = np.exp(logits - np.repeat(b[:, :, np.newaxis], logits.shape[2], axis=2))

    return pdf / np.repeat(np.sum(pdf, axis=2)[:, :, np.newaxis], logits.shape[2], axis=2)


def sample_from_pdf(pdf, n):
    """
INPUT: pdf of a single time step - tensor of size N x #out_directions (N is the batch size)
       n - how many samples to draw from each pdf

OUTPUT: selected_dirs - the indices of directions sampled from the pdfs, a vector of size N
"""

    numbers = np.arange(pdf.shape[1])
    selected_dirs = np.zeros((len(pdf), n), dtype=int)
    for i in range(len(pdf)):
        rand_samples = rv_discrete(values=(numbers, pdf[i, :]))
        selected_dirs[i, :] = int(rand_samples.rvs(size=n))

    return selected_dirs


def argmax_from_pdf(pdf):
    """
INPUT: pdf of a single time step - tensor of size N x #out_directions (N is the batch size)
       n - how many samples to draw from each pdf

OUTPUT: selected_dirs - the indices of directions sampled from the pdfs, a vector of size N
"""

    selected_dirs = np.argmax(pdf, axis=1)

    return selected_dirs


def idx2direction(direction_idxs, sphere):
    """
INPUT: direction_idxs - tensor of size N x 1 (where N is the batch size) holding indices of sphere directions

OUTPUT: direction_vecs - tensor of size N x 3 (where N is the batch size) holding corresponding sphere directions
"""
    direction_vecs = np.zeros((len(direction_idxs), 3))

    for i in range(len(direction_idxs)):
        if direction_idxs[i] == 724:
            direction_vecs[i, :] = [0.0, 0.0, 0.0]
        else:
            direction_vecs[i, :] = [sphere.x[direction_idxs[i]], sphere.y[direction_idxs[i]],
                                    sphere.z[direction_idxs[i]]]

    return direction_vecs


def fiber_lengths(fibers_list, voxel_size):
    """
INPUT: fibers_list - python list of size N, where each element is a fiber represented by a Ln x 3 np array
        voxel_size - a vector of size 3, (vx,vy,vz), representing the voxel size in mm.

OUTPUT: lengths - a np array (vector) of length N, holding the tota, arc-length of each fiber
"""
    lengths = np.zeros(len(fibers_list))

    for i in range(len(fibers_list)):
        lengths[i] = sum(np.linalg.norm((fibers_list[i][1:, :] - fibers_list[i][:-1, :]) * voxel_size, axis=1))

    return lengths


def mask_dilate(mask, SE=ndimage.generate_binary_structure(3, 1)):
    """
INPUT: mask - a 3D binary image (np array)
       SE - a 3D boolean structure element

OUTPUT: out_mask - dilated mask using the specified SE
"""
    out_mask = ndimage.binary_dilation(mask, structure=SE).astype(mask.dtype)

    return out_mask


def is_within_mask(positions, mask):
    """
INPUT: positions - a Nx3 vector of brain positions
       mask - a 3D binary mask

OUTPUT: is_inside - a Nx1 boolean vector specifying if p=positions[i,:] is inside the mask
"""
    out_mask = mask[positions[:, 0].astype(int), positions[:, 1].astype(int), positions[:, 2].astype(int)]

    return out_mask


def calc_angles_matrix(sphere):
    """
INPUT: sphere object with d vertices uniformally distributed on the unit sphere

OUTPUT: theta_mat = dxd symmetric matrix, where theta_mat[i,j]=angle(vi,vj) in degrees
# """
    vecs = np.zeros((len(sphere.x), 3))
    vecs[:, 0] = sphere.x
    vecs[:, 1] = sphere.y
    vecs[:, 2] = sphere.z

    dot_mat = np.matmul(vecs, vecs.T)
    norm_vec = np.linalg.norm(vecs, axis=1)[:, None]
    norm_mat = np.matmul(norm_vec, norm_vec.T)

    res_mat = np.clip(np.divide(dot_mat, norm_mat), -1, 1)
    theta_mat = np.array(np.arccos(res_mat))
    theta_mat = np.nan_to_num(theta_mat)
    theta_mat_deg = np.degrees(theta_mat)

    return theta_mat_deg


def init_seeds(WM_mask, n_seeds):
    """
INPUT: a binary WM mask - np array of size XxYxZ with "1" in WM voxels, "0" othwerwise
       n_seeds - how many seed points to draw

OUTPUT: seed_points - zero-padded np array of size n x 3, holding n random points within the WM_mask
"""

    # mask_idxs = 0.5 * np.array(np.nonzero(WM_mask)).T
    mask_idxs = np.array(np.nonzero(WM_mask)).T
    seed_points = mask_idxs[random.sample(range(len(mask_idxs)), n_seeds)]
    return seed_points


def zero_pad_seeds(seed_points, n, time_steps):
    """
INPUT: a binary WM mask - np array of size XxYxZ with "1" in WM voxels, "0" othwerwise
       n - size of seed_points batch
       time_steps - length of the RNN graph

OUTPUT: seed_points - zero-padded np array of size n x time_steps x 3, holding n random points within the WM_mask
"""

    padded_seed_points = np.zeros((n, time_steps, 3))
    padded_seed_points[:, 0, :] = seed_points

    return padded_seed_points


def calc_entropy_threshold(entropy_params, tracking_steps):
    return entropy_params[0] * np.exp(-np.arange(tracking_steps) / entropy_params[1]) + entropy_params[2]


def output_tractogram(streamlines_list):
    tractogram_array = streamlines.array_sequence.ArraySequence()
    for j in range(len(streamlines_list)):
        for i in range(len(streamlines_list[j])):
            streamline = streamlines_list[j][i]
            tractogram_array.append(streamline)

    return tractogram_array
