#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: float, Y: float, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    K = X.shape[0]
    fvs = np.zeros((K, feature_width**2))
    padding = feature_width//2
    for i in range(K):
        fv = image_bw[Y[i]-padding+1:Y[i]+padding+1, X[i]-padding+1:X[i]+padding+1].flatten()
        fv = fv/np.linalg.norm(fv)
        fvs[i] = fv

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
