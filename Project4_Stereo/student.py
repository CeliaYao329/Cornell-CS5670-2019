# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    N = len(images)
    (height, width, channel) = images[0].shape
    pixels = np.zeros((height, width, channel, N), dtype=float)
    for n in range(N):
        for h in range(height):
            for w in range(width):
                for c in range(channel):
                    pixels[h, w, c, n] = images[n][h, w, c]

    lights_t = np.transpose(lights)
    from numpy.linalg import inv
    from numpy import linalg as LA
    alb = np.zeros((height, width, channel), dtype=float)
    normals = np.zeros((height, width, 3), dtype=float)
    for h in range(height):
        for w in range(width):
            g = np.zeros((3, 1), dtype=float)
            blank = False
            for c in range(channel):
                i = pixels[h, w, c, ].transpose()
                g = inv(np.dot(lights_t, lights)).dot(np.dot(lights_t, i))
                alb[h, w, c] = LA.norm(g)
                if alb[h, w, c] < 1e-7:
                    blank = True
                    alb[h, w, c] = 0
            if blank is False:
                normals[h, w, ] = g.reshape((3,))/LA.norm(g)
    # print alb, normals
    return alb, normals

# from scipy.misc import imread
#
# test_images = [imread('test_materials/sphere%02d.png' % i) for i in xrange(3)]
# test_lights = np.load('test_materials/sphere_lights.npy')
# correct_normals = np.load('test_materials/sphere_normals.npy')
#
# albedo, normals = compute_photometric_stereo_impl(test_lights.T, test_images)
# if (np.abs(normals - correct_normals) < 1e-5)[100:-100, 100:-100].all():
#     print("normal ok")
# if (np.abs(albedo[100:-100, 100:-100] / 255.0 - 1) < 1e-1).all():
#     print("albedo ok")

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    raise NotImplementedError()



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    raise NotImplementedError()


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    raise NotImplementedError()


