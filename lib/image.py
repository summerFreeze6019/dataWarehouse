from __future__ import division

import glob
import os
import re
import matplotlib.pyplot as pp
import numpy as np

import fastcluster
from scipy import optimize
from scipy.cluster.hierarchy import fcluster

import skimage.data
from skimage import color, filter, io, segmentation
from skimage import exposure, transform, measure, feature


def preprocess(image, height=50, block_size=50):
    """Turn to greyscale, scale to a height, and then threshold to binary
    """

    image = color.rgb2grey(image)
    size_factor = float(height) / image.shape[0]
    new_size = [int(e*size_factor) for e in image.shape]
    image = transform.resize(image, new_size)
    image = filter.threshold_adaptive(image, block_size=30)

    return image


def crop(img, boundaries):
    h = boundaries['box_h']
    w = boundaries['box_w']
    x = boundaries['box_x']
    y = boundaries['box_y']
    
    return img[y:y+h, x:x+w]


def split_rigid(image, charseps):
    """Split an image into characters. Charseps should be a list of ints
    giving the horizontal location to split
    """
    n_chars = len(charseps) - 1

    chars = []
    for i in range(n_chars):
        char = image[:, charseps[i]:charseps[i+1]]
        char = transform.resize(char, output_shape=(25, 15))
        char = filter.threshold_adaptive(char, block_size=30)
        chars.append(char)

    return chars


def split(image, plot=False):
    """Split an image into characters, using a clustering approach
    """
    iimage = segmentation.clear_border(np.logical_not(image))

    downsample = 1
    char_width = 30
    #resize_to = (45, 27)
    #resize_to = (41, 25)
    resize_to = (38, 23)
    # weight given to the y coordinate
    y_wieght = 0.1


    # x coordinate of every lit up pixle
    x = np.asarray(np.vstack(np.where(iimage > 0)), dtype=np.float)
    # needs to be 2d for the clustering alg.
    x = x.T[::downsample, :]

    x[:,0] *= y_wieght
    #x = x.reshape(len(x), 1)
    # make 6 clusters using single linkage

    linkage_method = 'single'

    while True:
        Z = fastcluster.linkage(x, method=linkage_method)
        labels = fcluster(Z, t=6, criterion='maxclust')

        # try kmeans? this is for debugging
        #centers, _ = kmeans(x, k_or_guess=6)
        #labels, _ = vq(x, centers)

        unique_labels = np.unique(labels)

        # single linkage can fail to give 6 clusters in the face
        # of equal distances. So let's fall back to ward
        if len(unique_labels) != 6:
            if linkage_method == 'ward':
                raise Exception('Sorry!!!!')
            linkage_method = 'ward'
            continue

        cluster_xlims = []
        sizes = []
        for i in unique_labels:
            x_coords = x[labels==i][:,1]
            #print 'Cluster %d has %d elements' % (i, len(x_coords))
            sizes.append(len(x_coords))

            cluster_xlims.append((np.min(x_coords), np.max(x_coords)))

        if np.min(sizes) < 75 / downsample:
            bad_cluster = unique_labels[np.argmin(sizes)]
            #print 'REMOVING CLUSTER', bad_cluster

            x = x[labels != bad_cluster]
        else:
            break

    cluster_xlims.sort(key=lambda x: x[0])

    chars = []
    for xlim in cluster_xlims:
        char = np.zeros((iimage.shape[0], char_width), dtype=np.float)
        w = xlim[1] - xlim[0]

        try:
            # try to embed the character into a white background
            # before shrinking to the final output size
            char[:, char_width/2-w/2:char_width/2+w/2] = iimage[:, xlim[0]:xlim[1]]

            char = transform.resize(char, output_shape=resize_to)

        except ValueError:
            # but if it doesn't fit, shrink it to fit
            # when shrinking, we want it to be float
            char = transform.resize(1.0*iimage[:, xlim[0]:xlim[1]],
                            output_shape=resize_to)
            # but then we need to threshold it back

        #char = filter.threshold_adaptive(char, block_size=resize_to[1])
        chars.append(np.asarray(char, dtype=np.bool))
        #chars.append(char)


    if plot:
        fig, axes = pp.subplots(ncols=6)
        for ax, char in zip(axes, chars):
            ax.imshow(char)

    return chars


def align_fft(image, target):
    """Align two images using the cross-correlation objective function
    by transforming into the fourier domain
    
    For details, see R Szeliski, "Image Alignment and Stitching: A Tutorial",
    page 20
    """
    def convolve(a1, a2):
        if not (a1.ndim == a2.ndim == 2):
            raise ValueError('Arrays must be 2ds')

        f_a1 = np.fft.fft2(a1)
        f_a2 = np.fft.fft2(a2)

        E_cc = np.fft.ifft2(f_a1 * f_a2.conjugate())
        return np.real(E_cc)

    # for colored data (2d), do the correlation on each channel separately
    # and then sum the results
    if image.ndim == target.ndim == 3:
        E_cc = np.sum((convolve(image[:,:,i], target[:,:,i]) for i in range(3)), axis=0)
    else:
        E_cc = convolve(image, target)

    # this gets the 2d coordinates of the argmax
    y, x = np.unravel_index(np.argmax(E_cc), E_cc.shape)
    
    # apply the minimum image convention to get negative translations, since
    # the transform doesn't do wrapping
    y2 = image.shape[0] - y
    x2 = image.shape[1] - x
    if abs(y2) < y:
        y = -y2
    if abs(x2) < x:
        x = -x2
    #print x,y

    # transform the initial image, and return it
    tform = transform.SimilarityTransform(translation=(x, y))
    result = transform.warp(image, inverse_map=tform)
    return result


if __name__ == '__main__':
    import glob
    import random
    from skimage import io, data, color

    img1 = io.imread(random.sample(glob.glob('images2/*.png'),1)[0])
    img2 = io.imread(random.sample(glob.glob('images2/*.png'),1)[0])

    
    tform = transform.SimilarityTransform(translation=(10, -10))
    img1 = transform.warp(img1, inverse_map=tform)

    result = align_fft(img1, img2)
    
    fig, axes = pp.subplots(ncols=2)
    pp.gray()
    axes[0].imshow(color.rgb2grey(img1) + color.rgb2grey(img2))
    axes[1].imshow(color.rgb2grey(result) + color.rgb2grey(img2))
    
    pp.show()
