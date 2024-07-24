"""Make the training set
"""
import numpy as np
import warnings
import scipy.misc
import cPickle as pickle
import matplotlib.pyplot as pp
import logging
logging.basicConfig(level=logging.DEBUG)

from skimage import color, filter, io, transform, morphology
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier

def shift(img):
    """Shift a binary image randomly within the frame
    
    Uses a convex hull calculation to make sure it doesn't translate
    the image out of the frame.
    """
    hull = morphology.convex_hull_image(1-img)

    horizontal = np.where(np.sum(hull, axis=0) > 0)[0]
    vertical = np.where(np.sum(hull, axis=1) > 0)[0]

    max_left = -np.min(horizontal)
    max_right = img.shape[1] - np.max(horizontal)
    max_down = -np.min(vertical)
    max_up = img.shape[0] - np.max(vertical)
    
    shift_x = np.random.randint(max_left, max_right)
    shift_y = np.random.randint(max_down, max_up)

    #print "SHIFT", shift_x, shift_y
    
    def shift(xy):
        xy[:, 0] -= shift_x
        xy[:, 1] -= shift_y
        return xy
        
    return np.logical_not(transform.warp(np.logical_not(img), shift))

def noise(img, rho=0.01, sigma=0.5, block_size=50):
    """Add two forms of noise to a binary image
    
    First, flip a fraction, rho, of the bits. The bits to flip are
    selected uniformly at random.
    
    Second, add white noise to the image, and then re-threshold it back to
    binary. Here, errors in the thresholding lead to a new "splotchy" error
    pattern, especially near the edges.
    """
    
    mask = scipy.sparse.rand(img.shape[0], img.shape[1], density=rho)
    mask.data = np.ones_like(mask.data)
    img = np.mod(img + mask, 2)
    
    img = img + sigma * np.random.random(img.shape)
    img = filter.threshold_adaptive(img, block_size=block_size)
    
    return img

def modify(img):
    """Randomly modify an image
    
    This is a preprocessing step for training an OCR classifier. It takes
    in an image and casts it to greyscale, reshapes it, and adds some
    (1) rotations, (2) translations and (3) noise.
    
    If more efficiency is needed, we could factor out some of the initial
    nonrandom transforms.
    """
    
    block_size = np.random.uniform(20, 40)
    rotation = 5*np.random.randn()
    
    #print 'BLOCK SIZE', block_size
    #print 'ROTATION  ', rotation
    
    img = color.rgb2grey(img)
    img = transform.resize(img, output_shape=(50,30))
    img = filter.threshold_adaptive(img, block_size=block_size)
    
    # rotate the image
    img = np.logical_not(transform.rotate(np.logical_not(img), rotation))
    # translate the image
    img = shift(img)
    # add some noise to the image
    img = noise(img)
    
    img = transform.resize(img, output_shape=(25,15))
    return filter.threshold_adaptive(img, block_size=25)
    
    
def main():
    """
    Create a demonstration image showing what the syntetic training data
    looks like
    """
    nrows = 6
    ncols = 6
    
    fig, ax = pp.subplots(nrows=nrows, ncols=ncols, figsize=(8, 5))
    pp.gray()
    
    for i in range(nrows):
        for j in range(ncols):
            c = np.random.randint(10)
            img = modify(io.imread('training/base/%d-0.jpg' % c))
            ax[i, j].imshow(img, interpolation='none')
            ax[i, j].axis('off')
            
    #pp.show()
    pp.suptitle('Synthetic Training Data (Example), accuracy ~ 0.98')
    pp.savefig('synthetic.png')
    
    

def fit(output_fn):
    """Fit a classifier, and save it to disk
    
    Load up the 'base' training examples, make a bunch of random
    modifications, and then use them as training examples to fit a classifier.
    
    """
    num_digits = 10
    mods_per_digit = 200
    
    # we're anticipating that the images are 25x15.
    X = np.zeros((num_digits*mods_per_digit, 25*15), dtype=np.bool)
    y = np.zeros((num_digits*mods_per_digit), dtype=np.int)
    
    # load up the data and make the random perturbations
    for i in range(num_digits):
       img = io.imread('data/training_base/%d.jpg' % i)
       for j in range(mods_per_digit):
           # collapse the image to a 1d array
           X[i*mods_per_digit + j] = modify(img).reshape(-1)
           y[i*mods_per_digit + j] = i

    logging.info("SAMPLED DATA")

    clf = svm.LinearSVC()
    
    # run 5-fold cross validaton to assess the accuracy of the classfier
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)        
    logging.info('CLASSIFIER SCORES')
    logging.info('%s, mean=%s', scores, np.mean(scores))
    
    # fit a new classifier and save it to disk
    clf.fit(X, y)
    with open(output_fn, 'wb') as fid:
        pickle.dump(clf, fid)
        
    logging.info('SAVED CLASSIFIER TO %s' % output_fn)

if __name__ == '__main__':
    fit('classifier.pickl')
    #main()
    