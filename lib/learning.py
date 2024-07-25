import numpy as np

from skimage import io
from image import align_fft, preprocess, split, crop


def predict(clf, boundaries, img=None, align_template=None):
    if isinstance(img, basestring):
        raw_img = io.imread(img)
    elif isinstance(img, np.ndarray):
        raw_img = img
    else:
        raise TypeError()

    if align_template is None:
        aligned_img = raw_img
    else:
        aligned_img = align_fft(raw_img, align_template)

    cropped_img = crop(aligned_img, boundaries)
    binarized_img = preprocess(cropped_img)
    chars = split(binarized_img)

    # this gets an array of ints
    predictions = clf.predict(np.array([c.reshape(-1) for c in chars]))
    # make into a string
    prediction = ''.join([str(p) for p in predictions])
    return prediction, binarized_img, chars
