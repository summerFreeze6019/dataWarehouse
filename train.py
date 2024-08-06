import yaml
import os
import re
import time
import numpy as np
import matplotlib.pyplot as pp
import glob
import cPickle as pickle

from skimage import io
from sklearn import cross_validation, svm

from lib.learning import predict
from lib.image import split, crop

def collect_dataset():
    with(open('training/training_set.yml')) as f:
        labels = yaml.load(f)

    X, y = [], []
    
    for fn, answer in labels.iteritems():
        path = os.path.join('training', fn)
        chars = split(io.imread(path))
        
        print 'fn', fn

        assert len(chars) == len(answer)
        for char, label in zip(chars, answer):
            X.append(char.reshape(-1)) # flatten
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)
    
    print 'COLLECTED DATASET'
    
    return X, y


def train(X, y):
    clf = svm.LinearSVC()
    # run 5-fold cross validaton to assess the accuracy of the classfier
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)        
    print 'CLASSIFIER SCORES -- 5 FOLD CROSS VALIDATION'
    print '%s, mean=%s' % (scores, np.mean(scores))

    clf.fit(X, y)
    
    return clf


def interactive_benchmark(clf, raw_images_glob, save_to):
    with(open('settings.yml')) as f:
        settings = yaml.load(f)

    # interactive plotting
    pp.ion()
    pp.figure()
    pp.gray()
    
    if not os.path.exists(save_to):
        print "CREATING DIRECTORY", save_to
        os.makedirs(save_to)
    
    instructions = '[Enter] to accept, [q] to quit, or key the correct answer'
    print instructions
    
    lines = []
    
    raw_images_fn = glob.glob(raw_images_glob)
    training_images = set(yaml.load(open('training/training_set.yml')).keys())
    
    for fn in raw_images_fn:
        if os.path.basename(fn) in training_images:
            print 'Already trained on %s. Skipping' % fn
            continue
        
        prediction, binarized_img, chars = predict(clf, img=fn,
            boundaries=settings['crop'][0])
        
        # plot the binarize_img across the top panel
        pp.clf()
        pp.subplot(2,1,1)
        pp.imshow(binarized_img)
        
        # plot the 6 chars (separated) across the bottom panel
        for i in range(6):
            pp.subplot(2,6,7+i)
            pp.imshow(chars[i])
            
        title = ' '.join(prediction[:3]) + '  ' + ' '.join(prediction[3:])
        pp.suptitle(title, fontsize='xx-large')
        pp.draw()
        
        ask_again = True
        while ask_again:
            response = raw_input('[Enter/q/key]: ')
            ask_again = False  # default
            
            if response == '':
                line = '%s: "%s"' % (os.path.basename(fn), prediction)
                print line
                lines.append(line)
                io.imsave(os.path.join(save_to, os.path.basename(fn)),
                    1.0*binarized_img)

            elif re.match('\d{6}', response):
                line = '%s: "%s"' % (os.path.basename(fn), response)
                print 'CORRECTED', line
                lines.append(line)
                io.imsave(os.path.join(save_to, os.path.basename(fn)),
                    1.0*binarized_img)

            elif response == 'q':
                print 'Quitting...\n'
                print ('Instructions: Move the files from %s into the training '
                       'directory, and add the lines below to the file'
                       'training_set.yml:\n' % save_to)
                print os.linesep.join(lines)
                return

            else:
                print instructions
                ask_again = True
    
def main():
    X, y = collect_dataset()
    clf = train(X, y)
    
    
    clf_fn = 'classifier.pickle'
    print 'Saving classifier to disk: %s' % clf_fn
    with open(clf_fn, 'wb') as f:
        pickle.dump(clf, f)
    
    interactive_benchmark(clf, raw_images_glob='raw_images/*.png',
        save_to='interactive_training')
        
    
if __name__ == '__main__':
    main()