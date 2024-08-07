import yaml
import cPickle as pickle
import matplotlib.pyplot as pp

from lib.camera import capture_image
from lib.image import preprocess, split
from lib.learning import predict


def main(which, debug=True):
    with(open('settings.yml')) as f:
        settings = yaml.load(f)

    clf = pickle.load(open('classifier.pickle'))
    
    img = capture_image()
    
    prediction, binarized_img, chars = predict(clf, img=img,
         boundaries=settings['crop'][which])
     
    
    if debug:
        pp.gray()        
        pp.subplot(2,1,1)
        pp.title('Prediction: %s' % prediction, fontsize='xx-large')
        pp.imshow(binarized_img)
        for i in range(6):
            pp.subplot(2,6,7+i)
            pp.imshow(chars[i])
        pp.show()

    print prediction

if __name__ == '__main__':
    main(0)
    main(1)

