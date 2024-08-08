from lib.camera import capture_image
from lib.image import preprocess
import yaml
import glob
import random

from matplotlib.patches import Rectangle
from lib.rectangle import DraggableRectangle

from skimage import io
import matplotlib.pyplot as pp

with open('settings.yml') as f:
    settings = yaml.load(f)
    box_xy = (settings['crop'][0]['box_x'], settings['crop'][0]['box_y'])
    box_width = settings['crop'][0]['box_w']
    box_height = settings['crop'][0]['box_h']
    template = settings['template_img']

img = io.imread(template)
#img = capture_image(tmp_fn='template.png')


fig, axes = pp.subplots(nrows=2, ncols=1, figsize=(8, 5))
pp.gray()

zoom_horizontal = (250, 550)
zoom_vertial = (450, 150)
# box_xy = (370, 342)
# box_width = 122
# box_height = 37


axes[0].imshow(img)
axes[0].set_xlim(*zoom_horizontal)
axes[0].set_ylim(*zoom_vertial)

zoomrect = DraggableRectangle(Rectangle(box_xy, height=box_height,
                                    width=box_width, alpha=0.25), ax=axes[0])
zoomrect.connect()

def onpress(event, force=False):
    if force or event.inaxes == axes[0]:
        x0, y0 = zoomrect.rect.xy
        w0, h0 = zoomrect.rect.get_width(), zoomrect.rect.get_height()
        img2 = preprocess(img[y0:y0+h0, x0:x0+w0, :])
        axes[1].imshow(img2)
        axes[1].xaxis.grid(color='gray', which='minor', linestyle='dashed')
        
        print '\n\n'
        for k, v in dict(box_x=x0, box_y=y0, box_w=w0, box_h=h0).iteritems():
            print '%s: %s' % (k, v)

# trigger at beginning
onpress(None, force=True)

axes[0].figure.canvas.mpl_connect('button_press_event', onpress)


pp.show()


