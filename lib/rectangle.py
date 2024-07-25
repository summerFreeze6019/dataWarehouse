import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DraggableRectangle:
    lock = None  # only one can be animated at a time
    def __init__(self, rect, border_tol=.15, allow_resize=True, ax=None,
            allow_y_resize=True):
        self.rect = rect
        self.border_tol = border_tol
        self.allow_resize = allow_resize
        self.press = None
        self.background = None
        self.ax = ax
        self.allow_y_resize = allow_y_resize
        
        self.ax.add_patch(self.rect)        

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        '''on button press we will see if the mouse is over us and store some 
        data'''
        if event.inaxes != self.rect.axes: return
        if DraggableRectangle.lock is not None: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        #print 'event contains', self.rect.xy
        x0, y0 = self.rect.xy
        w0, h0 = self.rect.get_width(), self.rect.get_height()
        aspect_ratio = np.true_divide(w0, h0)
        self.press = x0, y0, w0, h0, aspect_ratio, event.xdata, event.ydata
        DraggableRectangle.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes: return
        x0, y0, w0, h0, aspect_ratio, xpress, ypress = self.press
        self.dx = event.xdata - xpress
        self.dy = event.ydata - ypress
        self.update_rect()

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        #self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    def update_rect(self):
        x0, y0, w0, h0, aspect_ratio, xpress, ypress = self.press
        dx, dy = self.dx, self.dy
        bt = self.border_tol
        if (not self.allow_resize or
            (abs(x0+np.true_divide(w0,2)-xpress)<np.true_divide(w0,2)-bt*w0 and
             abs(y0+np.true_divide(h0,2)-ypress)<np.true_divide(h0,2)-bt*h0)):
            self.rect.set_x(x0+dx)
            if self.allow_y_resize:
                self.rect.set_y(y0+dy)
        elif abs(x0-xpress)<bt*w0:
            self.rect.set_x(x0+dx)
            self.rect.set_width(w0-dx)
        elif abs(x0+w0-xpress)<bt*w0:
            self.rect.set_width(w0+dx)
        elif abs(y0-ypress)<bt*h0 and self.allow_y_resize:
            self.rect.set_y(y0+dy)
            self.rect.set_height(h0-dy)
        elif abs(y0+h0-ypress)<bt*h0 and self.allow_y_resize:
            self.rect.set_height(h0+dy)

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 6)
    drs = []

    for i in range(2):
        rect = Rectangle((i,0), height=1, width=0.5, alpha=0.5)
        dr = DraggableRectangle(rect, ax=ax)
        dr.connect()
        drs.append(dr)

    plt.show()