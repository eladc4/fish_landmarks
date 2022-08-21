import numpy as np
import matplotlib

from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return f'x={x:.01f}, y={y:.01f}, z={tuple(z.data)}'


def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y
    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print('data coords %f %f' % (event.xdata, event.ydata))


def on_click(event):
    if event.button is MouseButton.LEFT:
        print('disconnecting callback')
        plt.disconnect(binding_id)


fig, ax = plt.subplots()
im = ax.imshow(mpimg.imread("C:\project\dataset_2d\imgs\img_000000.jpg"), interpolation='none')
ax.format_coord = Formatter(im)
binding_id = plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)
plt.show()


a=1
