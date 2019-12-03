import numpy as np
from chainer_bcnn.data.augmentor import DataAugmentor, Crop3D, Flip3D, Affine3D
from chainer_bcnn.data import load_image, save_image
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='image.mhd')
    parser.add_argument('--label', type=str, default='label.mhd')
    args = parser.parse_args()

    augmentor = DataAugmentor()
    augmentor.add(Crop3D(size=(100, 200, 300)))
    augmentor.add(Flip3D(axis=2))
    augmentor.add(Affine3D(
        rotation=(15., 15., 15.),
        translate=(10., 10., 10.),
        shear=(np.pi / 8, np.pi / 8, np.pi / 8),
        zoom=(0.8, 1.2),
        keep_aspect_ratio=True,
        fill_mode=('constant', 'constant'),
        cval=(-3000., -1.),
        interp_order=(0, 0)))

    augmentor.summary('augment.json')

    x_in, spacing = load_image(args.image)
    x_in = np.expand_dims(x_in, axis=0)  # add channel-axis
    x_in = x_in.astype(np.float32)

    y_in, _ = load_image(args.label)
    y_in = y_in.astype(np.float32)

    tic = time.time()
    x_out, y_out = augmentor.apply(x_in, y_in)
    print('time: %f [sec]' % (time.time()-tic))

    save_image('x_out.mha', x_out[0, :, :, :], spacing)
    save_image('y_out.mha', y_out, spacing)


if __name__ == '__main__':
    main()
