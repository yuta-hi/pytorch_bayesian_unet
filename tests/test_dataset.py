from pytorch_bcnn.data import load_image, save_image
from pytorch_bcnn.datasets import ImageDataset, VolumeDataset
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse

patient_list = ['k1565', 'k1585']

class_list = ['background', 'pelvis', 'femur', 'adductor_muscles',
              'biceps_femoris_muscle', 'gluteus_maximus_muscle',
              'gluteus_medius_muscle', 'gluteus_minimus_muscle',
              'gracilis_muscle', 'iliacus_muscle', 'obturator_externus_muscle',
              'obturator_internus_muscle', 'pectineus_muscle',
              'piriformis_muscle', 'psoas_major_muscle',
              'rectus_femoris_muscle', 'sartorius_muscle',
              'semimembranosus_muscle', 'semitendinosus_muscle',
              'tensor_fasciae_latae_muscle',
              'vastus_lateralis_muscle_and_vastus_intermedius_muscle',
              'vastus_medialis_muscle', 'sacrum']

dtypes = OrderedDict({
    'image': np.float32,
    'label': np.int64,
    'mask' : np.uint8,
})

mask_cvals = OrderedDict({
    'image': -3000,
    'label': 0,
})


def test_2d(root):

    filenames = OrderedDict({
        'image': '{root}/{patient}/slice/*image*.mhd',
        'label': '{root}/{patient}/slice/*mask*.mhd',
        'mask' : '{root}/{patient}/slice/*skin*.mhd',
    })

    dataset = ImageDataset(root,
                    patients=patient_list, classes=class_list,
                    dtypes=dtypes, filenames=filenames,
                    mask_cvals=mask_cvals)

    print('# dataset:', len(dataset))
    print('# classes:', dataset.n_classes)

    sample = dataset.get_example(0)

    print(sample[0].shape)
    print(sample[1].shape)

    plt.subplot(1, 2, 1)
    plt.imshow(sample[0][0, :, :], cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(sample[1][:, :])
    plt.colorbar()
    plt.show()


def test_3d(root):

    filenames = OrderedDict({
        'image': '{root}/{patient}/*image*.mhd',
        'label': '{root}/{patient}/*mask*.mhd',
        'mask' : '{root}/{patient}/*skin*.mhd',
    })

    dataset = VolumeDataset(root,
                    patients=patient_list, classes=class_list,
                    dtypes=dtypes, filenames=filenames,
                    mask_cvals=mask_cvals)

    print('# dataset:', len(dataset))
    print('# classes:', dataset.n_classes)

    sample = dataset.get_example(0)

    print(sample[0].shape)
    print(sample[1].shape)

    plt.subplot(1, 2, 1)
    plt.imshow(sample[0][0, :, :, 100], cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(sample[1][:, :, 100])
    plt.colorbar()
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    args = parser.parse_args()

    test_2d(args.root)
    test_3d(args.root)

if __name__ == "__main__":
    main()
