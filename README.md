# BCNNs

This is Chainer implementation for Bayesian Convolutional Neural Networks.

(Keras re-impremitation is also available: [keras_bayesian_unet](https://github.com/yuta-hi/keras_bayesian_unet))

In this project, we assume the following two scenarios, especially for medical imaging.
1. Two-dimensional segmentation / regression with the 2D U-Net.  (e.g., 2D x-ray, laparoscopic images, and CT slices)
3. Three-dimensional segmentation / regression with the 3D U-Net.  (e.g., 3D CT volumes)

<img src='figs/bayesian_unet.gif' width='600px'>

This is a part of following works.
```
@article{hiasa2019automated,
  title={Automated Muscle Segmentation from Clinical CT using Bayesian U-Net for Personalized Musculoskeletal Modeling},
  author={Hiasa, Yuta and Otake, Yoshito and Takao, Masaki and Ogawa, Takeshi and Sugano, Nobuhiko and Sato, Yoshinobu},
  journal={IEEE Transactions on Medical Imaging},
  year={2019 (in press)},
  doi={10.1109/TMI.2019.2940555},
}

@article{sakamoto2019bayesian,
  title={Bayesian Segmentation of Hip and Thigh Muscles in Metal Artifact-Contaminated CT using Convolutional Neural Network-Enhanced Normalized Metal Artifact Reduction},
  author={Sakamoto, Mitsuki and Hiasa, Yuta and Otake, Yoshito and Takao, Masaki and Suzuki, Yuki and Sugano, Nobuhiko and Sato, Yoshinobu},
  journal={Journal of Signal Processing Systems},
  year={2019 (in press)}
  doi={10.1007/s11265-019-01507-z},
}

@inproceedings{hiasa2018surgical,
  title={Surgical tools segmentation in laparoscopic images using convolutional neural networks with uncertainty estimation and semi-supervised learning},
  author={Hiasa, Y and Otake, Y and Nakatani, S and Harada, H and Kanaji, S and Kakeji, Y and Sato, Yoshinobu},
  booktitle={Proc. International Conference of Computer Assisted Radiology and Surgery},
  pages={14--15},
  year={2018}
}
```

## Requirements
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Chainer 6.5

## Getting started
### Installation
- Install Chainer and dependencies from https://chainer.org/
- For other requirements, see [requirements.txt](requirements.txt).
- Install from this repository
```bash
git clone https://github.com/yuta-hi/bayesian_unet
cd bayesian_unet
python setup.py install
```

## Examples

The data set we used are medical images and it is difficult to share due to ethical issues.
Thus, we prepared the following examples using synthetic or public data set.

### Curve regression

Approximation of the function $y = x\sin x$, which reproduces the previous work [Y. Gal, et al.].
Training and validation data sets were sampled within the range of [-5, 5].
On the other hand, test data set was sampled within the range of [-10, 10].
The predicted variance on test data set showed high values for the unseen samples, but not for the distribution of training data set.

```bash
python examples/curve_regression/train_and_test_epistemic.py
python examples/curve_regression/train_and_test_epistemic.py --test_on_test
```
<img src='figs/curve_regression_1.png' width='600px'>

```bash
python examples/curve_regression/train_and_test_epistemic_aleatoric.py
python examples/curve_regression/train_and_test_epistemic_aleatoric.py --test_on_test
```
<img src='figs/curve_regression_2.png' width='600px'>

### MNIST classification

Ten digits classification.
A subset of samples was used for the training data set.
In the default setting, 1,000 samples are used for training and 1,000 samples are used for validation.
The distribution of predicted variance for correct and wrong predictions on the test data set (10,000 samples) were visualized.

```bash
python examples/mnist_classification/train_and_test_epistemic.py
python examples/mnist_classification/train_and_test_epistemic.py --test_on_test
```
<img src='figs/mnist_classification.png' width='600px'>

### EndoVis segmentation

Segmentation of surgical instruments from laparoscopic images.
Data set is downloaded from https://endovissub-instrument.grand-challenge.org/ .
Training and test data sets consist 160 and 140 images, respectively.

```bash
python examples/miccai_endovis_segmentation/preprocess.py # download the dataset and convert label format
python examples/miccai_endovis_segmentation/train_and_test_epistemic.py
python examples/miccai_endovis_segmentation/train_and_test_epistemic.py --test_on_test
```

<img src='figs/miccai_endovis_segmentation_samples.png' width='600px'>
<img src='figs/miccai_endovis_segmentation.png' width='400px'><br>

### Image synthesis with adversarial training

Aerial-to-Map translation. This example focuses on how the adversarial training affects uncertainty behavior.
This is mainly followed the previous work [P. Isola, et al.].
In this example, the generator is replaced to Bayesian U-Net for uncertainty estimates.
And, spectral normalization [T. Miyato et al.] is applied to the patch discriminator for stabilizing the optimization.

```bash
cd examples/map_synthesis
python preprocess.py # download and normalize the dataset
python train_and_test_baseline.py --out logs/baseline
python train_and_test_pix2pix.py --out logs/pix2pix
```
Note that this is under construction.

<img src='figs/map_synthesis_samples.png' width='600px'><br>

### Heatmap regression

On going.

<img src='figs/miccai_heatmap_regression.png' width='600px'><br>

## Usage

Please follow the description to define these objects.
- [datasets](#datasets)
- [data augmentor](#augmentor)
- [data normalizer](#normalizer)
- [model](#model)
- [visualizer](#visualizer)
- [validator](#validator)
- [inferencer](#inferencer)
- (optional) [singularity image](#singularity)

### <a name="datasets"></a>Setup datasets
You can define your own dataset like below.
PNG, JPG, BMP and meta image format (MHD, MHA) are supported.

- [case #1] 2D images
```python
from chainer_bcnn.datasets import ImageDataset

data_root = './data'
patients = ['ID0', 'ID1', 'ID2'] # NOTE: 3 patients
class_list = ['background', 'liver', 'tumor']

augmentor = None # NOTE: please set if you have..
normalizer = None # NOTE: please set if you have..

dtypes = OrderedDict({
    'image': np.float32,
    'label': np.int32, # NOTE: if categorical label
#    'mask': np.uint8, # NOTE: please set if you have..
})

filenames = OrderedDict({
    'image': '{root}/{patient}/*_image.mhd',
    'label': '{root}/{patient}/*_label.mhd',
#    'mask' : '{root}/{patient}/*_mask.mhd', # NOTE: please set if you have..
})

dataset = ImageDataset(data_root, patients, classes=class_list,
                dtypes=dtypes, filenames=filenames, augmentor=augmentor, normalizer=normalizer)

```
- [case #2] 3D volumes
```python
from chainer_bcnn.datasets import VolumeDataset

...

dataset = VolumeDataset(data_root, patients, classes=class_list,
                dtypes=dtypes, filenames=filenames, augmentor=augmentor, normalizer=normalizer)
```

- [case #3] Custom dataset
```python
from chainer_bcnn.datasets import BaseDataset

class CustomDataset(BaseDataset):

    ...

    raise NotImplementedError()
```


### <a name="augmentor"></a>Setup data augmentor
You can use the data augmentor based on geometric transformation, which has stochastic behavior.

```python
from chainer_bcnn.data.augmentor import DataAugmentor
from chainer_bcnn.data.augmentor import Crop2D, Flip2D, Affine2D
from chainer_bcnn.data.augmentor import Crop3D, Flip3D, Affine3D

augmentor = DataAugmentor()
augmentor.add(Crop2D(size=(300,400)))
augmentor.add(Flip2D(axis=1))
augmentor.add(Affine2D(rotation=15.,
                       translate=(10.,10.),
                       shear=0.25,
                       zoom=(0.8, 1.2),
                       keep_aspect_ratio=True,
                       fill_mode=('nearest', 'constant'),
                       cval=(0.,0.),
                       interp_order=(3,0)))

augmentor.summary('augment.json')
```

### <a name="normalizer"></a>Setup data normalizer
You can use the data normalizer based on intensity transformation.

```python

from chainer_bcnn.data.normalizer import Normalizer
from chainer_bcnn.data.normalizer import Clip2D, Subtract2D, Divide2D, Quantize2D
from chainer_bcnn.data.normalizer import Clip3D, Subtract3D, Divide3D, Quantize3D

normalizer = Normalizer()
normalizer.add(Clip2D((-150, 350)))
normalizer.add(Quantize2D(8))
normalizer.add(Subtract2D(0.5))
normalizer.add(Divide2D(1./255.))

normalizer.summary('norm.json')
```

### <a name="model"></a>Setup model

<details><summary> To see the computational graph of UNet, please ```click here```. </summary><div>
<img src='figs/unet.png' width='1920px'>
</div></details>


- [case #1] Segmentation

```python
from chainer_bcnn.models import BayesianUNet
from chainer_bcnn.links import Classifier
predictor = BayesianUNet(ndim=2,
                         out_channels=3,
                         nlayer=5,
                         nfilter=32)

lossfun = partial(softmax_cross_entropy,
                    normalize=False, class_weight=class_weight)

model = Classifier(predictor,
                    lossfun=lossfun)

```

- [case #2] Regression
```python
from chainer_bcnn.links import Regressor
from chainer_bcnn.functions.loss import sigmoid_soft_cross_entropy
from chainer.functions import mean_squared_error
...

lossfun = mean_squared_error
# lossfun = sigmoid_soft_cross_entropy # NOTE: if you want..

model = Regressor(predictor,
                  lossfun=lossfun)
```

- [case #3] Other problems (e.g., multi-task)
```python
from chainer_bcnn.models import UNetBase

class MultiTaskUNet(UNetBase):

    def __init__(self,
                 ndim,
                 foo, # TODO
                 bar, # TODO
                 nfilter=32,
                 nlayer=5,
                 conv_param=_default_conv_param,
                 pool_param=_default_pool_param,
                 upconv_param=_default_upconv_param,
                 norm_param=_default_norm_param,
                 activation_param=_default_activation_param,
                 dropout_param=_default_dropout_param,
                 residual=False,
                ):
        super(UNet, self).__init__(
                                ndim,
                                nfilter,
                                nlayer,
                                conv_param,
                                pool_param,
                                upconv_param,
                                norm_param,
                                activation_param,
                                dropout_param,
                                residual,)
        self._foo = foo
        self._bar = bar

        with self.init_scope():
            pass # TODO: foo, bar

    def forward(self, x):
        h = super().forward(x)
        # TODO: foo, bar
        raise NotImplementedError('foo is bar..')
```

### <a name="visualizer"></a>Setup visualizer

- [case #1] 2D segmentation
```python

from chainer_bcnn.visualizer import ImageVisualizer

transforms = {
    'x': lambda x: x,
    'y': lambda x: np.argmax(x, axis=0),
    't': lambda x: x,
}

_cmap = np.array([
    [0,0,0],  # NOTE: background (black)
    [1,0,0],  #       liver (red)
    [0,1,0]]) #       tumor (green)

cmaps = {
    'x': None,
    'y': _cmap,
    't': _cmap,
}

clims = {
    'x': (0., 255.),
    'y': None,
    't': None,
}

visualizer = ImageVisualizer(transforms=transforms,
                                cmaps=cmaps,
                                clims=clims)
```

- [case #2] 2D regression
```python
from chainer_bcnn.visualizer import ImageVisualizer
import matplotlib.pyplot as plt

def alpha_blend(heatmaps, cmap='jet'):
    assert heatmaps.ndim == 3

    ch, w, h = heatmaps.shape
    ret = np.zeros((3, w, h))
    mapper = plt.get_cmap(cmap, ch)

    for i in range(ch):
        color = np.ones((3, w, h)) \
                    * np.asarray(mapper(i)[:3]).reshape(-1,1,1)
        ret += (color * heatmaps[i])

    return ret

transforms = {
    'x': None,
    'y': lambda x: alpha_blend(F.sigmoid(x).data),
    't': lambda x: alpha_blend(x),
}

clims = {
    'x': (0., 255.),
    'y': (0., 1.),
    't': (0., 1.),
}

cmaps = None

visualizer = ImageVisualizer(transforms=transforms,
                                cmaps=cmaps,
                                clims=clims)
```
To visualize 3D volumes, you can pass the volume renderer to the `transforms` as described above.

### <a name="validator"></a>Setup validator
```python
from chainer_bcnn.extensions import Validator

...

valid_file = 'iter_{.updater.iteration:08}.png'

n_vis = 20 # NOTE: number of samples for visualization

trainer.extend(Validator(valid_iter, model, valid_file,
                            visualizer=visualizer, n_vis=n_vis,
                            device=gpu_id))
```

### <a name="inferencer"></a>Setup inferencer

- [case #1] Segmentation / Classification

```python
from chainer_bcnn.links import MCSampler
from chainer_bcnn.inference import Inferencer
import chainer.functions as F

mc_iteration = 50

model = MCSampler(predictor, # NOTE: e.g., BayesianUNet
                    mc_iteration=mc_iteration,
                    activation=partial(F.softmax, axis=1),
                    reduce_mean=partial(F.argmax, axis=1),
                    reduce_var=partial(F.mean, axis=1))

infer = Inferencer(test_iter, model, device=gpu_id)

estimated_labels, predicted_variances = infer.run()
```

### <a name="singularity"></a>(Optional) Setup singularity image
```bash
cd recipe
make all
```

## Related repositories
- [keras_bayesian_unet](https://github.com/yuta-hi/keras_bayesian_unet)

- [anatomy_viewer](https://github.com/yuta-hi/anatomy-viewer)
<img src='https://github.com/yuta-hi/anatomy-viewer/blob/master/figs/demo.gif' width='600px'>

