# napari-yolov5

[![License](https://img.shields.io/pypi/l/napari-yolov5.svg?color=green)](https://github.com/rdemets/napari-yolov5/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-yolov5.svg?color=green)](https://pypi.org/project/napari-yolov5)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-yolov5.svg?color=green)](https://python.org)
[![tests](https://github.com/rdemets/napari-yolov5/workflows/tests/badge.svg)](https://github.com/rdemets/napari-yolov5/actions)
[![codecov](https://codecov.io/gh/rdemets/napari-yolov5/branch/main/graph/badge.svg)](https://codecov.io/gh/rdemets/napari-yolov5)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-yolov5)](https://napari-hub.org/plugins/napari-yolov5)

Plugin adapted from Ultralytics to bring YOLOv5 into Napari. 

Training and detection can be done using the GUI. Training dataset must be prepared prior to using this plugin. Further development will allow users to use Napari to prepare the dataset. Follow instructions stated on [Ultralytics Github](https://github.com/ultralytics/yolov5) to prepare the dataset.

The plugin includes 3 pre-trained networks that are able to identify mitosis stages or apoptosis on soSPIM images. More details can be found on the [pre-print](https://www.biorxiv.org/content/10.1101/2021.03.26.437121v1.full).

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

First install conda and create an environment for the plugin
```
conda create --prefix env-napari-yolov5 python=3.9
conda activate env-napari-yolov5
```
You can install `napari-yolov5` and `napari` via [pip]:

    pip install napari-yolov5 
    pip install napari[all]

For GPU support :
```
pip uninstall torch
pip install torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

First select if you would like to train a new network or detect objects.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/1.jpg?raw=true)


***For `Training` :***

Data preparation should be done following [Ultralytics'](https://github.com/ultralytics/yolov5) instructions.

Select the size of the network, the number of epochs, the number of images per batch to load on the GPU, the size of the images (must be a stride of 32), and the name of the network.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/2.jpg?raw=true)

An example of the YAML config file is provided in `src/napari_yolov5/resources` folder.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/3.jpg?raw=true)


Progress can be seen on the Terminal. The viewer will switch to `Detection` mode automatically when the network is finished being trained.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/4.jpg?raw=true)


***For `Detection` :***

It is possible to perform the detection on a single layer chosen in the list, all the layers opened, or by giving a folder path. For folder detection, all the images will be loaded as a single stack.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/5.jpg?raw=true)

Nucleus size of the prediction layer has te be filled to resize the image to the training dataset. Nucleus size of the training dataset will be asked in case of a custom network.

Confidence threshold defines the minimum value for a detected object to be considered positive. 
iou nms threshold (intersection-over-union non-max-suppression) defines the overlapping area of two boxes as a single object. Only the box with the maximum confidence is kept.
Progress can be seen on the Terminal.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/6.jpg?raw=true)

Few options allow for modification on how the boxes are being displayed (default : box + class + confidence score ; box + class ; box only) and if the box coordinates and the image overlay will be exported.
Post-processing option will perform a simple 3D assignment based on 3D connected component analysis. A median filter (1x1x3 XYZ) is applied prior to the assignment. 
The centroid of each object is then saved into a new point layer as a 3D point with a random color for each class. 

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/7.jpg?raw=true)

The localisation of each centroid is saved and the path is shown in the Terminal at the end of the detection.

![alt text](https://github.com/rdemets/napari-yolov5/blob/main/src/napari_yolov5/resources/Readme/8.jpg?raw=true)


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"napari-yolov5" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
