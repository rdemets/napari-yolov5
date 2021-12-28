# YOLOv5 by Ultralytics, GPL-3.0 license


"""
This module is an example of a barebones function plugin for napari

It implements the ``napari_experimental_provide_function`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.

"""
from typing import TYPE_CHECKING

from enum import Enum
import numpy as np
from napari_plugin_engine import napari_hook_implementation

if TYPE_CHECKING:
    import napari


@napari_hook_implementation
def run():
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    #opt = parse_opt(True)
    #for k, v in kwargs.items():
    #    setattr(opt, k, v)
    #main(opt)
    print("Works")

#opt = parse_opt(False)
#main(opt)
