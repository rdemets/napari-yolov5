

from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import random
import numpy as np
import qtpy
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari 
from napari import Viewer
from napari.layers import Image, Shapes



from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from magicgui import widgets as mw
from magicgui.events import Event
from magicgui.application import use_app

import functools
import time
import numpy as np

from pathlib import Path
from warnings import warn
import os
import skimage.io as skio
import skimage as skk


import napari
from napari.qt.threading import thread_worker, create_worker
from napari.utils.colormaps import label_colormap
from typing import List
from enum import Enum

import torch


def abspath(root, relpath):
    from pathlib import Path
    root = Path(root)
    if root.is_dir():
        path = root/relpath
    else:
        path = root.parent/relpath
    return str(path.absolute())




#print(__file__)
logo = abspath(__file__, 'resources/SIMBA.png')




def widget_wrapper():

    @magicgui(
        label_head=dict(widget_type='Label', label=f'<center><img src="{logo}"><h2>SiMBA - NUS</h2></center>'),
        layout='vertical',
        main_task=dict(widget_type='RadioButtons', label='What do you want to do today ?', orientation='horizontal',
                       choices=['Training', 'Detection']),
        training_path=dict(widget_type='FileEdit', visible=False, label='config file (.yaml): ',
                           tooltip='path for YAML config file, find template bcc.yaml in the repository'),
        network_size=dict(widget_type='RadioButtons', visible=False, label='Network size', orientation='horizontal',
                          choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', ], value = 'yolov5s'),
        nb_epochs=dict(widget_type='LineEdit', visible=False, label='Epochs', value=100,
                       tooltip='number of cycles of training'),
        batch_size=dict(widget_type='LineEdit', visible=False, label='Batch Size', value=16,
                        tooltip='number of images per batch to send to the GPU'),
        img_resize=dict(widget_type='LineEdit', visible=False, label='Image Resize (multiple of 32)', value=1024,
                        tooltip='size of the image after resize'),
        project_name=dict(widget_type='LineEdit', visible=False, label='Project Name', value='exp',
                          tooltip='reference name of the project for future use'),
        #run_training_button=dict(widget_type='PushButton', visible=False, text='Train'),
        image_type=dict(widget_type='RadioButtons', visible=False, label='image layer', orientation='horizontal',
                        choices=['One Layer', 'All Layers', 'Folder'], value='One Layer',
                        tooltip='which image is going to be predicted'),
        # folder_detect= dict(widget_type=qtpy.QtWidgets.QFileDialog.getExistingDirectory(None,'Select Folder'), visible=False),
        folder_detect=dict(widget_type='LineEdit', label='Folder to detect: ', visible=False,
                           tooltip='Folder with images to detect'),
        model_type=dict(widget_type='ComboBox', visible=False, label='model name',
                        choices=['soSPIM Ki+ 1class', 'soSPIM Ki+ 4classes', 'soSPIM H2B 2classes', 'custom'],
                        value='soSPIM H2B 2classes', tooltip='please indicate which pre-trained model to use model'),
        predict_nucleus_size = dict(widget_type='LineEdit', visible=False, label='Nucleus Size in Prediction Set (in px)', value='100',
                         tooltip='approximate nucleus size to match dataset used for training'),
        custom_model=dict(widget_type='FileEdit', label='custom model path (.pt): ', visible=False,
                          tooltip='if model type is custom, specify folder path to it here'),
        training_nucleus_size=dict(widget_type='LineEdit', visible=False, label='Nucleus Size in Training Set (in px)',
                                  value='',
                                  tooltip='reference name of the project for future use'),
        save_detect=dict(widget_type='LineEdit', visible=False, label='Project Name', value='exp',
                         tooltip='reference name of the project for future use'),
        conf_threshold=dict(widget_type='FloatSlider', visible=False, name='conf_threshold', value=0.25, min=0, max=1.0,
                            step=0.05, tooltip='confidence threshold (set lower to get more events detected)'),
        iou_nms_threshold=dict(widget_type='FloatSlider', visible=False, name='iou_nms_threshold', value=0.45, min=0.0,
                               max=1.0, step=0.05, tooltip='threshold for NMS of overlapping objects'),
        hide_labels=dict(widget_type='CheckBox', visible=False, text='Box only', value=False,
                         tooltip='hide label text for each box'),
        hide_conf=dict(widget_type='CheckBox', visible=False, text='Hide confidence score', value=False,
                       tooltip='hide confidence text for each box'),
        box_line_thickness=dict(widget_type='Slider', visible=False, name='box_line_thickness', value=2, min=1, max=5,
                            step=1, tooltip='box line thickness'),
        title_post=dict(widget_type='Label', visible=False,
                        label='<h3><center><b>Post Processing:</b></center><br></h3>'),
        assignment_3d=dict(widget_type='CheckBox', visible=False, text='3D Assignment (Connected Component Analysis)', value=True,
                       tooltip='Replace boxes by centroid point'),
        change_save_dir=dict(widget_type='RadioButtons', visible=False, label='Choose export directory', orientation='horizontal',
                        choices=['Automatic','Manuel'],value = 'Automatic', tooltip='Folder to export detection'),
        folder_to_save=dict(widget_type='LineEdit', label='Export folder: ', visible=False,
                           tooltip='Folder for prediction to be saved'),
        # title_output = dict(widget_type='Label', visible=False, label='<h3><center><b>Outputs:</b></center><br></h3>'),
        save_txt=dict(widget_type='CheckBox', visible=False, text='Export detection as txt', value=True,
                      tooltip='save raw coordinates as txt'),
        save_img=dict(widget_type='CheckBox', visible=False, text='Export overlay as tif', value=True,
                      tooltip='save images and 3D assignment'),
        # save_button  = dict(widget_type='PushButton', visible=False, text='save outputs'),
        #run_detection_button=dict(widget_type='PushButton', visible=False, text='detect events'),
        call_button=True,
    )
    def widget(viewer: napari.viewer.Viewer,
               label_head,
               main_task,
               training_path,
               network_size,
               nb_epochs,
               batch_size,
               img_resize,
               project_name,
               #run_training_button,
               image_type,
               selected_layer: Image,
               folder_detect,
               predict_nucleus_size,
               model_type,
               custom_model,
               training_nucleus_size,
               conf_threshold,
               iou_nms_threshold,
               hide_conf,
               hide_labels,
               save_txt,
               save_img,
               box_line_thickness,
               title_post,
               assignment_3d,
               change_save_dir,
               save_detect,
               folder_to_save,
               # title_output,
               #run_detection_button,
               # save_button,
               ) -> None:

        if main_task == 'Training':
            from ._function import run_training
            JustOneTime = False
            path_yaml = training_path
            run_training(path_yaml, widget, viewer)


        elif main_task == 'Detection':
            from ._function import run_detect
            # Choose model
            if model_type == 'soSPIM Ki+ 1class':
                widget.custom_model.value = abspath(__file__, 'runs/train/Ki67_1class/weights/best.pt')
                widget.predict_nucleus_size.visible = False
                widget.training_nucleus_size.value = 100
            elif model_type == 'soSPIM Ki+ 4classes':
                widget.custom_model.value = abspath(__file__, 'runs/train/Ki67_Multiclass/weights/best.pt')
                widget.training_nucleus_size.value = 100
            elif model_type == 'soSPIM H2B 2classes':
                widget.custom_model.value = abspath(__file__, 'runs/train/H2B_2classes/weights/best.pt')
                widget.training_nucleus_size.value = 100


            # Load data
            data = np.array([])
            if image_type == 'One Layer':
                #try:
                data = selected_layer.data
                data = skk.img_as_ubyte(data)

                name = selected_layer.name
                result, points = run_detect(widget, data, name)
                viewer.add_image(result, name='Pred_'+selected_layer.name, blending='additive')
                if points:
                    points = np.array(points)
                    print('Total number of objects :',points.shape[0])
                    _,_,_,number_classes = points.max(axis = 0)

                    for classes in range(1,number_classes+1):
                        points_to_add = points[points[:,3]==classes]
                        print('Number of objects for class '+str(classes)+' : '+str(points_to_add.shape[0]))
                        points_to_add = np.delete(points_to_add, -1,-1) # Remove class column
                        face_color = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
                        if points_to_add.shape[0]!=0:
                            viewer.add_points(points_to_add, name='3D_Assign_cl' + str(classes), blending='additive', face_color = face_color, size = [5,round(result.shape[1]/50),round(result.shape[1]/50)], n_dimensional = True)
                else:
                    print('No object was found')

            elif image_type == 'Folder':
                First_image = True
                for file in os.listdir(folder_detect):
                    if file.endswith(".tif"):
                        path = os.path.join(folder_detect, file)
                        stack = skio.imread(path)
                        stack = skk.img_as_ubyte(stack)
                        if len(stack.shape) < 3:
                            stack = np.expand_dims(stack, axis=0)
                        if First_image:
                            data = stack
                            First_image = False
                        else:
                            data = np.concatenate((data, stack), axis=0)
                name = os.path.basename(os.path.normpath(folder_detect))
                result,points = run_detect(widget, data, name)
                viewer.add_image(result, name='Prediction_' + name, blending='additive')
                if points:
                    points = np.array(points)
                    print('Total number of objects :',points.shape[0])
                    _,_,_,number_classes = points.max(axis = 0)

                    for classes in range(1,number_classes+1):
                        points_to_add = points[points[:,3]==classes]
                        print('Number of objects for class '+str(classes)+' : '+str(points_to_add.shape[0]))
                        points_to_add = np.delete(points_to_add, -1,-1) # Remove class column
                        face_color = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
                        if points_to_add.shape[0]!=0:
                            viewer.add_points(points_to_add, name='3D_Assign_cl' + str(classes), blending='additive', face_color = face_color, size = [5,round(result.shape[1]/50),round(result.shape[1]/50)], n_dimensional = True)
                else:
                    print('No object was found')
            else:
                for index in range(len(viewer.layers)):
                    data = viewer.layers[index].data
                    data = skk.img_as_ubyte(data)

                    name = viewer.layers[index].name
                    print('\n\n####################   Prediction on layer: '+name+'  ####################\n')
                    result,points = run_detect(widget, data, name)
                    viewer.add_image(result, name='Pred_' + selected_layer.name, blending='additive')
                    if points:
                        points = np.array(points)
                        print('Total number of objects :', points.shape[0])
                        _, _, _, number_classes = points.max(axis=0)

                        for classes in range(1, number_classes + 1):
                            points_to_add = points[points[:, 3] == classes]
                            print('Number of objects for class ' + str(classes) + ' : ' + str(points_to_add.shape[0]))
                            points_to_add = np.delete(points_to_add, -1, -1)  # Remove class column
                            face_color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                            if points_to_add.shape[0] != 0:
                                viewer.add_points(points_to_add, name='3D_Assign_cl' + str(classes) +'_'+selected_layer.name, blending='additive',
                                              face_color=face_color,
                                              size=[5, round(result.shape[1] / 50), round(result.shape[1] / 50)],
                                              n_dimensional=True)
                    else:
                        print('No object was found')
            print('\n\n############# Detection End #############\n\n')
            
    @widget.change_save_dir.changed.connect
    def _change_savedir_display(e:Any):
        if widget.change_save_dir.value == 'Manuel':
            if widget.folder_to_save.value:
                pass
            else:
                widget.folder_to_save.value = qtpy.QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder')
                widget.folder_to_save.visible = True
        elif widget.change_save_dir.value == 'Automatic':
            widget.folder_to_save.visible = False

    @widget.main_task.changed.connect
    def _change_main_display(e: Any):
        if widget.main_task.value == 'Training':
            widget.training_path.visible = True
            #widget.run_training_button.visible = True
            widget.network_size.visible = True
            widget.nb_epochs.visible = True
            widget.batch_size.visible = True
            widget.img_resize.visible = True
            widget.project_name.visible = True
            widget.image_type.visible = False
            widget.selected_layer.visible = False
            widget.folder_detect.visible = False
            widget.model_type.visible = False
            widget.predict_nucleus_size.visible = False
            widget.custom_model.visible = False
            widget.training_nucleus_size.visible = False
            widget.save_detect.visible = False
            widget.conf_threshold.visible = False
            widget.iou_nms_threshold.visible = False
            widget.hide_labels.visible = False
            widget.hide_conf.visible = False
            widget.box_line_thickness.visible = False
            widget.title_post.visible = False
            widget.assignment_3d.visible = False
            widget.change_save_dir.visible = False
            widget.folder_to_save.visible = False
            #widget.run_detection_button.visible = False
            # widget.title_output.visible = False
            widget.save_txt.visible = False
            widget.save_img.visible = False
            # widget.save_button.visible = False
        else:
            widget.training_path.visible = False
            #widget.run_training_button.visible = False
            widget.network_size.visible = False
            widget.nb_epochs.visible = False
            widget.batch_size.visible = False
            widget.img_resize.visible = False
            widget.project_name.visible = False
            widget.image_type.visible = True
            widget.selected_layer.visible = True
            widget.folder_detect.visible = False
            widget.predict_nucleus_size.visible = True
            widget.model_type.visible = True
            widget.custom_model.visible = False
            widget.save_detect.visible = True
            widget.conf_threshold.visible = True
            widget.iou_nms_threshold.visible = True
            widget.hide_labels.visible = True
            widget.hide_conf.visible = True
            widget.box_line_thickness.visible = True
            widget.title_post.visible = True
            widget.assignment_3d.visible = True
            widget.change_save_dir.visible = True
            widget.folder_to_save.visible = False          
            #widget.run_detection_button.visible = True
            # widget.title_output.visible = True
            widget.save_txt.visible = True
            widget.save_img.visible = True
            # widget.save_button.visible = True

    @widget.image_type.changed.connect
    def _change_display(e: Any):
        if widget.image_type.value == 'Folder':
            widget.selected_layer.visible = False
            if widget.folder_detect.value:
                pass
            else:
                widget.folder_detect.value = qtpy.QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder')
            widget.folder_detect.visible = True
        elif widget.image_type.value == 'One Layer':
            widget.selected_layer.visible = True
            widget.folder_detect.visible = False
            widget.folder_detect.value= ''
        else:
            widget.folder_detect.visible = False
            widget.selected_layer.visible = False
            widget.folder_detect.value = ''


    @widget.model_type.changed.connect
    def _change_model_display(e: Any):
        if widget.model_type.value == 'custom':
            widget.custom_model.visible = True
            widget.training_nucleus_size.visible = True

        else:
            widget.custom_model.visible = False
            widget.training_nucleus_size.visible = False








    widget.label_head.value = '<center>This plugin aims to bring YOLOv5 <br>developed by Ultralytics into Napari viewer</center><br><tt><a href="https://github.com/ultralytics/yolov5" style="color:gray;">https://github.com/ultralytics/yolov5</a></tt>'





    return widget            


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'YOLOv5'}
