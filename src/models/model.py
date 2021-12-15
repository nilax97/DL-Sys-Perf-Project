from fc import *
from vgg import *
from resnet import *
from inception import *

def create_model():
    model_creator = dict()
    model_creator['vgg'] = create_vgg
    model_creator['resnet'] = create_resnet
    model_creator['inception'] = create_inception
    model_creator['fc'] = create_fc
    return model_creator

def visualize_model():
    model_visualizer = dict()
    model_visualizer['vgg'] = visualize_vgg_config
    model_visualizer['resnet'] = visualize_resnet_config
    model_visualizer['inception'] = visualize_inception_config
    model_visualizer['fc'] = visualize_fc_config
    return model_visualizer
