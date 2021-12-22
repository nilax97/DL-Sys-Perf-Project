# Models
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import time
import pickle
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")
def create_fc(config):
    config['hidden_layers'] = len(config['layers'])
    input = tf.keras.layers.Input(shape=config['input_shape'])
    if config['input_dropout'] is not None:
        x = tf.keras.layers.Dropout(config['input_dropout'])(input)
    else:
        x = input
    for i in range(config['hidden_layers']):
        dim = config['layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)
        if config['dropout'] is not None:
            x = tf.keras.layers.Dropout(config['dropout'])(x)
            
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Dense(config['output_shape'],activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                metrics=['accuracy'])
    return model

def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x)
    return x

def create_vgg(config):
    config['num_layers'] = len(config['vgg_layers'])
    input = tf.keras.layers.Input(shape=config['input_shape'])
    x = input
    for i in range(config['num_layers']):
        block_size = config['vgg_layers'][i]
        filter_num = config['filters'][i]
        act = 'relu'
        x = vgg_block(x,filter_num,block_size)
    x = tf.keras.layers.Flatten()(x)
    config['num_hidden_layers'] = len(config['hidden_layers'])
    for i in range(config['num_hidden_layers']):
        dim = config['hidden_layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)

    x = tf.keras.layers.Dense(config['output_shape'],activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                metrics=['accuracy'])
    return model

def inception_block(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = tf.keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    # 3x3 conv
    conv3 = tf.keras.layers.Conv2D(f2_in, (1,1), padding='same', activation='relu')(x)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    # 5x5 conv
    conv5 = tf.keras.layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    # 3x3 max pooling
    pool = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool = tf.keras.layers.Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
    pool = tf.keras.layers.BatchNormalization()(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def create_inception(config):
    config['num_layers'] = len(config['inception_layers'])
    input = tf.keras.layers.Input(shape=config['input_shape'])
    x = tf.keras.layers.Conv2D(64, (7,7), padding='valid', activation='relu', strides=(2,2))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = tf.keras.layers.Conv2D(128, (1,1), padding='same', activation='relu', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(192, (3,3), padding='same', activation='relu', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    for i in range(config['num_layers']):
        for j in range(config['inception_layers'][i]):
            x = inception_block(x,config['f1'][i][j],config['f2_in'][i][j],config['f2_out'][i][j],
                                                    config['f3_in'][i][j],config['f3_out'][i][j],config['f4_out'][i][j])
        x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    config['num_hidden_layers'] = len(config['hidden_layers'])
    for i in range(config['num_hidden_layers']):
        dim = config['hidden_layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)

    x = tf.keras.layers.Dense(config['output_shape'],activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                metrics=['accuracy'])
    return model

def conv_relu(x, filters, kernel_size, strides=1):
        
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

def identity_block(tensor, filters):
        
        x = conv_relu(tensor, filters=filters, kernel_size=1, strides=1)
        x = conv_relu(x, filters=filters, kernel_size=3, strides=1)
        x = tf.keras.layers.Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Add()([tensor,x])
        x = tf.keras.layers.ReLU()(x)
        
        return x

def identity_block_small(tensor, filters):
        
        x = conv_relu(tensor, filters=filters, kernel_size=3, strides=1)
        x = conv_relu(x, filters=filters, kernel_size=3, strides=1)
        
        x = tf.keras.layers.Add()([tensor,x])
        x = tf.keras.layers.ReLU()(x)
        
        return x

def projection_block(tensor, filters, strides):
        
        x = conv_relu(tensor, filters=filters, kernel_size=1, strides=strides)
        x = conv_relu(x, filters=filters, kernel_size=3, strides=1)
        x = tf.keras.layers.Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        shortcut = tf.keras.layers.Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Add()([shortcut,x])
        x = tf.keras.layers.ReLU()(x)
        
        return x

def projection_block_small(tensor, filters, strides):
        
        x = conv_relu(tensor, filters=filters, kernel_size=3, strides=strides)
        x = conv_relu(x, filters=filters, kernel_size=3, strides=1)
        
        shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=strides)(tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Add()([shortcut,x])
        x = tf.keras.layers.ReLU()(x)
        
        return x

def resnet_block(x, filters, reps, strides):
        
        x = projection_block(x, filters, strides)
        for _ in range(reps-1):
                x = identity_block(x,filters)
                
        return x

def resnet_block_small(x, filters, reps, strides):
        
        x = projection_block_small(x, filters, strides)
        for _ in range(reps):
                x = identity_block_small(x,filters)
                
        return x

def create_resnet(config):

    input = tf.keras.layers.Input(shape=config['input_shape'])

    x = conv_relu(input, filters=64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPool2D(pool_size = 3, strides =2)(x)
    if config['small']==False:
            x = resnet_block(x, filters=64, reps=config['resnet_layers'][0], strides=1)
            x = resnet_block(x, filters=128, reps=config['resnet_layers'][1], strides=2)
            x = resnet_block(x, filters=256, reps=config['resnet_layers'][2], strides=2)
            x = resnet_block(x, filters=512, reps=config['resnet_layers'][3], strides=2)
    else:
            x = resnet_block_small(x, filters=64, reps=config['resnet_layers'][0], strides=1)
            x = resnet_block_small(x, filters=128, reps=config['resnet_layers'][1], strides=2)
            x = resnet_block_small(x, filters=256, reps=config['resnet_layers'][2], strides=2)
            x = resnet_block_small(x, filters=512, reps=config['resnet_layers'][3], strides=2)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    config['num_hidden_layers'] = len(config['hidden_layers'])
    for i in range(config['num_hidden_layers']):
        dim = config['hidden_layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)

    output = tf.keras.layers.Dense(config['output_shape'], activation ='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                            metrics=['accuracy'])
    return model

def create_base_vgg_config(params):
    config = dict()
    config['input_shape'] = (params['input_shape'],params['input_shape'],3)
    config['vgg_layers'] = [params['vgg_layers']] * params['vgg_layers_size']

    filters = []
    for i in range(params['vgg_layers_size']):
        filters.append(params['filters']*(2**i))
    config['filters'] = filters

    config['hidden_layers'] = [params['hidden_layers']] * params['hidden_layers_size']
    config['output_shape'] = params['output_shape']

    config['input_size'] = params['input_size']
    config['batch_size'] = params['batch_size']
    
    config['model'] = "VGG"
    return config

def create_base_inception_config(params):
    config = dict()
    config['input_shape'] = (params['input_shape'],params['input_shape'],3)
    config['inception_layers'] = [params['inception_layers']] * 3

    config['f1'] = []
    config['f2_in'] = []
    config['f2_out'] = []
    config['f3_in'] = []
    config['f3_out'] = []
    config['f4_out'] = []

    for val in config['inception_layers']:
        config['f1'].append([params['f1']]*val)
        config['f2_in'].append([params['f2_in']]*val)
        config['f2_out'].append([params['f2_out']]*val)
        config['f3_in'].append([params['f3_in']]*val)
        config['f3_out'].append([params['f3_out']]*val)
        config['f4_out'].append([params['f4_out']]*val)

    config['hidden_layers'] = [params['hidden_layers']] * params['hidden_layers_size']
    config['output_shape'] = params['output_shape']

    config['input_size'] = params['input_size']
    config['batch_size'] = params['batch_size']
    
    config['model'] = "Inception"
    return config

def create_base_resnet_config(params):
    config = dict()
    config['input_shape'] = (params['input_shape'],params['input_shape'],3)
    config['small'] = False
    config['resnet_layers'] = [params['resnet_layers']] * 4

    config['hidden_layers'] = [params['hidden_layers']] * params['hidden_layers_size']
    config['output_shape'] = params['output_shape']

    config['input_size'] = params['input_size']
    config['batch_size'] = params['batch_size']
    
    config['model'] = "ResNet"
    return config

def create_base_fc_config(params):
    config = dict()
    config['input_shape'] = params['input_shape']
    config['input_dropout'] = 0.2
    config['dropout'] = 0.5

    config['layers'] = [1000] * params['hidden_layers']
    config['output_shape'] = params['output_shape']

    config['input_size'] = params['input_size']
    config['batch_size'] = params['batch_size']
    
    config['model'] = "FC"
    return config

def get_flops(model, batch_size=None,allowed_flops=['MatMul', 'Mul', 'Rsqrt', 'BiasAdd', 'Sub', 'Softmax', 'Conv2D', 'MaxPool', 'Mean']):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output'] = 'none'
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    
    ret_val = dict()
    for fl in allowed_flops:
        ret_val[fl] = 0
    f = flops.children
    while(len(f) > 0):
        if f[0].name in allowed_flops:
            ret_val[f[0].name] = f[0].total_float_ops
        f = f[0].children
    return ret_val

def get_weights(model):
    ret_val = dict()
    ret_val['trainable'] = np.sum([np.product([xi for xi in x.get_shape()]) for x in model.trainable_weights])
    ret_val['non_trainable'] = np.sum([np.product([xi for xi in x.get_shape()]) for x in model.non_trainable_weights])
    return ret_val

def get_layers(model):
    ret_val = dict()
    for l in model.layers:
        name = l.__class__.__name__
        if name in ret_val:
            ret_val[name] += 1
        else:
            ret_val[name] = 1
    return ret_val

allowed_flops = ['MatMul', 'Mul', 'Rsqrt', 'BiasAdd', 'Sub', 'Softmax', 'Conv2D', 'MaxPool', 'Mean']
def get_model_params(model,batch_size = 64,x_shape=[]):
    flops = get_flops(model)
    weights = get_weights(model)
    layers = get_layers(model)
    
    return flops,weights,layers

model_creator = dict()
model_creator['vgg'] = create_vgg
model_creator['resnet'] = create_resnet
model_creator['inception'] = create_inception
model_creator['fc'] = create_fc

base_param_creator = dict()
base_param_creator['vgg'] = create_base_vgg_config
base_param_creator['resnet'] = create_base_resnet_config
base_param_creator['inception'] = create_base_inception_config
base_param_creator['fc'] = create_base_fc_config

def get_single_train_data(output_config):
    x = []
    y = []
    x.append(list(output_config['flops_param'].values()) + 
             list(output_config['layers_param'].values()) + 
             list(output_config['weights_param'].values()))
    x[-1].append(output_config['input_size'])
    x[-1].append(output_config['batch_size'])
    x = np.asarray(x).astype('float64')
    return x

def get_training_time(config,model_name):
    input_config = base_param_creator[model_name](config)
    model = model_creator[model_name](input_config)
    flops,weights,layers = get_model_params(model)
    input_config['flops_param'] = flops
    input_config['weights_param'] = weights
    input_config['layers_param'] = layers
    
    with open('results/trained_model.pickle', 'rb') as handle:
                models = pickle.load(handle)
    
    x = get_single_train_data(input_config)
    
    time = models[model_name].predict(x)[0]
    return time * config['epochs']