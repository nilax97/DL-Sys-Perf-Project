import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

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

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.perf_counter() - self.epoch_time_start)

def run_model(config,model_type):
    model = create_model()[model_type](config)
    flops,weights,layers = get_model_params(model)
    config['flops_param'] = flops
    config['weights_param'] = weights
    config['layers_param'] = layers
    dataset = create_image_dataset(config)
    time_callback = TimeHistory()
    steps_per_epoch = config['input_size'] // config['batch_size']
    hist = model.fit(dataset,steps_per_epoch=steps_per_epoch, epochs=6, callbacks = [time_callback],verbose=False)
    config['train_times'] = time_callback.times[1:]
    return config
