import numpy as np

def get_params(model_range):
    params = dict()
    for key in model_range.keys():
        val = model_range[key]
        if val[-1] == 0:
            params[key] = np.linspace(val[0],val[1],num=val[2]).astype('int')
        elif val[-1] == 1:
            params[key] = np.logspace(val[0],val[1],num=val[2],base=2).astype('int')
        elif val[-1] == 2:
            params[key] = np.logspace(val[0],val[1],num=val[2],base=10).astype('int')
    return params

def create_base_params(params):
    config = dict()
    for k,v in params.items():
        config[k] = v[0]
    return config


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

def create_config():
    config_creator = dict()
    config_creator['vgg'] = create_base_vgg_config
    config_creator['resnet'] = create_base_resnet_config
    config_creator['inception'] = create_base_inception_config
    config_creator['fc'] = create_base_fc_config
    return config_creator
