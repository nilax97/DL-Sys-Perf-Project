vgg_range = dict()
vgg_range['input_shape'] = [128,1024,15,0]
vgg_range['input_size'] = [10,19,10,1] # Logspace 9
vgg_range['vgg_layers'] = [2,10,9,0]
vgg_range['vgg_layers_size'] = [1,7,7,0] 
vgg_range['filters'] = [4,10,7,1] # Logspace 8
vgg_range['hidden_layers_size'] = [1,10,10,0]
vgg_range['hidden_layers'] = [2,3.5,4,2] # Logspace 7
vgg_range['output_shape'] = [1,10,10,1] # Logspace 10
vgg_range['batch_size'] = [3,10,8,1] #Logspace 8

inception_range = dict()
inception_range['input_shape'] = [128,1024,15,0]
inception_range['input_size'] = [10,19,10,1] # Logspace 9
inception_range['inception_layers'] = [1,5,5,0]
inception_range['f1'] = [64,320,5,0]
inception_range['f2_in'] = [128,384,5,0]
inception_range['f2_out'] = [192,448,5,0]
inception_range['f3_in'] = [32,160,5,0]
inception_range['f3_out'] = [32,160,5,0]
inception_range['f4_out'] = [32,160,5,0]
inception_range['hidden_layers_size'] = [1,10,10,0]
inception_range['hidden_layers'] = [2,3.5,4,2] # Logspace 7
inception_range['output_shape'] = [1,10,10,1] # Logspace 10
inception_range['batch_size'] = [3,10,8,1] #Logspace 8

resnet_range = dict()
resnet_range['input_shape'] = [128,1024,15,0]
resnet_range['input_size'] = [10,19,10,1] # Logspace 9
resnet_range['resnet_layers'] = [3,7,5,0]
resnet_range['hidden_layers_size'] = [1,10,10,0]
resnet_range['hidden_layers'] = [2,3.5,4,2] # Logspace 7
resnet_range['output_shape'] = [1,10,10,1] # Logspace 10
resnet_range['batch_size'] = [3,10,8,1] #Logspace 8

fc_range = dict()
fc_range['input_shape'] = [128,1024,15,0]
fc_range['input_size'] = [10,19,10,1] # Logspace 9
fc_range['hidden_layers'] = [1,10,10,0]
fc_range['output_shape'] = [1,10,10,1] # Logspace 10
fc_range['batch_size'] = [3,10,8,1] #Logspace 8

def get_ranges():
    ranges = dict()
    ranges['vgg'] = vgg_range
    ranges['resnet'] = resnet_range
    ranges['inception'] = inception_range
    ranges['fc'] = fc_range
    return ranges
