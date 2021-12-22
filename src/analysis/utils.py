import os
import pickle
from experiment.create_config import get_params
from experiment.ranges import get_ranges
from experiment.create_config import create_base_params,create_config
from experiment.utils import run_model

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from collections import defaultdict
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

def get_time_train_data(model_type,folder_name):
    x = []
    y = []
    params = get_params(get_ranges()[model_type])
    for key in params:
        base_param = create_base_params(params)
        for value in params[key]:
            filename = f'{folder_name}/{model_type}/{model_type}-{key}-{value}.pickle'
            if os.path.exists(filename):
                with open(filename, 'rb') as handle:
                    output_config = pickle.load(handle)
                x.append(list(output_config['flops_param'].values()) + 
                        list(output_config['layers_param'].values()) + 
                        list(output_config['weights_param'].values()))
                x[-1].append(output_config['input_size'])
                x[-1].append(output_config['batch_size'])
                y.append(list(output_config['train_times']))
    x = np.asarray(x).astype('float64')
    y = np.asarray(y).astype('float64')
    y = np.mean(y,axis=1)
    return x,y

def get_single_time_train_data(output_config):
    x = []
    y = []
    x.append(list(output_config['flops_param'].values()) +
             list(output_config['layers_param'].values()) +
             list(output_config['weights_param'].values()))
    x[-1].append(output_config['input_size'])
    x[-1].append(output_config['batch_size'])
    y.append(list(output_config['train_times']))
    x = np.asarray(x).astype('float64')
    y = np.asarray(y).astype('float64')
    y = np.mean(y,axis=1)
    return x,y

def get_trained_model(model_type,folder_name):
    x,y = get_time_train_data(model_type,folder_name)
    min_error = float('inf')
    min_model = None
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        rf = RandomForestRegressor(n_estimators=20)
        rf.fit(x_train, y_train)
        y_pred_train = rf.predict(x_train)
        y_pred_test = rf.predict(x_test)
        train_error = mean_absolute_percentage_error(y_train,y_pred_train)
        test_error = mean_absolute_percentage_error(y_test,y_pred_test)
        if test_error < min_error:
            min_loss = test_error
            min_model = rf
        if (test_error - train_error) < 0.05 and max(test_error,train_error) < 0.1:
            break
    y_pred_train = min_model.predict(x_train)
    y_pred_test = min_model.predict(x_test)
#     print(f'{model_type} - train MSE : {mean_absolute_percentage_error(y_train,y_pred_train)}')
#     print(f'{model_type} - test MSE : {mean_absolute_percentage_error(y_test,y_pred_test)}')

    return min_model

def get_models(folder_name):
    models = dict()
    models['vgg'] = get_trained_model('vgg',folder_name)
    models['inception'] = get_trained_model('inception',folder_name)
    models['resnet'] = get_trained_model('resnet',folder_name)
    models['fc'] = get_trained_model('fc',folder_name)
    return models
