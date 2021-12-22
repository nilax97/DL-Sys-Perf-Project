import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from collections import defaultdict
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from analysis.utils import get_models,get_single_time_train_data
from analysis.models import get_models

models = get_models()


def get_model_stored_train_times(model_name,dir_path):
	# Necessary Imports
	data_dict = defaultdict(list)

	for file_name in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file_name)
		file_name_elems = file_name.split('-')
		varying_attr_name = file_name_elems[1]
		varying_attr_val = int(file_name_elems[2][:file_name_elems[2].find('.')])

		# Forming a list of tuples for each of the varying attributes...
		if varying_attr_name not in data_dict.keys():
			data_dict[varying_attr_name] = []

		if os.path.isfile(file_path):
			file_to_read = open(file_path, "rb")
			data = pickle.load(file_to_read)
			x,y = get_single_time_train_data(data)
			y_pred = models[model_name].predict(x)[0]
			data_dict[varying_attr_name].append([varying_attr_val, np.mean(data['train_times']), y_pred])

	# Sorting the list of tuples to get cleaner analysis
	for varying_attr_name, varying_attr_vals in data_dict.items():
		data_dict[varying_attr_name] = sorted(varying_attr_vals, key = lambda x : x[0])

	return data_dict

# Visualize all the data for the training times
def plot_varying_attrs_vs_train_times(model_name, dict_data):
	import matplotlib.pyplot as plt
	import seaborn as sns

	num_plots = len(dict_data)
	fig, axs = plt.subplots(num_plots,figsize=(15,num_plots*5))
	# print(f'Training Times (y-axis) for Models (with varying attributes): {model_name.upper()}')

	plt_counter = 0
	for varying_attr_name, varying_attr_vals in dict_data.items():
		x_vals = [attr[0] for attr in varying_attr_vals]
		y_vals = [attr[1] for attr in varying_attr_vals]
		y_pred_vals = [attr[2] for attr in varying_attr_vals]
		axs[plt_counter].set_title(varying_attr_name)
		axs[plt_counter].plot(x_vals,y_vals,c='blue',label='Actual')
		axs[plt_counter].plot(x_vals,y_pred_vals,c='orange',label='Predicted')
		axs[plt_counter].legend()
		plt_counter += 1
	
	# Saving the image
	plt.savefig(os.path.join("results", f'{model_name}_plot_vs.png'))


def get_model_train_times_per_aux_vars(model_name,dir_path):
	# Necessary Imports
	import os
	import numpy as np
	from collections import defaultdict
	import pickle

	data_dict = defaultdict(dict)

	for file_name in os.listdir(dir_path):
		if file_name.find("input_shape") == -1 and file_name.find("layers") == -1:
			continue
		file_path = os.path.join(dir_path, file_name)
		file_name_elems = file_name.split('-')
		varying_attr_name = file_name_elems[1]

		# Forming a dict of list of tuples for each of the varying attributes...
		if varying_attr_name not in data_dict.keys():
			data_dict[varying_attr_name] = dict()

		if os.path.isfile(file_path):
			file_to_read = open(file_path, "rb")
			data = pickle.load(file_to_read)
			
			x,y = get_single_time_train_data(data)
			y_pred = models[model_name].predict(x)[0]

			total_flops = sum(list(data['flops_param'].values()))
			if 'flops_param' not in data_dict[varying_attr_name].keys():
				data_dict[varying_attr_name]['flops_param'] = []
			data_dict[varying_attr_name]['flops_param'].append((total_flops, np.mean(data['train_times']),y_pred))

			trainable_params = data['weights_param']['trainable']
			if 'trainable' not in data_dict[varying_attr_name].keys():
				data_dict[varying_attr_name]['trainable'] = []
			data_dict[varying_attr_name]['trainable'].append((trainable_params, np.mean(data['train_times']),y_pred))

			total_depth = sum(list(data['layers_param'].values()))
			if 'layers_param' not in data_dict[varying_attr_name].keys():
				data_dict[varying_attr_name]['layers_param'] = []
			data_dict[varying_attr_name]['layers_param'].append((total_depth, np.mean(data['train_times']),y_pred))

	# Sorting the list of tuples to get cleaner analysis
	for varying_attr_name, varying_attr_vals in data_dict.items():
		for varying_attr_val_name, varying_data in varying_attr_vals.items():
			data_dict[varying_attr_name][varying_attr_val_name] = sorted(varying_data, key = lambda x : x[0])

	return data_dict

# Visualize all the data for the training times
def plot_varying_aux_attrs_vs_train_times(model_name, dict_data):
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	aux_vars = ['flops_param', 'trainable', 'layers_param']
	num_plots = len(dict_data)*len(aux_vars)
	fig, axs = plt.subplots(num_plots,figsize=(15,num_plots*5))
	# print(f'Training Times (y-axis) for Models (with varying attributes): {model_name.upper()}')


	plt_counter = 0
	for varying_attr_name, varying_attr_vals in dict_data.items():
		for varying_attr_val_name, varying_attr_data in varying_attr_vals.items():
			x_vals = [x[0] for x in varying_attr_data]
			y_vals = [x[1] for x in varying_attr_data]
			y_pred_vals = [x[2] for x in varying_attr_data]
			axs[plt_counter].set_title(f"{varying_attr_val_name} w.r.t {varying_attr_name}")
			axs[plt_counter].plot(x_vals,y_vals,c='blue',label='Actual')
			axs[plt_counter].plot(x_vals,y_pred_vals,c='orange',label='Predicted')
			axs[plt_counter].legend()
			plt_counter += 1
	
	# Saving the image
	plt.savefig(os.path.join("results", f'{model_name}_plot_vs_aux.png'))

def run_analysis(model_name):
	plot_varying_attrs_vs_train_times(model_name, 
		get_model_stored_train_times(model_name,"pickle/"+model_name))

	plot_varying_aux_attrs_vs_train_times(model_name, 
		get_model_train_times_per_aux_vars(model_name,"pickle/"+model_name))
