import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout
import functools

from ui.utils import *

def create_initial_model_input(init_model_type):
	model_types = ['VGG', 'ResNet', 'Inception', 'FC']
	model_type_dropdown = widgets.Dropdown(
		options=model_types,
		value=init_model_type,
		description='Model Type:',
		disabled=False,
	)
	dropdown = model_type_dropdown.observe(create_inputs, names = 'value')
	display(model_type_dropdown)

def create_vgg_inputs():
	inp_shape_dropdown = widgets.Dropdown(
		options=[128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
		description='Input Shape (s x s x 3): ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_shape_dropdown.value = inp_shape_dropdown.options[0]

	inp_size_dropdown = widgets.Dropdown(
		options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
		description='Input Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_size_dropdown.value = inp_size_dropdown.options[0]

	vgg_layer_size_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5, 6, 7],
		description='VGG Layer Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	vgg_layer_size_dropdown.value = vgg_layer_size_dropdown.options[0]

	vgg_layers_dropdown = widgets.Dropdown(
		options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
		description='VGG Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	vgg_layers_dropdown.value = vgg_layers_dropdown.options[0]

	hidden_layers_dropdown = widgets.Dropdown(
		options=[100, 316, 1000, 3162],
		description='Hidden Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layers_dropdown.value = hidden_layers_dropdown.options[0]

	hidden_layer_size_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		description='Hidden Layer Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layer_size_dropdown.value = hidden_layer_size_dropdown.options[0]

	filters_dropdown = widgets.Dropdown(
		options=[16, 32, 64, 128, 256, 512, 1024],
		description='Number of Filters: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	filters_dropdown.value = filters_dropdown.options[0]

	out_shape_dropdown = widgets.Dropdown(
		options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
		description='Desired Output Shape: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	out_shape_dropdown.value = out_shape_dropdown.options[0]

	batch_size_dropdown = widgets.Dropdown(
		options=[8, 16, 32, 64, 128, 256, 512, 1024],
		description='Batch Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	batch_size_dropdown.value = batch_size_dropdown.options[0]

	epochs_txtbox = widgets.BoundedIntText(
		value=100,
		min=0,
		max=1000000,
		step=1,
		description='Epochs:',
		style = {'description_width': '150px'},
		disabled=False
	)

	run_button = widgets.Button(description = "Let's Calculate!")
	run_button.on_click(functools.partial(run_vgg_model_and_get_output, \
										inputs = [
												inp_shape_dropdown,
												inp_size_dropdown,
												vgg_layer_size_dropdown,
												vgg_layers_dropdown,
												hidden_layer_size_dropdown,
												hidden_layers_dropdown,
												filters_dropdown,
												out_shape_dropdown,
												batch_size_dropdown,
												epochs_txtbox
												]))

	display(inp_shape_dropdown)
	display(inp_size_dropdown)
	display(vgg_layer_size_dropdown)
	display(vgg_layers_dropdown)
	display(hidden_layer_size_dropdown)
	display(hidden_layers_dropdown)
	display(filters_dropdown)
	display(out_shape_dropdown)
	display(batch_size_dropdown)
	display(epochs_txtbox)
	display(run_button)

def run_vgg_model_and_get_output(btn, inputs):
	config = dict()
	config['input_shape'] = int(inputs[0].value)
	config['input_size'] = int(inputs[1].value)
	config['vgg_layers'] = int(inputs[3].value)
	config['vgg_layers_size'] = int(inputs[2].value)
	config['filters'] = int(inputs[6].value)
	config['hidden_layers_size'] = int(inputs[4].value)
	config['hidden_layers'] = int(inputs[5].value)
	config['output_shape'] = int(inputs[7].value)
	config['batch_size'] = int(inputs[8].value)
	config['epochs'] = int(inputs[9].value)

	# Model Run function
	training_time = get_training_time(config,'vgg')
	print(f"Training Time: {training_time}")

def create_resnet_inputs():
	inp_shape_dropdown = widgets.Dropdown(
		options=[128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
		description='Input Shape (s x s x 3): ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_shape_dropdown.value = inp_shape_dropdown.options[0]

	inp_size_dropdown = widgets.Dropdown(
		options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
		description='Input Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_size_dropdown.value = inp_size_dropdown.options[0]

	resnet_layers_dropdown = widgets.Dropdown(
		options=[3, 4, 5, 6, 7],
		description='ResNet Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	resnet_layers_dropdown.value = resnet_layers_dropdown.options[0]

	hidden_layers_dropdown = widgets.Dropdown(
		options=[100, 316, 1000, 3162],
		description='Hidden Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layers_dropdown.value = hidden_layers_dropdown.options[0]

	hidden_layer_size_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		description='Hidden Layer Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layer_size_dropdown.value = hidden_layer_size_dropdown.options[0]

	out_shape_dropdown = widgets.Dropdown(
		options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
		description='Desired Output Shape: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	out_shape_dropdown.value = out_shape_dropdown.options[0]

	batch_size_dropdown = widgets.Dropdown(
		options=[8, 16, 32, 64, 128, 256, 512, 1024],
		description='Batch Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	batch_size_dropdown.value = batch_size_dropdown.options[0]

	epochs_txtbox = widgets.BoundedIntText(
		value=100,
		min=0,
		max=1000000,
		step=1,
		description='Epochs:',
		style = {'description_width': '150px'},
		disabled=False
	)

	run_button = widgets.Button(description = "Let's Calculate!")
	run_button.on_click(functools.partial(run_resnet_model_and_get_output, \
										inputs = [
												inp_shape_dropdown,
												inp_size_dropdown,
												resnet_layers_dropdown,
												hidden_layer_size_dropdown,
												hidden_layers_dropdown,
												out_shape_dropdown,
												batch_size_dropdown,
												epochs_txtbox
												]))
	
	display(inp_shape_dropdown)
	display(inp_size_dropdown)
	display(resnet_layers_dropdown)
	display(hidden_layer_size_dropdown)
	display(hidden_layers_dropdown)
	display(out_shape_dropdown)
	display(batch_size_dropdown)
	display(epochs_txtbox)
	display(run_button)

def run_resnet_model_and_get_output(btn, inputs):
	config = dict()
	config['input_shape'] = int(inputs[0].value)
	config['input_size'] = int(inputs[1].value)
	config['resnet_layers'] = int(inputs[2].value)
	config['hidden_layers_size'] = int(inputs[3].value)
	config['hidden_layers'] = int(inputs[4].value)
	config['output_shape'] = int(inputs[5].value)
	config['batch_size'] = int(inputs[6].value)
	config['epochs'] = int(inputs[7].value)

	# Model Run function
	training_time = get_training_time(config,'resnet')
	print(f"Training Time: {training_time}")

def create_inception_inputs():
	inp_shape_dropdown = widgets.Dropdown(
		options=[128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
		description='Input Shape (s x s x 3): ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_shape_dropdown.value = inp_shape_dropdown.options[0]

	inp_size_dropdown = widgets.Dropdown(
		options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
		description='Input Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_size_dropdown.value = inp_size_dropdown.options[0]

	inception_layers_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5],
		description='Inception Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inception_layers_dropdown.value = inception_layers_dropdown.options[0]

	f1_dropdown = widgets.Dropdown(
		options=[64, 128, 192, 256, 320],
		description='F1 Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f1_dropdown.value = f1_dropdown.options[0]

	f2_in_dropdown = widgets.Dropdown(
		options=[128, 192, 256, 320, 384],
		description='F2 In Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f2_in_dropdown.value = f2_in_dropdown.options[0]

	f2_out_dropdown = widgets.Dropdown(
		options=[192, 256, 320, 384, 448],
		description='F2 Out Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f2_out_dropdown.value = f2_out_dropdown.options[0]

	f3_in_dropdown = widgets.Dropdown(
		options=[32, 64, 96, 128, 160],
		description='F3 In Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f3_in_dropdown.value = f3_in_dropdown.options[0]

	f3_out_dropdown = widgets.Dropdown(
		options=[32, 64, 96, 128, 160],
		description='F3 Out Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f3_out_dropdown.value = f3_out_dropdown.options[0]

	f4_out_dropdown = widgets.Dropdown(
		options=[32, 64, 96, 128, 160],
		description='F4 Out Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	f4_out_dropdown.value = f4_out_dropdown.options[0]

	hidden_layer_size_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		description='Hidden Layer Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layer_size_dropdown.value = hidden_layer_size_dropdown.options[0]

	hidden_layers_dropdown = widgets.Dropdown(
		options=[100, 316, 1000, 3162],
		description='Hidden Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layers_dropdown.value = hidden_layers_dropdown.options[0]

	out_shape_dropdown = widgets.Dropdown(
		options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
		description='Desired Output Shape: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	out_shape_dropdown.value = out_shape_dropdown.options[0]

	batch_size_dropdown = widgets.Dropdown(
		options=[8, 16, 32, 64, 128, 256, 512, 1024],
		description='Batch Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	batch_size_dropdown.value = batch_size_dropdown.options[0]

	epochs_txtbox = widgets.BoundedIntText(
		value=100,
		min=0,
		max=1000000,
		step=1,
		description='Epochs:',
		style = {'description_width': '150px'},
		disabled=False
	)

	run_button = widgets.Button(description = "Let's Calculate!")
	run_button.on_click(functools.partial(run_inception_model_and_get_output, \
										inputs = [
												inp_shape_dropdown,
												inp_size_dropdown,
												inception_layers_dropdown,
												f1_dropdown,
												f2_in_dropdown,
												f2_out_dropdown,
												f3_in_dropdown,
												f3_out_dropdown,
												f4_out_dropdown,
												hidden_layer_size_dropdown,
												hidden_layers_dropdown,
												out_shape_dropdown,
												batch_size_dropdown,
												epochs_txtbox
												]))
	

	display(inp_shape_dropdown)
	display(inp_size_dropdown)
	display(inception_layers_dropdown)
	display(f1_dropdown)
	display(f2_in_dropdown)
	display(f2_out_dropdown)
	display(f3_in_dropdown)
	display(f3_out_dropdown)
	display(f4_out_dropdown)
	display(hidden_layer_size_dropdown)
	display(hidden_layers_dropdown)
	display(out_shape_dropdown)
	display(batch_size_dropdown)
	display(epochs_txtbox)
	display(run_button)

def run_inception_model_and_get_output(btn, inputs):
	config = dict()
	config['input_shape'] = int(inputs[0].value)
	config['input_size'] = int(inputs[1].value)
	config['inception_layers'] = int(inputs[2].value)
	config['f1'] = int(inputs[3].value)
	config['f2_in'] = int(inputs[4].value)
	config['f2_out'] = int(inputs[5].value)
	config['f3_in'] = int(inputs[6].value)
	config['f3_out'] = int(inputs[7].value)
	config['f4_out'] = int(inputs[8].value)
	config['hidden_layers_size'] = int(inputs[9].value)
	config['hidden_layers'] = int(inputs[10].value)
	config['output_shape'] = int(inputs[11].value)
	config['batch_size'] = int(inputs[12].value)
	config['epochs'] = int(inputs[13].value)

	# Model Run function
	training_time = get_training_time(config,'inception')
	print(f"Training Time: {training_time}")

def create_fc_inputs():
	inp_shape_dropdown = widgets.Dropdown(
		options=[128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
		description='Input Shape (s x s x 3): ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_shape_dropdown.value = inp_shape_dropdown.options[0]

	inp_size_dropdown = widgets.Dropdown(
		options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
		description='Input Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	inp_size_dropdown.value = inp_size_dropdown.options[0]

	hidden_layers_dropdown = widgets.Dropdown(
		options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
		description='FC Hidden Layers: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	hidden_layers_dropdown.value = hidden_layers_dropdown.options[0]

	out_shape_dropdown = widgets.Dropdown(
		options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
		description='Desired Output Shape: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	out_shape_dropdown.value = out_shape_dropdown.options[0]

	batch_size_dropdown = widgets.Dropdown(
		options=[8, 16, 32, 64, 128, 256, 512, 1024],
		description='Batch Size: ',
		style = {'description_width': '150px'},
		disabled=False,
	)
	batch_size_dropdown.value = batch_size_dropdown.options[0]

	epochs_txtbox = widgets.BoundedIntText(
		value=100,
		min=0,
		max=1000000,
		step=1,
		description='Epochs:',
		style = {'description_width': '150px'},
		disabled=False
	)

	run_button = widgets.Button(description = "Let's Calculate!")
	run_button.on_click(functools.partial(run_fc_model_and_get_output, \
											inputs = [
											inp_shape_dropdown,
											inp_size_dropdown,
											hidden_layers_dropdown,
											out_shape_dropdown,
											batch_size_dropdown,
											epochs_txtbox
											]))

	display(inp_shape_dropdown)
	display(inp_size_dropdown)
	display(hidden_layers_dropdown)
	display(out_shape_dropdown)
	display(batch_size_dropdown)
	display(epochs_txtbox)
	display(run_button)

def run_fc_model_and_get_output(btn, inputs):
	config = dict()
	config['input_shape'] = int(inputs[0].value)
	config['input_size'] = int(inputs[1].value)
	config['hidden_layers'] = int(inputs[2].value)
	config['output_shape'] = int(inputs[3].value)
	config['batch_size'] = int(inputs[4].value)
	config['epochs'] = int(inputs[5].value)

	# Model Run function
	training_time = get_training_time(config,'fc')
	print(f"Training Time: {training_time}")

def create_inputs(change):
	if change.new == "VGG":
		clear_output()
		create_initial_model_input("VGG")
		create_vgg_inputs()
	elif change.new == "ResNet":
		clear_output()
		create_initial_model_input("ResNet")
		create_resnet_inputs()
	elif change.new == "Inception":
		clear_output()
		create_initial_model_input("Inception")
		create_inception_inputs()
	else:
		clear_output()
		create_initial_model_input("FC")
		create_fc_inputs()

