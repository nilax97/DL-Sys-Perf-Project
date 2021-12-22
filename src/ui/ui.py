from ui.params import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

def create_ui():
	# Initially FC is chosen
	create_initial_model_input('FC')
	create_fc_inputs()