import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from collections import defaultdict
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from analysis.utils import get_models,get_single_time_train_data

def get_model_stored_train_times(dir_path,folder_name):
    # Necessary Imports
    data_dict = defaultdict(list)
    models = get_models(folder_name)
    for file_name in os.listdir(os.path.join(folder_name,dir_path)):
        file_path = os.path.join(folder_name,dir_path, file_name)
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
            y_pred = models[dir_path].predict(x)[0]
            data_dict[varying_attr_name].append([varying_attr_val, np.mean(data['train_times']), y_pred])

    # Sorting the list of tuples to get cleaner analysis
    for varying_attr_name, varying_attr_vals in data_dict.items():
        data_dict[varying_attr_name] = sorted(varying_attr_vals, key = lambda x : x[0])

    return data_dict

# Visualize all the data for the training times
def plot_varying_attrs_vs_train_times(model_name, dict_data):
    import matplotlib.pyplot as plt
    import seaborn as sns

#     plt.rcParams["figure.figsize"] = (15,40)
    num_plots = len(dict_data)
    fig, axs = plt.subplots(num_plots,figsize=(15,num_plots*5))
    print(f'Training Times (y-axis) for Models (with varying attributes): {model_name.upper()}')

    plt_counter = 0
    for varying_attr_name, varying_attr_vals in dict_data.items():
        x_vals = [attr[0] for attr in varying_attr_vals]
        y_vals = [attr[1] for attr in varying_attr_vals]
        y_pred_vals = [attr[2] for attr in varying_attr_vals]
        axs[plt_counter].set_title(varying_attr_name)
        axs[plt_counter].plot(x_vals,y_vals,c='blue',label='Actual')
        axs[plt_counter].plot(x_vals,y_pred_vals,c='orange',label='Predicted')
        axs[plt_counter].legend()
#         sns.barplot(x_vals, y_vals, ax = axs[plt_counter])
        plt_counter += 1
    
    # Saving the image
    plt.savefig(os.path.join("Visualizations", f'{model_name}_data_vs.png'))
