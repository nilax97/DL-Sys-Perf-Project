import os
import pickle
from experiment.create_config import get_params
from experiment.ranges import get_ranges
from experiment.create_config import create_base_params,create_config
from experiment.utils import run_model

def run_experiment(model_type,folder):
    params = get_params(get_ranges()[model_type])
    for key in params:
        base_param = create_base_params(params)
        for value in params[key]:
            print(f'[{model_type}] Running {key} : {value}')
            filename = f'{folder}/{model_type}/{model_type}-{key}-{value}.pickle'
            if os.path.exists(filename):
                with open(filename, 'rb') as handle:
                    output_config = pickle.load(handle)
                    print(output_config['train_times'])
                continue
            base_param[key] = value
            output_config = run_model(create_config()[model_type](base_param),model_type)
            print(output_config['train_times'])
            with open(filename, 'wb') as handle:
                pickle.dump(output_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
