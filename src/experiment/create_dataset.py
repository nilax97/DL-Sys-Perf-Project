import tensorflow as tf

def create_image_dataset(config):
    input_shape = [128] + list(config['input_shape'])
    output_shape = [128] + [config['output_shape']]
    batch_size = config['batch_size']
    x = tf.random.uniform(shape=input_shape)
    y = tf.random.uniform(shape=output_shape)
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    return dataset

def create_fc_dataset(config):
    input_shape = [128] + [config['input_shape']]
    output_shape = [128] + [config['output_shape']]
    batch_size = config['batch_size']
    x = tf.random.uniform(shape=input_shape)
    y = tf.random.uniform(shape=output_shape)
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    return dataset
