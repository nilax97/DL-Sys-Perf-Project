import tensorflow as tf

def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x)
    return x

def create_vgg(config):
    config['num_layers'] = len(config['vgg_layers'])
    input = tf.keras.layers.Input(shape=config['input_shape'])
    x = input
    for i in range(config['num_layers']):
        block_size = config['vgg_layers'][i]
        filter_num = config['filters'][i]
        act = 'relu'
        x = vgg_block(x,filter_num,block_size)
    x = tf.keras.layers.Flatten()(x)
    config['num_hidden_layers'] = len(config['hidden_layers'])
    for i in range(config['num_hidden_layers']):
        dim = config['hidden_layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)

    x = tf.keras.layers.Dense(config['output_shape'],activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                metrics=['accuracy'])
    return model

def visualize_vgg_config():
    vgg_config = dict()
    vgg_config['input_shape'] = (128,128,3)
    vgg_config['vgg_layers'] = [3,3,3]
    vgg_config['filters'] = [64,128,256]
    vgg_config['hidden_layers'] = [100,100]
    vgg_config['output_shape'] = 20
    # Output activation = always sigmoid
    # All convolution layers have 3x3 kernel and same padding
    # All pooling layers (end of VGG block) reduce image size by half
    # All hidden layers activated with ReLU
    # Optimizer is always sgd with lr = 0.01 and momentum=0.9

    vgg_model = create_vgg(vgg_config)

    return tf.keras.utils.plot_model(vgg_model,show_shapes=True,to_file="results/viz_vgg.png")
