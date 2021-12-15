import tensorflow as tf

def create_fc(config):
    config['hidden_layers'] = len(config['layers'])
    input = tf.keras.layers.Input(shape=config['input_shape'])
    if config['input_dropout'] is not None:
        x = tf.keras.layers.Dropout(config['input_dropout'])(input)
    else:
        x = input
    for i in range(config['hidden_layers']):
        dim = config['layers'][i]
        act = 'relu'
        x = tf.keras.layers.Dense(dim,activation=act)(x)
        if config['dropout'] is not None:
            x = tf.keras.layers.Dropout(config['dropout'])(x)
            
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Dense(config['output_shape'],activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                metrics=['accuracy'])
    return model

def visualize_fc_config():
    fc_config = dict()
    fc_config['input_shape'] = 1000
    fc_config['output_shape'] = 10
    fc_config['input_dropout'] = 0.2
    fc_config['dropout'] = 0.5
    fc_config['hidden_layers'] = 2
    fc_config['layers'] = [1000,1000]
    # Output activation = always sigmoid
    # All hidden layers have same dropout
    # All hidden layers activated with ReLU
    # Optimizer is always sgd with lr = 0.01 and momentum=0.9

    fc_model = create_fc(fc_config)

    return tf.keras.utils.plot_model(fc_model,show_shapes=True,to_file="fc.png")
