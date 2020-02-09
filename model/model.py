from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def define_sequential(input_shape, output_class, hidden_neurons, dropout_val,
    in_activation, lyr_activation, out_activation,
    loss, optimizer, show_summary=False):

    num_hidden_layers = len(hidden_neurons)
    print("Number of nodes ::: {}".format(num_hidden_layers))
    model = Sequential()

    # Input layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                        input_shape=input_shape, activation=in_activation
                    )
                )
    # Define hiden layer
    for i in range(num_hidden_layers):
        model.add(Conv2D(filters=hidden_neurons[0], kernel_size=(3, 3),
            padding='same', activation=lyr_activation)
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=dropout_val))

    # Output Layer
    model.add(Flatten())
    model.add(Dense(units=512, activation=lyr_activation))
    model.add(Dropout(rate=dropout_val))
    model.add(Dense(units=output_class, activation=out_activation))

    # Compile the model
    model.compile(
        # Loss function
        loss=loss,
        # Optimizer function
        optimizer=optimizer,
        metrics=['accuracy']
    )

    if(show_summary == True):
        model.summary()
    return model
