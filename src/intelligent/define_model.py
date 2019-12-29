from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class Setup_Sequential:

    def __init__(self, dropout_val=0.3, show_summary=False,
        in_activation='relu', lyr_activation='relu', out_activation='softmax',
        loss='categorical_crossentropy', optimizer='adam'
    ):
        self.dropout_val = dropout_val
        self.show_summary = show_summary
        self.in_activation = in_activation
        self.lyr_activation = lyr_activation
        self.out_activation = out_activation
        self.loss=loss
        self.optimizer = optimizer

    """
    Define Sequential model
    """
    def define_model(self, nodes=[]):
        num_nodes = len(nodes)
        print("Number of nodes ::: {}".format(num_nodes))
        # Create a model and add layers
        model = Sequential()
        model = self.__define_input(model=model)
        model = self.__define_hidden_layer(model=model, nodes=nodes,
            num_nodes=num_nodes)
        model = self.__define_output_compile(model=model)

        if(self.show_summary == True):
            model.summary()
        return model

    """
    Define model input
    """
    def __define_input(self, model):
        # Input
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
            input_shape=(32, 32, 3), activation=self.in_activation)
        )
        return model

    """
    Define hiden layer
    """
    def __define_hidden_layer(self, model, nodes, num_nodes):
        # hidden layer
        for i in range(num_nodes):
            print("Setting layer {}".format(i))
            model.add(Conv2D(filters=nodes[i], kernel_size=(3, 3),
                padding='same', activation=self.lyr_activation)
            )
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=self.dropout_val))
        return model

    """
    Define model output and compile
    """
    def __define_output_compile(self, model):
        # Output
        model.add(Flatten())
        model.add(Dense(units=512, activation=self.lyr_activation))
        model.add(Dropout(rate=self.dropout_val))
        model.add(Dense(units=10, activation=self.out_activation))

        # Compile the model
        model.compile(
            # Loss function
            loss=self.loss,
            # Optimizer function
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        return model
