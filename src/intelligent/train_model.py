from pathlib import Path

class Train_Model:
    def __init__(self, batch_size=32, epochs=30):
        self.epochs=epochs
        self.batch_size=batch_size

    """
    Entry point for training
    """
    def train(self, model, tr_dat, tst_dat):
        model = self.__start(model, tr_dat, tst_dat)
        self.__save_structure_weight(model)
        print("Done training and Saving")

    """
    Start Training process
    """
    def __start(self, model, tr_dat, tst_dat):
        # Train the model
        model.fit(
            # Dataset to train
            tr_dat[0],
            # Expected label for dataset
            tr_dat[1],
            # How many images to be feed during training
            # typical 32 to 128
            batch_size=self.batch_size,
            # How many times to go through dataset
            # 1 full cycle = 1 epochs
            # More datasets, less epoch
            epochs=self.epochs,
            # What data used to validate
            # Data not used during training
            validation_data=tst_dat,
            # Randomizing data
            shuffle=True
        )
        return model

    """
    Save model structure and weights
    """
    def __save_structure_weight(self, model):
        # Save neural network structure
        print("Saving Model...")
        model_structure = model.to_json()
        f = Path("model_structure.json")
        f.write_text(model_structure)

        print("Saving Weights...")
        # Save neural network's trained weights
        model.save_weights("model_weights.h5")
