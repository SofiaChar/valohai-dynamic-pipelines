import os
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPool2D
import tensorflow as tf
import debugpy

# Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_name', type=str, default="all_harbors")
    parser.add_argument('--debug', type=bool, default=False)
    return parser.parse_args()

# Helper function to log training metrics
def log_metadata(epoch, logs):
    print()
    print(json.dumps({
        "epoch": epoch,
        "accuracy": logs["accuracy"],
        "loss": logs["loss"]
    }))

def main():
    if args.debug:
        # Listen on port 5678
        debugpy.listen(5678)
        # The script is halted here, until a debugger is attached
        debugpy.wait_for_client()

    args = parse_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    dataset_name = args.dataset_name


    # Read the data
    print("Reading the data...")

    input_data_dir = os.getenv('VH_INPUTS_DIR', '.inputs')
    path_dataset = os.path.join(input_data_dir, 'dataset/preprocessed_data.npz')

    with np.load(path_dataset, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_val, y_val = f["x_val"], f["y_val"]

    print("Starting the model training...")

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_regularizer=None,
            padding="same",
            input_shape=(150, 150, 3),
        )
    )
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu", kernel_regularizer=None))
    model.add(Dense(16, activation="relu", kernel_regularizer=None))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(5, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=[callback],
    )

    # Evaluate the model and print out the test metrics as JSON
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=2)
    print(json.dumps({
        "val_accuracy": val_accuracy,
        "val_loss": val_loss
    }))
    model.summary()
    print("Model training completed")

    # Save the trained model
    print("Saving the trained model...")
    output_dir_path = os.getenv('VH_OUTPUTS_DIR', '.outputs')
    dataset_name = dataset_name
    output_path = os.path.join(output_dir_path, f"model-" + dataset_name + ".h5")
    model.save(output_path)
    print("Saved completed artefacts to outputs")

if __name__ == '__main__':
    main()
