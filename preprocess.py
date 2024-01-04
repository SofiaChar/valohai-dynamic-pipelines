import cv2
import numpy as np
import pandas as pd
from utils.image import ImagePreprocessing
import valohai
import json
import os

# Get dataset names and the size for validation set
dataset_names = valohai.parameters("dataset_names").value
validation_split = valohai.parameters("validation_split").value

# Read the execution details from the configuration file for dataset naming
f = open("/valohai/config/execution.json")
exec_details = json.load(f)

# Get the execution ID
exec_id = exec_details["valohai.execution-id"]

# Preprocess the data
for dataset in dataset_names:
    print("Preprocessing for dataset: " + dataset)
    images = valohai.inputs("dataset").paths()

    # Get the correct labels for the dataset
    labels = pd.read_csv(valohai.inputs("labels").path("train_" + dataset + ".csv"))
    labels.set_index("image", inplace=True)

    train_images = []
    test_images = []
    files_tmp = []
    categories_tmp = []

    for file in images:
        filename = os.path.basename(file)
        # Do we have a label for this image
        try:
            category = labels.loc[filename].item()
            categories_tmp.append(category)
            files_tmp.append(filename)
            image = cv2.imread(file)
            train_images.append(image)
        except:
            if len(test_images) <= 1000:
                image = cv2.imread(file)
                test_images.append(image)

    # Creating a new dataframe to be used for the preprocessing.
    df = pd.DataFrame(data={"image": files_tmp, "category": categories_tmp})

    print("Resizing training and test data...")
    preprocess = ImagePreprocessing(
        train_images, test_images, height=150, length=len(train_images), dataframe=df
    )
    rez_images, LABELS, test_rez_images = preprocess.Reshape()

    print("Finding labels for the images...")
    onehot_labels = preprocess.OneHot(LABELS)

    print("Splitting to training and validation data...")
    X_train, X_val, Y_train, Y_val = preprocess.splitdata(
        rez_images, onehot_labels, validation_split
    )

    # Save preprocessed training data
    print("Saving preprocessed data...")
    path = valohai.outputs("train").path("preprocessed_data_" + dataset + ".npz")
    np.savez_compressed(
        path, x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val
    )

    metadata_train = {
        "valohai.dataset-versions": ["dataset://" + dataset + "_train/" + exec_id]
    }

    metadata_path = valohai.outputs("train").path(
        "preprocessed_data_" + dataset + ".npz.metadata.json"
    )
    with open(metadata_path, "w") as outfile:
        json.dump(metadata_train, outfile)

    # Save preprocessed test data
    path_test_data = valohai.outputs("test/" + dataset + "/").path(
        "preprocessed_test_data_" + dataset + ".npz"
    )
    np.savez_compressed(path_test_data, test_data=test_rez_images)

    metadata_test = {
        "valohai.dataset-versions": ["dataset://" + dataset + "_test/" + exec_id]
    }

    metadata_path = valohai.outputs("test/" + dataset + "/").path(
        "preprocessed_test_data_" + dataset + ".npz.metadata.json"
    )
    with open(metadata_path, "w") as outfile:
        json.dump(metadata_test, outfile)

    print("Save completed")

print(json.dumps({"run_training_for": dataset_names}))
