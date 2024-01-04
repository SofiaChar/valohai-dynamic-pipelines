import numpy as np
import random
from PIL import Image
import os
from utils.model import load_model
import valohai

# Define that data and model paths
dataset_names = valohai.parameters("dataset_names").value

# Possible ship categories
category = {"Cargo": 1, "Military": 2, "Carrier": 3, "Cruise": 4, "Tankers": 5}

for dataset in dataset_names:
    print("Running predictions for model trained with dataset: " + dataset)
    testset_data_path = valohai.inputs("test_dataset").path(f"test/{dataset}/*")
    model_paths_all = valohai.inputs("model").paths()

    # Run predictions for all models provided as inputs
    for model_path in model_paths_all:
        if dataset in model_path:
            model = load_model(model_path)
            model_filename = os.path.basename(model_path)
            model_name = model_filename.rstrip(".h5")

    with np.load(testset_data_path, allow_pickle=True) as f:
        test_data = f["test_data"]

    predictions = model.predict(test_data)

    # Pick 3 random images from test set to save with the predicted cateogory
    test_img = []
    for k in range(0, 3):
        y = random.randrange(len(test_data))
        test_img.append(y)

    # Save images and predictions
    for i in test_img:
        img = Image.fromarray(test_data[i], "RGB")

        for key in category:
            if category[key] == np.argmax(predictions[i]) + 1:
                print(f"Predicted ship type: {key}")
                im_path = f"predictions/{model_name}/img{i}_{key}.png"

                img.save(valohai.outputs().path(im_path))
