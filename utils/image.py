import cv2
import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class ImagePreprocessing:
    def __init__(self, train_images, test_images, height, length, dataframe):
        self.train_images = train_images
        self.test_images = test_images
        self.height = height
        self.length = length
        self.dataframe = dataframe

    def Resize(self, TAG):
        processed_images = []
        if TAG == "Train":
            for i in range(len(self.train_images)):
                im = cv2.resize(self.train_images[i], dsize=(self.height, self.height))
                processed_images.append(im)
        elif TAG == "Test":
            for i in range(len(self.test_images)):
                im = cv2.resize(self.test_images[i], dsize=(self.height, self.height))
                processed_images.append(im)
        return processed_images

    def Reshape(self):
        # resizing Images
        self.rez_train_image = self.Resize("Train")
        self.rez_test_image = self.Resize("Test")
        # fetching labels for training and testing
        self.train_labels = self.dataframe["category"][: self.length]

        # converting into array
        self.label_array = self.toarray(self.train_labels)

        # reshaping label array
        self.labels = self.label_array.reshape(len(self.label_array), 1)

        # reshaping images
        self.pro_images = np.reshape(
            self.rez_train_image,
            (len(self.rez_train_image), self.height, self.height, 3),
        )
        self.test_pro_images = np.reshape(
            self.rez_test_image, (len(self.rez_test_image), self.height, self.height, 3)
        )

        return self.pro_images, self.labels, self.test_pro_images

    def toarray(self, series):
        return np.array(series)

    def splitdata(self, TRAIN_images, LABELS, val_size):
        X_train, X_val, Y_train, Y_val = train_test_split(
            TRAIN_images, LABELS, test_size=val_size, random_state=42
        )
        return X_train, X_val, Y_train, Y_val

    def OneHot(self, x):
        columnTransformer = ColumnTransformer(
            [("encoder", OneHotEncoder(categories="auto"), [0])],
            remainder="passthrough",
        )
        x = columnTransformer.fit_transform(x).toarray()
        return x
