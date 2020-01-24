import numpy as np
import pandas as pd
from collections import namedtuple

class Batch():
     def __init__(self, features, labels):
        self.features = features
        self.labels = labels


class BatchLoader():
    def __init__(self, path):
        self.path = path
        self.batch = None

    def __enter__(self):
        data = None
        try:
            data = pd.read_csv(self.path)
            print("Train data loaded.")
        except Exception as e:
            print("Train data could not be loaded: {}".format(e))

        if data is not None:
            features = data.iloc[:, 2].values
            labels = data.iloc[:, 1].values

        batch = Batch(features, labels)

        self.batch = batch
        return self.batch
        
    def __exit__(self, type, value, traceback):
        pass