import numpy as np
import pandas as pd

class TrainDataLoader():
    def __init__(self, path):
        self.path = path
        self.data = None

    def __enter__(self):
        try:
            self.data = pd.read_csv(self.path)

            print("Train data loaded.")

            return self.data
        except Exception as e:
            print("Train data could not be loaded: {}".format(e))
        
    def __exit__(self, type, value, traceback):
        pass