import numpy as np

class TrainDataLoader():
    def __init__(self, path):
        self.path = path
        self.data = None

    def __enter__(self):
        try:
            self.data = open(self.path)

            print("Train data loaded.")
            
            return self.data
        except Exception as e:
            print("Train data could not be loaded: {}".format(e))
        
    def __exit__(self, type, value, traceback):
        pass