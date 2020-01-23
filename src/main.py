import argparse
from utils import TrainDataLoader

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
args = parser.parse_args()

train_data = TrainDataLoader(args.train_data)

with train_data as data:
    print(train_data)