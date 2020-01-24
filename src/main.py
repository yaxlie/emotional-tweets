import argparse
from utils import BatchLoader

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
args = parser.parse_args()

batch_loader = BatchLoader(args.train_data)
with batch_loader as batch:
    print(batch.features)