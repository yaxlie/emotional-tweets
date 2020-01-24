import argparse
from train import train

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
args = parser.parse_args()

model = train(args.train_data)