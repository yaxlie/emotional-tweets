import argparse
from train import train
from test import test

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
parser.add_argument('-t', '--test-data', help='path to testing .csv data', type=str)
args = parser.parse_args()

model = train(args.train_data)

if args.test_data:
    test(model, args.test_data)