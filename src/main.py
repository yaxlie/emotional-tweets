import argparse
from train import train
from test import test
from predict import predict_to_csv

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
parser.add_argument('-t', '--test-data', help='path to testing .csv data', type=str)
parser.add_argument('-p', '--predict-data', help='path to .csv file to predict', type=str)
parser.add_argument('-o', '--output', help='output file', type=str)
args = parser.parse_args()

model, vectorizer = train(args.train_data)

if args.test_data:
    test(model, vectorizer, args.test_data)

if args.predict_data:
    dest = args.output if args.output else 'predictions.csv'
    predict_to_csv(model, vectorizer, args.predict_data, dest)