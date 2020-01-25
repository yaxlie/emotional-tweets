import argparse
from train import train
from predict import predict_to_csv

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
parser.add_argument('-t', '--test', help='do you want to split the data to test sets', type=bool, default=False)
parser.add_argument('-p', '--predict-data', help='path to .csv file to predict', type=str)
parser.add_argument('-o', '--output', help='output file', type=str)
args = parser.parse_args()

model, vectorizer = train(args.train_data, args.test)

if args.predict_data:
    dest = args.output if args.output else 'predictions.csv'
    predict_to_csv(model, vectorizer, args.predict_data, dest)