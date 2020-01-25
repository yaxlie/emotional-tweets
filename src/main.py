import argparse
from train import train
from predict import predict_to_csv
import pickle

SAVED_MODEL = 'model.p' # TODO: as arg

parser = argparse.ArgumentParser(description='Predict the emotion related to the tweet.')
parser.add_argument('-d', '--train-data', help='path to training .csv data', type=str)
parser.add_argument('-t', '--test', help='do you want to split the data to test sets', type=bool, default=False)
parser.add_argument('-p', '--predict-data', help='path to .csv file to predict', type=str)
parser.add_argument('-o', '--output', help='output file', type=str)
parser.add_argument('-c', '--clean', help='clean data', type=bool, default=False)

args = parser.parse_args()

if args.train_data:
    model, vectorizer = train(args.train_data, args.test)
    pickle.dump((model, vectorizer), open(SAVED_MODEL, "wb"))
else:
    model, vectorizer = pickle.load(open(SAVED_MODEL, "rb"))

if args.predict_data:
    dest = args.output if args.output else 'predictions.csv'
    predict_to_csv(model, vectorizer, args.predict_data, dest, args.clean)
