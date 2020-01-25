from utils import preprocess_text, lemmatize_text, simple_stemmer, tokenize
import argparse
import nltk

parser = argparse.ArgumentParser(description='Tests')
parser.add_argument('-s', '--sentence', help='words to process', type=str)
args = parser.parse_args()

text = args.sentence

text = preprocess_text(text)
print(text)

text = lemmatize_text(text)
print(text)

# text = simple_stemmer(text)
print(text)

print(nltk.pos_tag(text.split()))

text = tokenize(text)
print(text)
