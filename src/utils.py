import numpy as np
import pandas as pd
import nltk
import re
from collections import namedtuple
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class Batch():
     def __init__(self, ids, features, labels):
        self.ids = ids
        self.features = features
        self.labels = labels


class BatchLoader():
    """
    Inspired by https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
    """

    def __init__(self, path, vectorizer=None, has_labels=True, clean=True):
        self.path = path
        self.vectorizer = vectorizer # TODO: inheritance
        self.has_labels = has_labels
        self.clean = clean
        self.batch = None

    def __enter__(self):
        data = None
        labels = None

        try:
            data = pd.read_csv(self.path)

            if self.clean:
                # Files can be corrupted - clean it
                data['Id']=pd.to_numeric(data['Id'],errors='coerce')
                data.dropna(inplace=True)
                data=data[data.Tweet.str.contains("Not Available") == False]

                # balance data
                # data = data.groupby('Category')
                # data = data.apply(lambda x: x.sample(data.size().min()).reset_index(drop=True))

            print("Data loaded.")
        except Exception as e:
            print("Data could not be loaded: {}".format(e))

        if data is not None:
            ids = data.iloc[:, 0].values

            if self.has_labels:
                features = data.iloc[:, 2].values
                labels = data.iloc[:, 1].values
            else:
                features = data.iloc[:, 1].values

        self.batch = Batch(ids, features, labels)

        self.__normalize_features()
        self.__convert_features()

        return self.batch
        
    def __exit__(self, type, value, traceback):
        pass
    
    def __normalize_features(self):
        texts = []

        for sentence in range(0, len(self.batch.features)):
            text = str(self.batch.features[sentence])
            texts.append(process_text(text))

        self.batch.features = texts

    def __convert_features(self):
        """
        Sentences have to be written in math.
        Therefore use TF-IDF notation
        """

        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

        if self.vectorizer:
            self.batch.features = self.vectorizer.transform(self.batch.features).toarray()
        else:
            # vectorizer = CountVectorizer(max_features=5000)
            vectorizer = TfidfVectorizer (max_features=2000, stop_words=stopwords.words('english'))
            self.batch.features = vectorizer.fit_transform(self.batch.features).toarray()

def process_text(text):
    # Remove all links
    text = preprocess_text(text)

    text = lemmatize_text(text)
    # text = simple_stemmer(text)

    text = tokenize(text)

    return text

def preprocess_text(text):
    text = re.sub(r'http[^\s]+', ' ', text)

    # Remove all mentions
    text = re.sub(r'@[^\s]+', ' ', text)

    # # Remove all the special characters
    # text = re.sub(r'\W', ' ', text)

    # # remove all single characters
    # text= re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # # Remove single characters from the start
    # text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # # Removing prefixed 'b'
    # text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()

    return text

def tokenize(text):
    # text = word_tokenize(text)
    text = nltk.pos_tag(text.split())

    # see https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/ VBD
    text = [word[0] for word in text if word[1] in ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'UH', 'VBG']]
    return ' '.join(text)
    # only (RB)adverbs and (JJ)adjectives and VBP NN

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text)

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text