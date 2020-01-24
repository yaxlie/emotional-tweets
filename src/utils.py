import numpy as np
import pandas as pd
import nltk
import re
from collections import namedtuple
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Batch():
     def __init__(self, ids, features, labels):
        self.ids = ids
        self.features = features
        self.labels = labels


class BatchLoader():
    """
    Inspired by https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
    """

    def __init__(self, path, vectorizer=None, has_labels=True):
        self.path = path
        self.vectorizer = vectorizer # TODO: inheritance
        self.has_labels = has_labels
        self.batch = None

    def __enter__(self):
        data = None
        labels = None

        try:
            data = pd.read_csv(self.path)
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
        processed_features = []

        for sentence in range(0, len(self.batch.features)):
            processed_feature = str(self.batch.features[sentence])

            # Remove all links
            processed_feature = re.sub(r'http[^\s]+', ' ', processed_feature)

            # Remove all mentions
            processed_feature = re.sub(r'@[^\s]+', ' ', processed_feature)

            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', processed_feature)

            # remove all single characters
            processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)

            # Converting to Lowercase
            processed_feature = processed_feature.lower()

            processed_features.append(processed_feature)

        self.batch.features = processed_features

    def __convert_features(self):
        """
        Sentences have to be written in math.
        Therefore use TF-IDF notation
        """
        nltk.download('stopwords')

        if self.vectorizer:
            self.batch.features = self.vectorizer.transform(self.batch.features).toarray()
        else:
            vectorizer = TfidfVectorizer (max_features=2500, stop_words=stopwords.words('english'))
            self.batch.features = vectorizer.fit_transform(self.batch.features).toarray()