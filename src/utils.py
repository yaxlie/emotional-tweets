import numpy as np
import pandas as pd
import nltk
import re
from collections import namedtuple
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Batch():
     def __init__(self, features, labels):
        self.features = features
        self.labels = labels


class BatchLoader():
    """
    Inspired by https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
    """

    def __init__(self, path):
        self.path = path
        self.batch = None

    def __enter__(self):
        data = None
        try:
            data = pd.read_csv(self.path)
            print("Train data loaded.")
        except Exception as e:
            print("Train data could not be loaded: {}".format(e))

        if data is not None:
            features = data.iloc[:, 2].values
            labels = data.iloc[:, 1].values

        self.batch = Batch(features, labels)

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

        vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        self.batch.features = vectorizer.fit_transform(self.batch.features).toarray()