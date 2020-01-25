from utils import BatchLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train(data, test=False):
    batch_loader = BatchLoader(data)
    with batch_loader as batch:
        print('Preparing data...')

        if test:
            X_train, X_test, y_train, y_test = train_test_split(batch.features, batch.labels, test_size=0.2, random_state=0)
        else:
            X_train = batch.features
            y_train = batch.labels

        print('Train...')
        text_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        text_classifier.fit(X_train, y_train)

        print('Training finished!')

        if test:
            predictions = text_classifier.predict(X_test)

            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            print(accuracy_score(y_test, predictions))

        return text_classifier, batch_loader.vectorizer