from utils import BatchLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

        # model = RandomForestClassifier(n_estimators=50, random_state=0)

        model = MLPClassifier(
            hidden_layer_sizes=(100,100,100), 
            max_iter=100,
            # n_iter_no_change=200, 
            alpha=0.0001, 
            activation='relu',
            # learning_rate="adaptive",
            solver='adam', 
            verbose=10,  
            random_state=21,
            tol=0.000000001
        )

        model.fit(X_train, y_train)

        print('Training finished!')

        if test:
            predictions = model.predict(X_test)

            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            print(accuracy_score(y_test, predictions))

        return model, batch_loader.vectorizer