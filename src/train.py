from utils import BatchLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train(data):
    batch_loader = BatchLoader(data)
    with batch_loader as batch:
        print('Preparing data...')
        X_train, X_test, y_train, y_test = train_test_split(batch.features, batch.labels, test_size=0.2, random_state=0)

        print('Train...')
        text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        text_classifier.fit(X_train, y_train)

        print('Training finished!')
        
        return text_classifier