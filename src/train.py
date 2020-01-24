from utils import BatchLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train(data):
    batch_loader = BatchLoader(data)
    with batch_loader as batch:
        print('Preparing data...')

        print('Train...')
        text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        text_classifier.fit(batch.features, batch.labels)

        print('Training finished!')

        return text_classifier