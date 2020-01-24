from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import BatchLoader

def test(model, data):
    batch_loader = BatchLoader(data)
    with batch_loader as batch:
        predictions = model.predict(batch.features)

        print(confusion_matrix(batch.labels, predictions))
        print(classification_report(batch.labels, predictions))
        print(accuracy_score(batch.labels, predictions))