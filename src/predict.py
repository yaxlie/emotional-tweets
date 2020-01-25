from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import BatchLoader
import pandas

def predict_to_csv(model, vectorizer, data, dest='prediction.csv', clean=False):
    batch_loader = BatchLoader(data, vectorizer, has_labels=False, clean=clean)
    with batch_loader as batch:
        predictions = model.predict(batch.features)

        df = pandas.DataFrame(predictions, columns=['Category'], index=batch.ids)
        df.index.name = 'Id'

        df.to_csv (dest, index = True, header=True)

    print("Predictions exported to {}.".format(dest))