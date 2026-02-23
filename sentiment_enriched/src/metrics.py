from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def compute_metrics(p):

    predictions, labels = p

    predictions = predictions.argmax(axis=-1)
    cm = confusion_matrix(labels, predictions)

    print("Confusion Matrix:\n", cm)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }