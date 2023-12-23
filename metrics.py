import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


def get_auc(labels, predictions) -> None:
    """
    Calculate the Area Under the Curve (AUC) for the ROC curve,
    plot the ROC curve, and print the classification report.

    Args:
        labels: True labels.
        predictions: Predicted probabilities.

    Returns:
        None
    """

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, _ = roc_curve(labels, predictions)

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Convert probabilities to binary predictions
    predictions = [1 if p > 0 else -1 for p in predictions]

    # Convert labels and predictions to numpy arrays
    labels = np.asarray(labels, dtype=np.int32)
    predictions = np.asarray(predictions, dtype=np.int32)

    # Print the classification report
    print(classification_report(labels, predictions))
