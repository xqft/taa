from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import numpy as np


class MetricsManager():
    def __init__(self):
        pass

    def print_metrics(self, y_test, y_pred, weights_test):

        precision = precision_score(y_test, y_pred, average='binary', sample_weight=weights_test)
        recall = recall_score(y_test, y_pred, average='binary', sample_weight=weights_test)
        f1 = f1_score(y_test, y_pred, average='binary', sample_weight=weights_test)
        accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Accuracy Score:", accuracy)

    def print_confusion_matrix(self, y_test, y_pred, weights_test):

        conf_matrix = confusion_matrix(y_test, y_pred, sample_weight=weights_test)

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(fpr, tpr, label=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate (Fall-Out)', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curve', fontsize=16)
        if label is not None:
            plt.legend(loc="lower right")
        plt.grid(True)

    def plot_precision_recall_vs_threshold(y_train, y_scores, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(y_train, y_scores, sample_weight=sample_weight)

        fig, ax = plt.subplots()
        plt.plot(thresholds, precision[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recall[:-1], "g-", label="Recall", linewidth=2)

        plt.title('Precision-Recall Curve')
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.legend()

        plt.show()

    def get_best_threshold(precision, recall, thresholds):

        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)

        max_f1_index = np.argmax(f1_scores)
        best_threshold = thresholds[max_f1_index]

        return best_threshold
