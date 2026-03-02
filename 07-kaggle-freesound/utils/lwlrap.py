import numpy as np
import tensorflow as tf
import sklearn.metrics

# Core calculation of label precisions for one test sample.

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (
        1 + class_rankings[pos_class_indices].astype(float)
    )
    return pos_class_indices, precision_at_hits

def _one_sample_positive_class_precisions_tf(batch):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """

    truth, scores = batch

    # Retrieval list of classes for this sample.
    retrieved_classes = tf.argsort(scores, direction="DESCENDING")
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = tf.argsort(retrieved_classes)
    # Which of these is a true label?
    retrieved_class_true = tf.gather(truth, retrieved_classes)
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))
    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    idx = tf.where(truth)[:, 0]
    i = tf.boolean_mask(class_rankings, truth)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = tf.math.add(tf.constant(1, dtype=tf.float32), tf.cast(i, tf.float32))
    precisions = r / c

    dense = tf.scatter_nd(idx[:, None], precisions, [scores.shape[0]])
    return dense


# All-in-one calculation of per-class lwlrap.


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(
            scores[sample_num, :], truth[sample_num, :]
        )
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits
        )
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(
        1, labels_per_class
    )
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# Calculate the overall lwlrap using sklearn.metrics function.


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices],
    )
    return overall_lwlrap


# Accumulator object version.

@tf.keras.utils.register_keras_serializable()
class LwLrap(tf.keras.metrics.Metric):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self, dtype=None, num_classes=80, name="lwlrap"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self._per_class_cumulative_precision = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros',
        )

        self._per_class_cumulative_count = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros',
        )

    def update_state(self, batch_truth, batch_scores, sample_weight=None):
        precisions = tf.map_fn(
            fn=_one_sample_positive_class_precisions_tf,
            elems=(batch_truth, batch_scores),
            fn_output_signature=tf.float32,
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._per_class_cumulative_precision.assign_add(total_precisions)
        self._per_class_cumulative_count.assign_add(total_increments)

    def result(self):
        per_class_lwlrap = self._per_class_cumulative_precision / tf.maximum(self._per_class_cumulative_count, 1.0)
        per_class_weight = self._per_class_cumulative_count / tf.reduce_sum(self._per_class_cumulative_count)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_state(self):
        self._per_class_cumulative_precision.assign(self._per_class_cumulative_precision * 0)
        self._per_class_cumulative_count.assign(self._per_class_cumulative_count * 0)
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_classes": self.num_classes}
