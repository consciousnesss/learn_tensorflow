from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os


def main():
    # Data sets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = os.path.expanduser("~/Downloads/iris_training.csv"),  # http://download.tensorflow.org/data/iris_training.csv
        target_dtype=np.int,
        features_dtype=np.float32
    )

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = os.path.expanduser("~/Downloads/iris_test.csv"),  # http://download.tensorflow.org/data/iris_test.csv
        target_dtype=np.int,
        features_dtype=np.float32
    )

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/tmp/iris_model")

    classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

    accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

    print('Accuracy: %.2f' % accuracy_score)

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = list(classifier.predict(new_samples))
    print('Predictions: {}'.format(str(y)))



if __name__ == '__main__':
    main()
