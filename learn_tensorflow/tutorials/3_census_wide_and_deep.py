
import tempfile
import urllib
import os

import pandas as pd
import tensorflow as tf


def main():
    '''
    https://www.tensorflow.org/versions/r1.0/tutorials/wide_and_deep/
    '''
    train_file_name = os.path.expanduser('~/conss/adult.data')
    test_file_name = os.path.expanduser('~/conss/adult.test')
    if not os.path.exists(train_file_name):
        print('donwloading...')
        urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file_name)
    if not os.path.exists(test_file_name):
        print('donwloading...')
        urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file_name)

    print("Preparing features...")
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    with open(train_file_name) as train_file:
        df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

    with open(test_file_name) as test_file:
        df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

    LABEL_COLUMN = "label"
    df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                           "relationship", "race", "gender", "native_country"]
    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

    def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        continuous_cols = {k: tf.constant(df[k].values)
                           for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(continuous_cols.items() + categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)
        # Returns the feature columns and the label.
        return feature_cols, label

    def train_input_fn():
        return input_fn(df_train)

    def eval_input_fn():
        return input_fn(df_test)

    # Categorical base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
    race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
        "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    wide_columns = [
        gender, native_country, education, occupation, workclass, relationship, age_buckets,
        tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]

    deep_columns = [
        tf.contrib.layers.embedding_column(workclass, dimension=8),
        tf.contrib.layers.embedding_column(education, dimension=8),
        tf.contrib.layers.embedding_column(gender, dimension=8),
        tf.contrib.layers.embedding_column(relationship, dimension=8),
        tf.contrib.layers.embedding_column(native_country, dimension=8),
        tf.contrib.layers.embedding_column(occupation, dimension=8),
        age, education_num, capital_gain, capital_loss, hours_per_week]

    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

    m.fit(input_fn=train_input_fn, steps=200)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print "%s: %s" % (key, results[key])


if __name__ == '__main__':
    main()
