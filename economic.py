from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

_CSV_COLUMNS = [
    'state', 'gov_function', 'ft_equiv_empmt', 'ft_equiv_var', 'ttl_oct_payroll', 'oct_payroll_var',
    'ft_empmt', 'ft_var', 'pt_empmt', 'pt_var', 'ft_payroll', 'ft_payroll_var', 'pt_payroll',
    'pt_payroll_var', 'pt_hrs_paid', 'pt_hrs_var', 'growth'

]

_CSV_COLUMN_DEFAULTS = [[''], [''], [0], [0.0], tf.constant([], dtype=tf.int64), [0.0], [0], [0.0], [0], [0.0], tf.constant([], dtype=tf.int64), [0.0], tf.constant([], dtype=tf.int64), [0.0], [0], [0.0],[0]]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='C:/Users/owner/PycharmProjects/finalProject/tmp/census_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=5, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=30, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='C:/Users/owner/PycharmProjects/finalProject/tmp/census_data/govpayrolls_withclass.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str,
    default='C:/Users/owner/PycharmProjects/finalProject/tmp/census_data/govpayrolls_testwithclass.csv',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 3276,
    'validation': 1560,
}


def build_model_columns():
    """Builds a set of wide and deep feature columns."""

    # Continuous columns
    ft_equiv_empmt = tf.feature_column.numeric_column('ft_equiv_emps')
    ft_equiv_var = tf.feature_column.numeric_column('ft_equiv_emps')

    ttl_oct_payroll = tf.feature_column.numeric_column('ttl_oct_payroll')
    oct_payroll_var = tf.feature_column.numeric_column('oct_payroll_var')
    ft_payroll = tf.feature_column.numeric_column('ft_payroll')
    ft_empmt = tf.feature_column.numeric_column('ft_empmt')
    ft_var = tf.feature_column.numeric_column('ft_var')

    pt_empmt = tf.feature_column.numeric_column('pt_empmt')
    pt_var = tf.feature_column.numeric_column('pt_var')
    ft_payroll_var = tf.feature_column.numeric_column('ft_payroll_var')
    pt_payroll = tf.feature_column.numeric_column('pt_payroll')
    pt_payroll_var = tf.feature_column.numeric_column('pt_payroll_var')

    pt_hrs_paid = tf.feature_column.numeric_column('pt_hrs_paid')
    pt_hrs_var = tf.feature_column.numeric_column('pt_hrs_var')

    pt_var_buckets = tf.feature_column.bucketized_column(
        pt_var, boundaries=[0, 1, 5, 10, 15, 20, 25, 50])
    ft_var_buckets = tf.feature_column.bucketized_column(
        ft_var, boundaries=[0, 1, 5, 10, 15, 20, 25, 50])
    ft_payroll_buckets = tf.feature_column.bucketized_column(
        ft_payroll, boundaries=[0, 10000, 1000000, 10000000, 100000000])
    pt_payroll_buckets = tf.feature_column.bucketized_column(
        pt_payroll, boundaries=[0, 10000, 1000000, 10000000, 100000000])
    oct_payroll_buckets = tf.feature_column.bucketized_column(
        ttl_oct_payroll, boundaries=[0, 10000, 1000000, 10000000, 100000000])
    state = tf.feature_column.categorical_column_with_vocabulary_list(
        'state', ['US', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
                  'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM',
                  'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
                  'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
                  ])

    gov_function = tf.feature_column.categorical_column_with_hash_bucket(
        'gov_function', hash_bucket_size=1000)


    # Wide columns and deep columnss
    base_columns = [
         ft_var_buckets, oct_payroll_var, pt_var_buckets, gov_function,ft_payroll_buckets, pt_payroll_buckets, oct_payroll_buckets
    ]

    crossed_columns = [

        tf.feature_column.crossed_column(
            ['ft_var', 'ft_empmt'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['ft_empmt', 'ft_equiv_empmt'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['ft_var', 'gov_function'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        pt_payroll, ttl_oct_payroll, tf.feature_column.indicator_column(ft_payroll_buckets),tf.feature_column.embedding_column(gov_function, dimension=8),
        tf.feature_column.embedding_column(state, dimension=8),

    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)


        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('growth')
        return features, tf.equal(labels, 1)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
