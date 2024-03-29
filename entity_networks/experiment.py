from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

from entity_networks.hooks import EarlyStoppingHook
from entity_networks.inputs import generate_input_fn
from entity_networks.serving import generate_serving_input_fn
from entity_networks.model import model_fn

BATCH_SIZE = 32
NUM_BLOCKS = 20
EMBEDDING_SIZE = 100
CLIP_GRADIENTS = 40.0

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"

pickle_path = _DIR + "data.pkl"
metadata_path = _DIR + "tf/tfmetadata.json"
instances_path = _DIR + "tf/tfinstances.json"
partition_path = _DIR + "storyid_partition.txt"
annotation_path = _DIR + "json_version/annotations.json"
dataset_path_test = _DIR + "tf/commonsense_test.tfrecords"
dataset_path_train = _DIR + "tf/commonsense_train.tfrecords" 

def generate_experiment_fn(data_dir, dataset_id, num_epochs,
                           learning_rate_min, learning_rate_max,
                           learning_rate_step_size, gradient_noise_scale):
    "Return _experiment_fn for use with learn_runner."
    def _experiment_fn(output_dir):
        with tf.gfile.Open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

        #train_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'train'))
        #eval_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'test'))

        train_input_fn = generate_input_fn(
            filename=dataset_path_train,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=num_epochs,
            shuffle=True)

        
        eval_input_fn = generate_input_fn(
            filename=dataset_path_test,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        vocab_size = metadata['vocab_size']
        embedding_dim = metadata['embedding_dim']
        task_size = metadata['dataset_size'] # NOTE : DO SOMETHING ABOUT THIS
        train_steps_per_epoch = task_size // BATCH_SIZE

        run_config = tf.contrib.learn.RunConfig(
            save_summary_steps=train_steps_per_epoch,
            save_checkpoints_steps=5 * train_steps_per_epoch,
            save_checkpoints_secs=None)

        params = {
            'vocab_size': vocab_size,
            'embedding_size': embedding_dim,
            'labels_dim' : metadata['labels_dim'],
            'num_blocks': NUM_BLOCKS,
            'learning_rate_min': learning_rate_min,
            'learning_rate_max': learning_rate_max,
            'learning_rate_step_size': learning_rate_step_size * train_steps_per_epoch,
            'clip_gradients': CLIP_GRADIENTS,
            'gradient_noise_scale': gradient_noise_scale,
        }

        estimator = tf.contrib.learn.Estimator(
            model_dir=output_dir,
            model_fn=model_fn,
            config=run_config,
            params=params)

        # TODO : Put precision and recall
        eval_metrics = {
            'f1': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.f1_score)
        }

        train_monitors = [
            EarlyStoppingHook(
                input_fn=eval_input_fn,
                estimator=estimator,
                metrics=eval_metrics,
                metric_name='f1',
                every_steps=5 * train_steps_per_epoch,
                max_patience=50 * train_steps_per_epoch,
                minimize=False)
        ]

        serving_input_fn = generate_serving_input_fn(metadata)
        export_strategy = tf.contrib.learn.utils.make_export_strategy(
            serving_input_fn)

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_monitors=train_monitors,
            train_steps=None,
            eval_steps=None,
            export_strategies=[export_strategy],
            min_eval_frequency=100)
        return experiment

    return _experiment_fn
