"""
Serving input function definition.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_serving_input_fn(metadata):
    "Returns _serving_input_fn for use with an export strategy."
    max_sentence_length = metadata['max_sentence_length']
    max_word_length = metadata['max_word_length']
    embedding_dim = metadata['embedding_dim']
    mask_dim = metadata['mask_dim']
    labels_dim = metadata['labels_dim']

    def _serving_input_fn():
        story_placeholder = tf.placeholder(
            shape=[max_sentence_length, max_word_length, embedding_dim],
            dtype=tf.float32,
            name='story')
        labels_placeholder = tf.placeholder(
            shape=[mask_dim, labels_dim],
            dtype=tf.float32,
            name='labels')
        mask_placeholder = tf.placeholder(shape=[mask_dim], type=tf.float32, name='mask')

        feature_placeholders = {
            'story': story_placeholder,
            'labels': labels_placeholder,
            'mask': mask_placeholder
        }

        features = {
            key: tf.expand_dims(tensor, axis=0)
            for key, tensor in feature_placeholders.items()
        }

        input_fn_ops = tf.contrib.learn.utils.input_fn_utils.InputFnOps(
            features=features,
            labels=None,
            default_inputs=feature_placeholders)

        return input_fn_ops

    return _serving_input_fn
