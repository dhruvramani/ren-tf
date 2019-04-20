"""
Module responsible for input data.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_input_fn(filename, metadata, batch_size, num_epochs=None, shuffle=False):
    "Return _input_fn for use with Experiment."
    def _input_fn():
        max_sentence_length = metadata['max_sentence_length']
        max_word_length = metadata['max_word_length']
        labels_dim = metadata['labels_dim']
        mask_dim = metadata['mask_dim']
        embedding_dim = metadata['embedding_dim']

        with tf.device('/cpu:0'):
            story_feature = tf.FixedLenFeature(
                shape=[max_sentence_length, max_word_length, embedding_dim],
                dtype=tf.float32)

            labels_feature = tf.FixedLenFeature(
                shape=[mask_dim, labels_dim],
                dtype=tf.float32)

            mask_feature = tf.FixedLenFeature(shape=[mask_dim], dtype=tf.float32)

            features = {
                'story': story_feature, 
                'labels': labels_feature,
                'mask': mask_feature
            }

            record_features = tf.contrib.learn.read_batch_record_features(
                file_pattern=filename,
                features=features,
                batch_size=batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = record_features['story']
            labels = record_features['labels']
            mask = record_features['mask']

            features = {
                'story': story,
                'mask': mask
            }

            return features, labels

    return _input_fn
