from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import random
import argparse
import tensorflow as tf

from tqdm import tqdm

from entity_networks.inputs import generate_input_fn

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"

pickle_path = _DIR + "data.pkl"
metadata_path = _DIR + "tf/tfmetadata.json"
instances_path = _DIR + "tf/tfinstances.json"
partition_path = _DIR + "storyid_partition.txt"
annotation_path = _DIR + "json_version/annotations.json" 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='Directory containing data',
        default=_DIR)
    
    args = parser.parse_args()

    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    filename = os.path.join(_DIR, 'tf/commonsense_test.tfrecords')
    input_fn = generate_input_fn(
        filename=filename,
        metadata=metadata,
        batch_size=BATCH_SIZE, # NOTE : wut
        num_epochs=1,
        shuffle=False)

    with tf.Graph().as_default():
        features, answer = input_fn()

        story = features['story']

        instances = []

        with tf.train.SingularMonitoredSession() as sess:
            while not sess.should_stop():
                story_, answer_ = sess.run([story, answer])

                instance = {
                    'story': story_[0].tolist(),
                    'answer': answer_[0].tolist(),
                }

                instances.append(instance)

        metadata['instances'] = random.sample(instances, k=10)

        with open(instances_path, 'w') as f:
            f.write(json.dumps(metadata))

if __name__ == '__main__':
    main()
