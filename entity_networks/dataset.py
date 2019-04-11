import re
import os
import json
import bcolz
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import OrderedDict

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/"
_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
_EMB_DIM = 100
_MAX_WLEN = 18
_STEP_SIZE = 

pickle_path = _DIR + "data/commonsense/data.pkl"
partition_path = _DIR + "data/commonsense/storyid_partition.txt"
annotation_path = _DIR + "data/commonsense/json_version/annotations.json" 

classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

def init_glove(glove_path=_GLOVE_PATH): # Run only first time
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/27B.{}d.dat'.format(glove_path, _EMB_DIM), mode='w')
    with open('{}/glove.twitter.27B.{}d.txt'.format(glove_path, _EMB_DIM), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors.reshape((1193514, _EMB_DIM)), rootdir='{}/27B.{}.dat'.format(glove_path, _EMB_DIM), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/27B.{}_words.pkl'.format(glove_path, _EMB_DIM), 'wb'))
    pickle.dump(word2idx, open('{}/27B.{}_idx.pkl'.format(glove_path, _EMB_DIM), 'wb'))
    return idx


def tokenize(sentence):
    "Tokenize a string by splitting on non-word characters and stripping whitespace."
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

def load_glove():
    vectors = bcolz.open('{}/27B.{}.dat'.format(_GLOVE_PATH, _EMB_DIM))[:]
    words = pickle.load(open('{}/27B.{}_words.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))
    word2idx = pickle.load(open('{}/27B.{}_idx.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))

    return {w: vectors[word2idx[w]] for w in words}

def get_labels(charay):
        ann = []
        for i in range(3):
            try :
                ann.append(charay["emotion"]["ann{}".format(i)]["plutchik"])
            except:
                print("ann{} ignored".format(i))

        if(len(ann) == 0): # NOTE - change this maybe
            return [0 for _ in classes]

        final_dict = dict()
        for classi in classes:
            final_dict[classi] = [1, 1, 1]

        for idx in range(len(ann)):
            for i in ann[idx]:
                if(i[:-2] in final_dict.keys()):
                    final_dict[i[:-2]][idx] = int(i[-1])

        majority = []
        for key in final_dict.keys():
            if(int(sum(final_dict[key]) / 3) >= 2):
                majority.append(key) #[key if(floor(sum(final_dict[key]) / 3) >= 2) for key in final_dict.keys()]

        onehot = [1 if i in majority else 0 for i in classes]
        return onehot

def create_dataset(load=True, data_type="train"):
    data_type = data_type.replace("train", "valid")
    annotation_file = open(annotation_path, "r")
    raw_data = json.load(annotation_file, object_pairs_hook=OrderedDict)
    glove = load_glove()

    text_arr, all_labels, char_arr, mask_arr = [], [], [], []
    with open(partition_path, "r") as partition_file:
        for line in partition_file:
            id_key = line.split("\t")[0]
            story = raw_data[id_key]
            
            if(story["partition"] != data_type):
                continue 

            sentences = story["lines"]
            characters = sentences[0]["characters"]
            
            s_dim, c_dim, count = len(sentences.keys()), len(characters.keys()), 0
            mask_dim = s_dim * c_dim
            embeddings, labels, mask = [], [], [0] * mask_dim

            for si in range(s_dim):
                sent = sentences[str(si)]
                text = sent["text"]
                
                embed_string = re.sub(r"[^a-zA-Z]+", ' ', text)
                embedding = [glove.get(word, glove['unk']) for word in embed_string.split(" ")]
                embeddings.append(embedding)

                charecs = sent["characters"].keys()
                labels[si] = []

                for cj in range(c_dim):
                    char = sent["characters"][charecs[cj]]
                    one_hot = get_labels(char)
                    labels[si][cj] = one_hot
                    if(1 in one_hot):
                        mask[count] = 1
                    count += 1

            mask = np.asarray(mask)
            labels = np.asarray(labels)
            embeddings = np.asarray(embeddings)
            labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
            
            all_labels.append(labels)   # Shape : [stories, s_d * c_d, labels_dim]
            mask_arr.append(mask)       # Shape : [stories, s_d * c_d]
            text_arr.append(embeddings) # Shape : [stories, s_d, words, embedding_dim]
            char_arr.append(c_dim)      # Shape : [stories, 1]  - No. of chars. - to find upper bound

    return text_arr, all_labels, mask_arr, char_arr


# Functions from TF implementation

def save_dataset(stories, path):
    """
    Save the stories into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.python_io.TFRecordWriter(path)
    for story, query, answer in stories:
        story_flat = [token_id for sentence in story for token_id in sentence]

        story_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=story_flat))
        query_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=query))
        answer_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[answer]))

        features = tf.train.Features(feature={
            'story': story_feature,
            'query': query_feature,
            'answer': answer_feature,
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
