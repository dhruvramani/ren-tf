import re
import os
import json
import bcolz
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import OrderedDict


#TODO :
# + Write code for getting and parsing test data
# + Fix rest of the files to use this data format

_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/data/commonsense/"
_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
_EMB_DIM = 100
_MAX_WLEN = 18
_VOCAB = -1 # Filled when function is called

pickle_path = _DIR + "data.pkl"
metadata_path = _DIR + "tf/tfmetadata.json"
instances_path = _DIR + "tf/tfinstances.json"
partition_path = _DIR + "storyid_partition.txt"
annotation_path = _DIR + "json_version/annotations.json" 
dataset_path_test = _DIR + "tf/commonsense_test.tfrecords"
dataset_path_train = _DIR + "tf/commonsense_train.tfrecords"


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
    global _VOCAB
    vectors = bcolz.open('{}/27B.{}.dat'.format(_GLOVE_PATH, _EMB_DIM))[:]
    words = pickle.load(open('{}/27B.{}_words.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))
    word2idx = pickle.load(open('{}/27B.{}_idx.pkl'.format(_GLOVE_PATH, _EMB_DIM), 'rb'))

    if(_VOCAB == -1):
        _VOCAB = len(words)

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

def create_dataset(load=True, data_type="test"):
    global annotation_path, partition_path
    data_type = "valid" if data_type == "train" else data_type
    annotation_file = open(annotation_path, "r")
    raw_data = json.load(annotation_file, object_pairs_hook=OrderedDict)
    glove = load_glove()

    text_arr, all_labels, char_arr, mask_arr = [], [], [], []
    stories_dat = []
    with open(partition_path, "r") as partition_file:
        for line in partition_file:
            id_key = line.split("\t")[0]
            story = raw_data[id_key]
            
            if(story["partition"] != data_type):
                continue 

            sentences = story["lines"]
            characters = sentences['1']["characters"]
            
            s_dim, c_dim, count = len(sentences.keys()), len(characters.keys()), 0
            mask_dim = s_dim * c_dim
            embeddings, labels, mask = [], [], [0] * mask_dim

            for si in range(s_dim):
                sent = sentences[str(si + 1)]
                text = sent["text"]
                
                embed_string = re.sub(r"[^a-zA-Z]+", ' ', text)
                embedding = [glove.get(word, glove['unk']) for word in embed_string.split(" ")]
                embeddings.append(embedding)
                
                charecs = list(sent["characters"].keys())
                labels.append([])
                for cj in range(c_dim):
                    char = sent["characters"][charecs[cj]]
                    one_hot = get_labels(char)
                    labels[si].append([])
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
            
            # OR - decide
            #stories_dat.append((embeddings, labels, mask, c_dim))

    return text_arr, all_labels, mask_arr, char_arr # stories_dat # - ALL ARE LISTS

def save_dataset(text_arr, all_labels, mask_arr, path):
    """
    Save the stories into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.io.TFRecordWriter(path)
    for i in range(text_arr.shape[0]):
        story_feature = tf.train.Feature(float_list=tf.train.FloatList(value=text_arr[i]))
        labels_feature = tf.train.Feature(float_list=tf.train.FloatList(value=all_labels[i]))
        mask_feature = tf.train.Feature(float_list=tf.train.FloatList(value=mask_arr[i]))

        features = tf.train.Features(feature={
            'story': story_feature,
            'labels': labels_feature,
            'mask': mask_feature,
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    '''
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
    '''

def pad_stories(text_arr, all_labels, mask_arr, max_sentence_length, max_word_length, max_char_length):
    
    for i in range(len(text_arr)):
        story = text_arr[i]
        shape = story.shape
        sentence_pad = max_sentence_length - shape[0]
        # FML
        new_story = []
        story_type = type(story[0])
        if(story_type != list):
            for j in range(shape[0]):
                a = story[j].tolist()
                new_story.append(a)
        for j in range(shape[0]):
            if(len(story[j]) != max_word_length):
                if(story_type != list):
                    a = new_story[j]
                    for i in range(max_word_length - len(a)):
                        a.append([0] * _EMB_DIM)
                    new_story[j] = np.asarray(a)
                else :
                    story[j] = story[j] + [[0] * _EMB_DIM] * (max_word_length - len(list(story[j])))
                    story[j] = np.asarray(story[j])
        if(story_type != list):
            story = np.asarray([new_story[k] for k in range(shape[0])])
        else:
            story = np.asarray([story[k] for k in range(shape[0])])
        if(sentence_pad != 0):
            text_arr[i] = np.pad(story, ((0, sentence_pad), (0, 0), (0, 0)), 'constant')
        else :
            text_arr[i] = story

    for i in range(len(text_arr)):
        if(text_arr[i].shape[1] != max_word_length):
            pad_length = max_word_length - text_arr[i].shape[1]
            text_arr[i] = np.pad(text_arr[i], ((0, 0), (0, pad_length), (0,0)), 'constant')
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        shape = label.shape
        pad_length = max_sentence_length * max_char_length - shape[0]
        all_labels[i] = np.pad(label, ((0, pad_length), (0,0)), 'constant')

    for i in range(len(mask_arr)):
        mask = mask_arr[i]
        shape = mask.shape
        pad_length = max_sentence_length * max_char_length - shape[0]
        mask_arr[i] = np.pad(mask, ((0, pad_length)), 'constant')

    mask_arr = np.asarray(mask_arr)     # Shape : [max_sentence_length, s_d * c_d]
    text_arr = np.asarray(text_arr)     # Shape : [max_sentence_length, max_word_length, embedding_dim]
    all_labels = np.asarray(all_labels) # Shape : [max_sentence_length, s_d * c_d, labels_dim]

    return text_arr, all_labels, mask_arr


def main():
    # init_glove() # RUN ONLY FOR THE FIRST TIME 
    if not os.path.exists(_DIR + 'tf/'):
        os.makedirs(_DIR + 'tf/')

    dataset_path_train = os.path.join(_DIR, 'tf/commonsense_train.tfrecords')
    #dataset_path_test = os.path.join(_DIR,  'tf/commonsense_test.tfrecords')

    text_arr, all_labels, mask_arr, char_arr = create_dataset()
    
    dataset_size = len(text_arr)
    sentence_lengths = [story.shape[0] for story in text_arr]
    word_lengths = [len(story[ss]) for story in text_arr for ss in range(story.shape[0])]

    max_sentence_length = max(sentence_lengths)
    max_word_length = max(word_lengths)
    max_char_length = max(char_arr)

    print(max_sentence_length, max_word_length, max_char_length)
    mask_dim, labels_dim = len(all_labels[0]), len(all_labels[0][0])
    embedding_dim = _EMB_DIM

    with open(metadata_path, 'w') as f:
        metadata = {
            'max_char_length': max_char_length,
            'max_word_length': max_word_length,
            'max_sentence_length': max_sentence_length,
            'mask_dim' : mask_dim,
            'labels_dim' : labels_dim,
            'emedding_dim' : embedding_dim,
            'vocab_size': _VOCAB,
            'dataset_size': dataset_size,
            'filenames': {
                'train': os.path.basename(dataset_path_train),
                'test': os.path.basename(dataset_path_test),
            }
        }
        json.dump(metadata, f)

    text_arr, all_labels, mask_arr = pad_stories(text_arr, all_labels, mask_arr, max_sentence_length, max_word_length, max_char_length)
    #stories_train = (text_arr, all_labels, mask_arr)
    print(text_arr.shape, all_labels.shape, mask_arr.shape)
    save_dataset(text_arr, all_labels, mask_arr, dataset_path_train)

if __name__ == '__main__':
    main()

