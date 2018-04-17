"""
Utils
"""
import os
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

from keras import layers
from keras.callbacks import Callback
from keras import backend as K


# ========== embedding utils ==========
# TODO: set your glove data path here
glove_path = os.path.expanduser('~/datasets/glove/')
assert os.path.exists(glove_path)


def load_glove(path=glove_path, embedding_dim=300, corpus_size=6, desired=None, verbose=False):
    """Load glove embeddings from original txt file
    """
    if embedding_dim != 300:
        assert embedding_dim in (50, 100, 200), 'embedding dim must be one of 50/100/200 if not 300'
        fpath = os.path.join(path, 'glove.6B.{}d.txt'.format(embedding_dim))
    else:
        assert corpus_size in (6, 42, 840), 'corpus type must be one of 6B/42B/840B'
        fpath = os.path.join(path, 'glove.{}B.300d.txt'.format(corpus_size))
    word2vec = {}
    print('loading glove from', fpath)
    f = open(fpath, 'r', encoding='utf8', errors='ignore')
    for line in tqdm(f, desc='glove') if verbose else f:
        values = line.split()
        word = values[0]  # the word
        if not desired or word in desired:
            coefs = np.asarray(values[1:], dtype="float32")
            word2vec[word] = coefs
    f.close()
    print('glove info: {} words, {} dims'.format(len(word2vec), embedding_dim))
    return word2vec


def get_embedding_mat(embeddings, word2index, embedding_dim, random_uniform_level=0.01, idx_from=2):
    """Use embeddings and word2index to get embedding-mat (for input layer)
    idx_from=2, usually, 0 for <PAD>, 1 for <OOV>
    """
    # embedding_mat = np.zeros((n_words, embedding_dim))
    n_words = len(word2index)
    for idx in range(0, idx_from):
        if idx in word2index.values():
            n_words -= 1
    n_words += idx_from
    embedding_mat = np.random.uniform(low=-random_uniform_level, high=random_uniform_level, size=(n_words, embedding_dim))
    embedding_mat[0] = np.zeros(embedding_dim)
    for word, idx in word2index.items():
        if idx < idx_from:
            continue
        embedding_vec = embeddings.get(word)
        if embedding_vec is not None:  # means we have this word's embedding
            embedding_mat[idx] = embedding_vec
    return embedding_mat


# ========== keras utils ==========
def att_process(candidates, att, activation='tanh'):
    """
    Attention Process (functional API, can get weights at the same time)
     - candidates: (*, maxlen, features)
     - att: (*, att_dim)
    """
    att_dim = K.int_shape(att)[-1]
    candidates2 = layers.TimeDistributed(
        layers.Dense(att_dim, activation=activation))(candidates)
    dotted = layers.dot([candidates2, att], axes=(2, 1), normalize=True)
    weights = layers.Activation('softmax')(dotted)  # (*, maxlen), sums up to 1
    weighted = layers.dot([candidates, weights], axes=(1, 1))
    return weighted, weights


class UpdateMonitor(Callback):
    """monitor a model's training process:
    monitor each layer's update rate (~1e-3 is good rate)
    """
    def __init__(self):
        super(UpdateMonitor, self).__init__()
        self.weights = None

    @classmethod
    def _get_updates(cls, old_weights, new_weights):
        """Calculate updates rate for layers' weights
        Note: only calculate the first parameter of a layer"""
        if not old_weights:
            old_weights = new_weights
        updates = []
        for old_layerwise_weights, new_layerwise_weights in zip(old_weights, new_weights):
            if len(old_layerwise_weights) == 0 or len(new_layerwise_weights) == 0:
                updates.append(None)
            else:
                w1, w2 = old_layerwise_weights[0], new_layerwise_weights[0]  # only check the first weight of a layer
                updates.append(norm(w2 - w1) / norm(w2))
        return updates

    def on_epoch_end(self, epoch, logs={}):
        # monitor update rates
        new_weights = _get_weights(self.model)
        updates = self._get_updates(old_weights=self.weights, new_weights=new_weights)
        self.weights = new_weights  # update
        updates_info = ', '.join('{:.4f}'.format(1e3 * update) if update else '-' for update in updates)
        print('- updates: 1e-3 * [{}]'.format(updates_info))


def _get_weights(model):
    """Get all layers' weights as a list of list:
    [[l1_w1, l1_w2, ...], ... , [ln_w1, ln_w2, ...]]"""
    weights = []
    for layer in model.layers:
        # if no weights, return value is []
        weights.append(layer.get_weights())
    return weights
