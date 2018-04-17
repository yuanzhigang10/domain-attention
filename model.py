"""
Domain Attention Model
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from nltk import word_tokenize
from collections import Counter

import tensorflow as tf
from keras.models import Model
from keras import layers, callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from data_helper import load_mdsd
from utils import load_glove, get_embedding_mat
from utils import att_process, UpdateMonitor


# for reproducibility
np.random.seed(2333)
tf.set_random_seed(2333)


# domains
DOMAINS = ('books', 'dvd', 'electronics', 'kitchen')


class SharedData:
    """global data"""
    coef = 0.04  # domain loss coefficient in total loss

    # model params
    embed_dim = 300  # word embedding dimension
    rnn_dim = 300  # rnn state dimension
    hidden_dim = 100  # hidden state dimension
    embed_dropout = 0.2  # dropout rate for word embedding layer
    fc_dropout = 0.2  # dropout rate for fully connected layer
    batch_size = 64  # training batch size
    epochs = 100  # maximal training epochs
    activation = 'relu'  # activation function of fully connected layer
    optimizer = 'adadelta'  # optimizer method for training
    RNN = layers.LSTM  # RNN type, can be SimpleRNN/GRU/LSTM/...

    # train params
    lr_factor = 0.1  # reduce factor of learning rate for training plateau
    lr_patience = 3  # reduce learning rate when val loss has stopped improving
    stop_patience = 5  # stop training when val loss has stopped improving

    # data params
    glove_corpus = 6  # glove corpus size, can be 6B/42B/840B
    min_count = 1  # minimal word frequency cutoff
    max_words = None  # maximal words kept (None means all kept)
    n_words = None
    maxlen = None
    word2index = None
    wv_weights = None  # word vectors weights


SD = SharedData()


def _tvt_split(_seqs, _slabels, splits=(7, 2, 1)):
    """train/val/test split for one single domain"""
    assert len(_seqs) == len(_slabels)
    splits = np.asarray(splits)
    splits = np.cumsum(splits / splits.sum())
    # shuffle
    indices = [range(len(_seqs))]
    np.random.shuffle(indices)
    _seqs = _seqs[indices]
    _slabels = _slabels[indices]
    # prepare data (balance data from all labels)
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for slabel in sorted(np.unique(_slabels)):
        seqs_ofs = _seqs[_slabels == slabel]
        slabels_ofs = _slabels[_slabels == slabel]
        # split
        split_ats = np.asarray(splits * len(seqs_ofs), dtype=int)
        X_train.extend(seqs_ofs[:split_ats[0]])
        X_val.extend(seqs_ofs[split_ats[0]:split_ats[1]])
        X_test.extend(seqs_ofs[split_ats[1]:])
        y_train.extend(slabels_ofs[:split_ats[0]])
        y_val.extend(slabels_ofs[split_ats[0]:split_ats[1]])
        y_test.extend(slabels_ofs[split_ats[1]:])
    X_train = np.asarray(X_train, dtype='int')
    X_val = np.asarray(X_val, dtype='int')
    X_test = np.asarray(X_test, dtype='int')
    y_train = np.asarray(y_train, dtype='int')
    y_val = np.asarray(y_val, dtype='int')
    y_test = np.asarray(y_test, dtype='int')
    print(' * X:', X_train.shape, X_val.shape, X_test.shape)
    print(' * y:', y_train.shape, y_val.shape, y_test.shape)
    return (X_train, X_val, X_test), (y_train, y_val, y_test)


def make_data():
    """data pre-processing"""
    global SD

    # load data
    print('loading data: Multi-Domain Sentiment Dataset v2')
    texts, s_labels, d_labels = load_mdsd(domains=DOMAINS)

    # build vocabulary for words
    print('building vocabulary')
    texts_tokens = []
    lens = []
    for text in texts:
        words = word_tokenize(text)
        for idx, word in enumerate(words):
            if word.isdigit():
                words[idx] = '<NUM>'  # replace number token with <NUM>
        texts_tokens.append(words)
        lens.append(len(words))
    maxlen = int(np.percentile(lens, 95))
    print('maxlen:', maxlen)
    counter = Counter()
    for words in texts_tokens:
        counter.update(words)
    word2index = {'<PAD>': 0, '<UNK>': 1}
    for idx, word_count in enumerate(counter.most_common(SD.max_words)):
        if word_count[1] >= SD.min_count:  # min_count
            word2index[word_count[0]] = idx + 2  # starting from 2, 0 used as <PAD>, 1 used as <OOV>
    n_words = len(word2index)
    print('n_words:', n_words)

    # data encode
    print('data encoding')
    seqs = []
    for words in texts_tokens:
        seqs.append([word2index.get(word, 1) for word in words])
    seqs_padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')
    s_labels = np.asarray(s_labels, dtype=int)
    d_labels = np.asarray(d_labels, dtype=int)

    # domain & train/val/test split
    print('labeled data: domain & train/val/test splitting')
    X_train, ys_train, yd_train = [], [], []
    X_val, ys_val, yd_val = [], [], []
    X_test_byd, ys_test_byd, yd_test_byd = {}, {}, {}
    for d_id, d_name in enumerate(DOMAINS):
        print(d_name, 'splitting')
        seqs_padded_ofd = seqs_padded[(d_labels == d_id) & (s_labels != -1)]
        slabels_ofd = s_labels[(d_labels == d_id) & (s_labels != -1)]
        print(' * all:', seqs_padded_ofd.shape, slabels_ofd.shape)
        (X_train_ofd, X_val_ofd, X_test_ofd), (y_train_ofd, y_val_ofd, y_test_ofd) = _tvt_split(seqs_padded_ofd, slabels_ofd)
        # train data (add this domain)
        X_train.extend(X_train_ofd)
        ys_train.extend(y_train_ofd)
        yd_train.extend([d_id] * len(X_train_ofd))
        # val data
        X_val.extend(X_val_ofd)
        ys_val.extend(y_val_ofd)
        yd_val.extend([d_id] * len(X_val_ofd))
        # test data
        X_test_byd[d_id] = X_test_ofd
        ys_test_byd[d_id] = to_categorical(y_test_ofd, num_classes=2)
        yd_test_byd[d_id] = to_categorical([d_id] * len(X_test_ofd), num_classes=len(DOMAINS))
    X_train = np.asarray(X_train, dtype='int')
    ys_train = to_categorical(ys_train, num_classes=2)
    yd_train = to_categorical(yd_train, num_classes=len(DOMAINS))
    X_val = np.asarray(X_val, dtype='int')
    ys_val = to_categorical(ys_val, num_classes=2)
    yd_val = to_categorical(yd_val, num_classes=len(DOMAINS))
    # combine test data from different domains
    X_test = np.concatenate([X_test_byd[idx] for idx in range(len(DOMAINS))])
    ys_test = np.concatenate([ys_test_byd[idx] for idx in range(len(DOMAINS))])
    yd_test = np.concatenate([yd_test_byd[idx] for idx in range(len(DOMAINS))])

    # shuffle train data
    indices = list(range(len(X_train)))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    ys_train = ys_train[indices]
    yd_train = yd_train[indices]
    print('combined labeled data:')
    print('  - train:', X_train.shape, ys_train.shape, yd_train.shape)
    print('  - val:', X_val.shape, ys_val.shape, yd_val.shape)
    print('  - test:', X_test.shape, ys_test.shape, yd_test.shape)
    for d_id, d_name in enumerate(DOMAINS):
        print('  - test for {}:'.format(d_name[:3]), X_test_byd[d_id].shape, ys_test_byd[d_id].shape, yd_test_byd[d_id].shape)

    # embeddings
    print('loading word embeddings from glove')
    embeddings = load_glove(embedding_dim=SD.embed_dim, desired=word2index.keys(), corpus_size=SD.glove_corpus)
    print('processing embedding matrix')
    embedding_mat = get_embedding_mat(embeddings, word2index, SD.embed_dim, idx_from=2)
    SD.wv_weights = [embedding_mat]

    # inject data into SharedData for other functions
    SD.maxlen = maxlen
    SD.n_words = n_words
    SD.word2index = word2index
    SD.X_train, SD.ys_train, SD.yd_train = X_train, ys_train, yd_train
    SD.X_val, SD.ys_val, SD.yd_val = X_val, ys_val, yd_val
    SD.X_test, SD.ys_test, SD.yd_test = X_test, ys_test, yd_test
    SD.X_test_byd, SD.ys_test_byd, SD.yd_test_byd = X_test_byd, ys_test_byd, yd_test_byd


def get_model():
    global SD

    # load embeddings
    weights = SD.wv_weights

    # the model
    print('\nbuilding the model')
    inputs = layers.Input(shape=(SD.maxlen,))
    embeddings = layers.Embedding(
        input_dim=SD.n_words,
        output_dim=SD.embed_dim,
        input_length=SD.maxlen,
        weights=weights)(inputs)
    embeddings = layers.SpatialDropout1D(rate=SD.embed_dropout)(embeddings)

    # domain part
    d_repr = layers.Bidirectional(SD.RNN(
        units=SD.rnn_dim,
        return_sequences=False))(embeddings)
    d_repr = layers.Dense(SD.hidden_dim, activation=SD.activation)(d_repr)
    d_repr = layers.Dropout(SD.fc_dropout)(d_repr)
    d_pred = layers.Dense(len(DOMAINS), activation='softmax', name='d_pred')(d_repr)

    # senti part
    # use domain representation as attention
    episodes = layers.Bidirectional(SD.RNN(
        units=SD.rnn_dim,
        return_sequences=True))(embeddings)
    selected, _ = att_process(candidates=episodes, att=d_repr)
    s_repr = layers.Dense(SD.hidden_dim, activation=SD.activation)(selected)
    s_repr = layers.Dropout(SD.fc_dropout)(s_repr)
    s_pred = layers.Dense(2, activation='softmax', name='s_pred')(s_repr)

    # model
    model = Model(
        inputs=inputs,
        outputs=[s_pred, d_pred])
    model.compile(optimizer=SD.optimizer, metrics=['acc'], loss={
        's_pred': 'categorical_crossentropy',
        'd_pred': 'categorical_crossentropy'
    }, loss_weights={
        's_pred': 1,
        'd_pred': SD.coef
    })
    model.summary()
    return model


def train_and_test(model):
    global SD

    # training
    updater = UpdateMonitor()
    reducer = callbacks.ReduceLROnPlateau(factor=SD.lr_factor, patience=SD.lr_patience, verbose=1)
    stopper = callbacks.EarlyStopping(patience=SD.stop_patience, verbose=1)
    cbks = [updater, reducer, stopper]
    print('\ntraining model')
    model.fit(
        SD.X_train,
        [SD.ys_train, SD.yd_train],
        validation_data=(SD.X_val, [SD.ys_val, SD.yd_val]),
        shuffle=True, batch_size=SD.batch_size, epochs=SD.epochs, verbose=2,
        callbacks=cbks)

    # evaluation
    print('\nTest evaluation:')
    for d_id, d_name in enumerate(DOMAINS):
        scores = model.evaluate(
            SD.X_test_byd[d_id],
            [SD.ys_test_byd[d_id], SD.yd_test_byd[d_id]],
            batch_size=SD.batch_size, verbose=0)
        print('{} acc: {:.4f}'.format(d_name[:3], scores[-2]))


if __name__ == '__main__':
    # data process
    make_data()

    # build & compile model
    model = get_model()

    # train and test
    train_and_test(model)

    print('\nprocess finished ~~~')
