import os
import random
import string

import numpy as np

from logger import get_logger

logger = get_logger(__name__)


def make_dirs(path, empty=False):
    """
    create dir in path and clear dir if required
    """
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    if empty:
        files = [os.path.join(dir_path, item) for item in os.listdir(dir_path)]
        for item in files:
            if os.path.isfile(item):
                os.remove(item)

    return dir_path


###
# data processing
###


def create_dictionary():
    """
    create char2id, id2char and vocab_size
    from printable ascii characters.
    """
    chars = sorted(ch for ch in string.printable if ch not in ("\x0b", "\x0c", "\r"))
    char2id = dict((ch, i + 1) for i, ch in enumerate(chars))
    char2id.update({"": 0})
    id2char = dict((char2id[ch], ch) for ch in char2id)
    vocab_size = len(char2id)
    return char2id, id2char, vocab_size

CHAR2ID, ID2CHAR, VOCAB_SIZE = create_dictionary()


def encode_text(text):
    """
    encode text to array of integers with CHAR2ID
    """
    return np.fromiter((CHAR2ID.get(ch, 0) for ch in text), int)


def decode_text(int_array):
    """
    decode array of integers to text with ID2CHAR
    """
    return "".join((ID2CHAR[ch] for ch in int_array))


def one_hot_encode(indices, num_classes):
    """
    one-hot encoding
    """
    return np.eye(num_classes)[indices]


def batch_generator(sequence, batch_size=64, seq_len=64, one_hot_features=False, one_hot_labels=False):
    """
    batch generator for sequence
    ensures that batches generated are continuous along axis 1
    so that hidden states can be kept across batches and epochs
    """
    # calculate effective length of text to use
    num_batches = (len(sequence) - 1) // (batch_size * seq_len)
    if num_batches == 0:
        raise ValueError("No batches created. Use smaller batch size or sequence length.")
    logger.info("number of batches: %s.", num_batches)
    rounded_len = num_batches * batch_size * seq_len
    logger.info("effective text length: %s.", rounded_len)

    x = np.reshape(sequence[: rounded_len], [batch_size, num_batches * seq_len])
    if one_hot_features:
        x = one_hot_encode(x, VOCAB_SIZE)
    logger.info("x shape: %s.", x.shape)

    y = np.reshape(sequence[1: rounded_len + 1], [batch_size, num_batches * seq_len])
    if one_hot_labels:
        y = one_hot_encode(y, VOCAB_SIZE)
    logger.info("y shape: %s.", y.shape)

    epoch = 0
    while True:
        # roll so that no need to reset rnn states over epochs
        x_epoch = np.split(np.roll(x, -epoch, axis=0), num_batches, axis=1)
        y_epoch = np.split(np.roll(y, -epoch, axis=0), num_batches, axis=1)
        for batch in range(num_batches):
            yield x_epoch[batch], y_epoch[batch]
        epoch += 1


###
# text generation
###


def generate_seed(text, seq_lens=(2, 4, 8, 16, 32)):
    """
    select subsequence randomly from input text
    """
    # randomly choose sequence length
    seq_len = random.choice(seq_lens)
    # randomly choose start index
    start_index = random.randint(0, len(text) - seq_len - 1)
    seed = text[start_index: start_index + seq_len]
    return seed


def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    # need 64 floating point precision
    probs = np.array(probs, dtype=np.float64)
    # set probabilities after top_n to 0
    probs[np.argsort(probs)[:-top_n]] = 0
    # renormalise probabilities
    probs /= np.sum(probs)
    sampled_index = np.random.choice(len(probs), p=probs)
    return sampled_index
