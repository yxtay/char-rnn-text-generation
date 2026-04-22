from argparse import ArgumentParser
import os
import random
import string
import sys

import numpy as np

from logger import get_logger

logger = get_logger(__name__)

###
# file system
###


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


def path_join(*paths, empty=False):
    """
    join paths and create dir
    """
    path = os.path.abspath(os.path.join(*paths))
    make_dirs(os.path.dirname(path), empty)
    return path


###
# data processing
###

def dictionary_from_chars(chars):
    """
    Build char2id, id2char and vocab_size from distinct characters.
    Index 0 is reserved for unknown / padding (empty string key).
    """
    chars = sorted(set(chars) - {"\x0b", "\x0c", "\r"})
    char2id = dict((ch, i + 1) for i, ch in enumerate(chars))
    char2id.update({"": 0})
    id2char = dict((char2id[ch], ch) for ch in char2id)
    vocab_size = len(char2id)
    return char2id, id2char, vocab_size


def create_dictionary():
    """
    create char2id, id2char and vocab_size
    from printable ascii characters.
    """
    chars = [ch for ch in string.printable if ch not in ("\x0b", "\x0c", "\r")]
    return dictionary_from_chars(chars)


CHAR2ID, ID2CHAR, VOCAB_SIZE = create_dictionary()


def apply_wordnet_char_vocabulary():
    """
    Replace the character alphabet with one derived from WordNet lemma strings.
    Mutates CHAR2ID and ID2CHAR in place so existing imports of those dicts stay
    valid; updates VOCAB_SIZE (re-read via ``import utils`` / ``utils.VOCAB_SIZE``).
    """
    global VOCAB_SIZE
    from wordnet_lexicon import lemma_character_inventory

    new_c2i, new_i2c, vs = dictionary_from_chars(lemma_character_inventory())
    CHAR2ID.clear()
    CHAR2ID.update(new_c2i)
    ID2CHAR.clear()
    ID2CHAR.update(new_i2c)
    VOCAB_SIZE = vs


def encode_text(text, char2id=None):
    """
    encode text to array of integers with CHAR2ID
    """
    if char2id is None:
        char2id = CHAR2ID
    return np.fromiter((char2id.get(ch, 0) for ch in text), int)


def decode_text(int_array, id2char=None):
    """
    decode array of integers to text with ID2CHAR
    """
    if id2char is None:
        id2char = ID2CHAR
    return "".join((id2char[ch] for ch in int_array))


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


###
# training corpora (single file or directory)
###


def list_training_text_files(text_path, extensions=(".txt",)):
    """
    Resolve ``text_path`` to a list of corpus files.

    If ``text_path`` is a file, returns a one-element list. If it is a
    directory, returns all non-hidden files whose names end with one of
    ``extensions``, sorted lexicographically (non-recursive).
    """
    abs_path = os.path.abspath(text_path)
    if os.path.isfile(abs_path):
        return [abs_path]
    if not os.path.isdir(abs_path):
        raise ValueError("Training text path is not a file or directory: {!r}".format(text_path))
    ext_tuple = tuple(e.lower() for e in extensions)
    files = []
    for name in sorted(os.listdir(abs_path)):
        if name.startswith("."):
            continue
        full = os.path.join(abs_path, name)
        if not os.path.isfile(full):
            continue
        lower = name.lower()
        if any(lower.endswith(ext) for ext in ext_tuple):
            files.append(full)
    if not files:
        raise ValueError(
            "No training text files (*{}) found in directory: {!r}".format(
                ",".join(extensions), abs_path))
    return files


def load_training_text(file_path):
    """Load the full text of a corpus file as UTF-8."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def corpus_for_training_epoch(text_paths, epoch_index):
    """
    Pick the corpus for training epoch ``epoch_index`` (0-based).

    With multiple files (directory mode), cycles in sorted path order.
    With a single file, always uses that file.
    Returns ``(path, text)``.
    """
    path = text_paths[epoch_index % len(text_paths)]
    text = load_training_text(path)
    return path, text


def resolve_seed_text_file(text_path):
    """
    For ``generate``: return a concrete text file path. If ``text_path`` is a
    directory, uses the first path from ``list_training_text_files`` (sorted).
    """
    return list_training_text_files(text_path)[0]


###
# main
###

def main(framework, train_main, generate_main):
    arg_parser = ArgumentParser(
        description="{} character embeddings LSTM text generation model.".format(framework))
    subparsers = arg_parser.add_subparsers(title="subcommands")

    # train args
    train_parser = subparsers.add_parser("train", help="train model on text file")
    train_parser.add_argument("--checkpoint-path", required=True,
                              help="path to save or load model checkpoints (required)")
    train_parser.add_argument("--text-path", required=True,
                              help="path to a UTF-8 text file or a directory of .txt files "
                                   "(directory: cycle one file per epoch, sorted order)")
    train_parser.add_argument("--restore", nargs="?", default=False, const=True,
                              help="whether to restore from checkpoint_path "
                                   "or from another path if specified")
    train_parser.add_argument("--seq-len", type=int, default=64,
                              help="sequence length of inputs and outputs (default: %(default)s)")
    train_parser.add_argument("--embedding-size", type=int, default=32,
                              help="character embedding size (default: %(default)s)")
    train_parser.add_argument("--rnn-size", type=int, default=128,
                              help="size of rnn cell (default: %(default)s)")
    train_parser.add_argument("--num-layers", type=int, default=2,
                              help="number of rnn layers (default: %(default)s)")
    train_parser.add_argument("--drop-rate", type=float, default=0.,
                              help="dropout rate for rnn layers (default: %(default)s)")
    train_parser.add_argument("--learning-rate", type=float, default=0.001,
                              help="learning rate (default: %(default)s)")
    train_parser.add_argument("--clip-norm", type=float, default=5.,
                              help="max norm to clip gradient (default: %(default)s)")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="training batch size (default: %(default)s)")
    train_parser.add_argument("--num-epochs", type=int, default=32,
                              help="number of epochs for training (default: %(default)s)")
    train_parser.add_argument("--log-path", default=os.path.join(os.path.dirname(__file__), "main.log"),
                              help="path of log file (default: %(default)s)")
    train_parser.add_argument("--wordnet-char-vocab", action="store_true",
                              help="restrict alphabet to chars appearing in WordNet lemmas "
                                   "(smaller softmax; requires NLTK + wordnet corpus)")
    train_parser.set_defaults(main=train_main)

    # generate args
    generate_parser = subparsers.add_parser("generate", help="generate text from trained model")
    generate_parser.add_argument("--checkpoint-path", required=True,
                                 help="path to load model checkpoints (required)")
    group = generate_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text-path", help="path of text file to generate seed")
    group.add_argument("--seed", default=None, help="seed character sequence")
    generate_parser.add_argument("--length", type=int, default=1024,
                                 help="length of character sequence to generate (default: %(default)s)")
    generate_parser.add_argument("--top-n", type=int, default=3,
                                 help="number of top choices to sample (default: %(default)s)")
    generate_parser.add_argument("--log-path", default=os.path.join(os.path.dirname(__file__), "main.log"),
                                 help="path of log file (default: %(default)s)")
    generate_parser.add_argument("--wordnet-char-vocab", action="store_true",
                                 help="use same WordNet-derived alphabet as training "
                                      "(required if the checkpoint was trained with this flag)")
    generate_parser.set_defaults(main=generate_main)

    args = arg_parser.parse_args()
    get_logger("__main__", log_path=args.log_path, console=True)
    logger = get_logger(__name__, log_path=args.log_path, console=True)
    if getattr(args, "wordnet_char_vocab", False):
        apply_wordnet_char_vocabulary()
        logger.info("WordNet-derived character vocabulary size: %s", VOCAB_SIZE)
    logger.debug("call: %s", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s", args)

    try:
        args.main(args)
    except Exception as e:
        logger.exception(e)
