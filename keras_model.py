from argparse import ArgumentParser
from os import path
import sys

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import Nadam

from logger import get_logger
from utils import (encode_text, generate_seed, ID2CHAR, sample_from_probs,
                   training_batch_generator, VOCAB_SIZE)

logger = get_logger(__name__)


def build_model(batch_size, seq_len, vocab_size=VOCAB_SIZE,
                embedding_size=32, rnn_size=128,
                num_layers=2, drop_rate=0.0):
    """
    build character embedding LSTM text generation model.
    """
    logger.info("Building model: batch_size=%s, seq_len=%s, vocab_size=%s, "
                "embedding_size=1%s, rnn_size=%s, num_layers=%s, drop_rate=%s.",
                batch_size, seq_len, vocab_size, embedding_size,
                rnn_size, num_layers, drop_rate)
    model = Sequential()
    # input shape: (batch_size, seq_len)
    model.add(Embedding(vocab_size, embedding_size,
                        batch_input_shape=(batch_size, seq_len)))
    # shape: (batch_size, seq_len, embedding_size)
    for _ in range(num_layers):
        model.add(LSTM(rnn_size, return_sequences=True, stateful=True))
        model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, rnn_size)
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    # output shape: (batch_size, seq_len, vocab_size)
    optimizer = Nadam(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    config = model.get_config()
    # change input shape, (batch_size, input_length) = (1, 1)
    config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
    inference_model = Sequential.from_config(config)
    return inference_model


def generate_text(model, seed, length=400, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
    generated = seed
    encoded = encode_text(seed)
    model.reset_states()

    for idx in encoded[:-1]:
        # input shape: (1, 1)
        x = np.array([[idx]])
        # set internal states
        model.predict(x)

    next_index = encoded[-1]
    for i in range(length):
        # input shape: (1, 1)
        x = np.array([[next_index]])
        # output shape: (1, 1, VOCAB_SIZE)
        probs = model.predict(x)[0, 0, :]
        next_index = sample_from_probs(probs, top_n)
        next_char = ID2CHAR[next_index]
        # append to sequence
        generated += next_char

    logger.info("generated text: \n%s\n", generated)
    return generated


class LoggerCallback(Callback):
    """
    callback to log information.
    generates text at the end of each epoch.
    """
    def __init__(self, text, model):
        super(LoggerCallback, self).__init__()
        self.text = text
        # build inference model using config from learning model
        self.inference_model = build_inference_model(model)
        self.inference_model.trainable = False

    def on_epoch_end(self, epoch, logs=None):
        logger.info("epoch: %s.", epoch)
        logger.info(logs)
        # create seed sequence for text generation
        seed = generate_seed(self.text)
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        generate_text(self.inference_model, seed)

    def on_train_end(self, logs=None):
        logger.info("end of training.")
        logger.info(logs)
        # create seed sequence for text generation
        seed = generate_seed(self.text)
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        generate_text(self.inference_model, seed, 1000, 3)


def train_main(args):
    """
    trains model specfied in args.
    main method for train subcommand.
    """
    # load text
    with open(args.text_path) as f:
        text = f.read()
    logger.info("corpus length: %s.", len(text))

    # load or build model
    if args.init_path and path.exists(args.init_path):
        model = load_model(args.init_path)
        logger.info("model initialised: %s.", args.init_path)
    else:
        model = build_model(batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            vocab_size=VOCAB_SIZE,
                            embedding_size=args.embedding_size,
                            rnn_size=args.rnn_size,
                            num_layers=args.num_layers,
                            drop_rate=args.drop_rate)

    # callbacks
    callbacks = [
        ModelCheckpoint(args.checkpoint_path, verbose=1, save_best_only=False),
        LoggerCallback(text, model)
    ]
    if args.tensorboard_path is not None:
        callbacks.append(TensorBoard(args.tensorboard_path))

    # train the model
    nb_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    model.fit_generator(training_batch_generator(text, args.batch_size, args.seq_len),
                        nb_batches, args.nb_epochs, callbacks=callbacks)
    return model


def generate_main(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    # load learning model for config and weights
    model = load_model(args.checkpoint_path)
    # build inference model and transfer weights
    inference_model = build_inference_model(model)
    inference_model.set_weights(model.get_weights())
    inference_model.trainable = False
    logger.info("model loaded: %s.", args.checkpoint_path)

    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    generate_text(inference_model, seed, args.length, 3)


if __name__ == "__main__":
    logger = get_logger(__name__, console=True)

    arg_parser = ArgumentParser(
        description="Keras character embedding LSTM text generation model.")
    subparsers = arg_parser.add_subparsers(title="subcommands")

    # train args
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("--text-path", required=True,
                              help="path of text file for training")
    train_parser.add_argument("--checkpoint-path", required=True,
                              help="path to save or load model checkpoints")
    train_parser.add_argument("--tensorboard-path", default=None,
                              help="path to save logs for tensorboard")
    train_parser.add_argument("--init-path", default=None,
                              help="path of stored model to initialise")
    train_parser.add_argument("--seq-len", type=int, default=64,
                              help="sequence length of inputs and outputs")
    train_parser.add_argument("--embedding-size", type=int, default=32,
                              help="character embedding size")
    train_parser.add_argument("--rnn-size", type=int, default=128,
                              help="size of rnn cell")
    train_parser.add_argument("--num-layers", type=int, default=2,
                              help="number of rnn layers")
    train_parser.add_argument("--drop-rate", type=float, default=0,
                              help="dropout rate for rnn layers")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="training batch size")
    train_parser.add_argument("--nb-epochs", type=int, default=32,
                              help="number of epochs for training")
    train_parser.set_defaults(main=train_main)

    # generate args
    generate_parser = subparsers.add_parser("generate", help="generate text from trained model")
    generate_parser.add_argument("--checkpoint-path", required=True,
                                 help="path to load model checkpoints")
    generate_parser.add_argument("--text-path", required=True,
                                 help="path of text file to generate seed")
    generate_parser.add_argument("--seed", default=None,
                                 help="seed character sequence to generate text")
    generate_parser.add_argument("--length", type=int, default=1000,
                                 help="length of sequence to generate")
    generate_parser.set_defaults(main=generate_main)

    args = arg_parser.parse_args()
    logger.debug("call: %s", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s", args)

    try:
        args.main(args)
    except Exception as e:
        logger.exception(e)
