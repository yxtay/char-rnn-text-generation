import os
import time

import numpy as np

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import Adam

from logger import get_logger
from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   make_dirs, sample_from_probs, VOCAB_SIZE)

logger = get_logger(__name__)


def build_model(batch_size, seq_len, vocab_size=VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, drop_rate=0.0,
                learning_rate=0.001, clip_norm=5.0):
    """
    build character embeddings LSTM text generation model.
    """
    logger.info("building model: batch_size=%s, seq_len=%s, vocab_size=%s, "
                "embedding_size=%s, rnn_size=%s, num_layers=%s, drop_rate=%s, "
                "learning_rate=%s, clip_norm=%s.",
                batch_size, seq_len, vocab_size, embedding_size,
                rnn_size, num_layers, drop_rate,
                learning_rate, clip_norm)
    model = Sequential()
    # input shape: (batch_size, seq_len)
    model.add(Embedding(vocab_size, embedding_size,
                        batch_input_shape=(batch_size, seq_len)))
    model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, embedding_size)
    for _ in range(num_layers):
        model.add(LSTM(rnn_size, return_sequences=True, stateful=True))
        model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, rnn_size)
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    # output shape: (batch_size, seq_len, vocab_size)
    optimizer = Adam(learning_rate, clipnorm=clip_norm)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    logger.info("building inference model.")
    config = model.get_config()
    # edit batch_size and seq_len
    config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
    inference_model = Sequential.from_config(config)
    inference_model.trainable = False
    return inference_model


def generate_text(model, seed, length=512, top_n=10):
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
        x = np.array([[idx]])
        # input shape: (1, 1)
        # set internal states
        model.predict(x)

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # input shape: (1, 1)
        probs = model.predict(x)
        # output shape: (1, 1, vocab_size)
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += ID2CHAR[next_index]

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
        self.time_train = self.time_epoch = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration_epoch = time.time() - self.time_epoch
        logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                    epoch, duration_epoch, logs["loss"])
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        # generate text
        seed = generate_seed(self.text)
        generate_text(self.inference_model, seed)

    def on_train_begin(self, logs=None):
        logger.info("start of training.")
        self.time_train = time.time()

    def on_train_end(self, logs=None):
        duration_train = time.time() - self.time_train
        logger.info("end of training, duration: %ds.", duration_train)
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        # generate text
        seed = generate_seed(self.text)
        generate_text(self.inference_model, seed, 1024, 3)


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
    if args.restore:
        load_path = args.checkpoint_path if args.restore is True else args.restore
        model = load_model(load_path)
        logger.info("model restored: %s.", load_path)
    else:
        model = build_model(batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            vocab_size=VOCAB_SIZE,
                            embedding_size=args.embedding_size,
                            rnn_size=args.rnn_size,
                            num_layers=args.num_layers,
                            drop_rate=args.drop_rate,
                            learning_rate=args.learning_rate,
                            clip_norm=args.clip_norm)

    # make and clear checkpoint directory
    log_dir = make_dirs(args.checkpoint_path, empty=True)
    model.save(args.checkpoint_path)
    logger.info("model saved: %s.", args.checkpoint_path)
    # callbacks
    callbacks = [
        ModelCheckpoint(args.checkpoint_path, verbose=1, save_best_only=False),
        TensorBoard(log_dir, write_graph=True, embeddings_freq=1,
                    embeddings_metadata={"embedding_1": os.path.abspath(os.path.join("data", "id2char.tsv"))}),
        LoggerCallback(text, model)
    ]

    # training start
    num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    model.reset_states()
    model.fit_generator(batch_generator(encode_text(text), args.batch_size, args.seq_len, one_hot_labels=True),
                        num_batches, args.num_epochs, callbacks=callbacks)
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
    logger.info("model loaded: %s.", args.checkpoint_path)

    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, args.top_n)


if __name__ == "__main__":
    main("Keras", train_main, generate_main)
