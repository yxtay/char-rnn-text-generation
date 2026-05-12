import json
import os
import time

import numpy as np
from tqdm import tqdm

import tensorflow as tf

from logger import get_logger
from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   make_dirs, sample_from_probs, VOCAB_SIZE)

logger = get_logger(__name__)


class CharRNN(tf.keras.Model):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_size=32,
                 rnn_size=128, num_layers=2, drop_rate=0.0):
        super(CharRNN, self).__init__()
        self.model_args = {"vocab_size": vocab_size, "embedding_size": embedding_size,
                           "rnn_size": rnn_size, "num_layers": num_layers,
                           "drop_rate": drop_rate}
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.rnn_layers = [tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
                           for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, states=None, training=False):
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        new_states = []
        for i, layer in enumerate(self.rnn_layers):
            state = states[i] if states is not None else None
            x, h, c = layer(x, initial_state=state, training=training)
            new_states.append([h, c])
            x = self.dropout(x, training=training)

        logits = self.dense(x)
        return logits, new_states


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
    generated = seed
    encoded = encode_text(seed)

    x = np.expand_dims(encoded[:-1], 0)
    # input shape: [1, seq_len]

    # get rnn state due to seed sequence
    logits, states = model(x, training=False)

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # input shape: [1, 1]
        logits, states = model(x, states=states, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        # output shape: [1, 1, vocab_size]
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += ID2CHAR[next_index]

    logger.info("generated text: \n%s\n", generated)
    return generated


def train_main(args):
    """
    trains model specified in args.
    main method for train subcommand.
    """
    # load text
    with open(args.text_path) as f:
        text = f.read()
    logger.info("corpus length: %s.", len(text))

    # restore or build model
    if args.restore:
        load_path = args.checkpoint_path if args.restore is True else args.restore
        with open("{}.json".format(args.checkpoint_path)) as f:
            model_args = json.load(f)
        model = CharRNN(**model_args)
        # Call model once to build it
        model(np.zeros((1, 1)))
        model.load_weights(load_path)
        logger.info("model restored: %s.", load_path)
    else:
        model_args = {"vocab_size": VOCAB_SIZE,
                      "embedding_size": args.embedding_size,
                      "rnn_size": args.rnn_size,
                      "num_layers": args.num_layers,
                      "drop_rate": args.drop_rate}
        model = CharRNN(**model_args)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clip_norm)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # make checkpoint directory
    log_dir = make_dirs(args.checkpoint_path, empty=True)
    with open("{}.json".format(args.checkpoint_path), "w") as f:
        json.dump(model.model_args, f, indent=2)

    # training start
    num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    data_iter = batch_generator(encode_text(text), args.batch_size, args.seq_len)

    states = None
    logger.info("start of training.")
    time_train = time.time()

    for i in range(args.num_epochs):
        epoch_losses = []
        time_epoch = time.time()
        # training epoch
        for j in tqdm(range(num_batches), desc="epoch {}/{}".format(i + 1, args.num_epochs)):
            x, y = next(data_iter)

            with tf.GradientTape() as tape:
                logits, states = model(x, states=states, training=True)
                loss = loss_fn(y, logits)

            # Detach states to prevent backpropagating through the entire history
            states = [[tf.stop_gradient(s) for s in layer_states] for layer_states in states]

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

        # logs
        duration_epoch = time.time() - time_epoch
        logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                    i + 1, duration_epoch, np.mean(epoch_losses))

        # checkpoint
        save_path = args.checkpoint_path
        if not save_path.endswith(".weights.h5"):
            save_path += ".weights.h5"
        model.save_weights(save_path)
        logger.info("model saved: %s.", save_path)

        # generate text
        seed = generate_seed(text)
        generate_text(model, seed)

    # training end
    duration_train = time.time() - time_train
    logger.info("end of training, duration: %ds.", duration_train)
    # generate text
    seed = generate_seed(text)
    generate_text(model, seed, 1024, 3)

    return model


def generate_main(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    with open("{}.json".format(args.checkpoint_path)) as f:
        model_args = json.load(f)
    model = CharRNN(**model_args)
    # Call model once to build it
    model(np.zeros((1, 1)))

    load_path = args.checkpoint_path
    if not os.path.exists(load_path) and os.path.exists(load_path + ".weights.h5"):
        load_path += ".weights.h5"

    model.load_weights(load_path)
    logger.info("model loaded: %s.", load_path)

    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    return generate_text(model, seed, args.length, args.top_n)


if __name__ == "__main__":
    main("TensorFlow", train_main, generate_main)
