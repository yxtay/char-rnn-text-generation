import json
import os
import time

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import layers, rnn
from tensorflow.contrib.tensorboard.plugins import projector

from logger import get_logger
from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   make_dirs, sample_from_probs, VOCAB_SIZE)

logger = get_logger(__name__)


def build_infer_graph(x, batch_size, vocab_size=VOCAB_SIZE, embedding_size=32,
                      rnn_size=128, num_layers=2, p_keep=1.0):
    """
    builds inference graph
    """
    infer_args = {"batch_size": batch_size, "vocab_size": vocab_size,
                  "embedding_size": embedding_size, "rnn_size": rnn_size,
                  "num_layers": num_layers, "p_keep": p_keep}
    logger.debug("building inference graph: %s.", infer_args)

    # other placeholders
    p_keep = tf.placeholder_with_default(p_keep, [], "p_keep")
    batch_size = tf.placeholder_with_default(batch_size, [], "batch_size")

    # embedding layer
    embed_seq = layers.embed_sequence(x, vocab_size, embedding_size)
    # shape: [batch_size, seq_len, embedding_size]
    embed_seq = tf.nn.dropout(embed_seq, keep_prob=p_keep)
    # shape: [batch_size, seq_len, embedding_size]

    # RNN layers
    cells = [rnn.LSTMCell(rnn_size) for _ in range(num_layers)]
    cells = [rnn.DropoutWrapper(cell, output_keep_prob=p_keep) for cell in cells]
    cells = rnn.MultiRNNCell(cells)
    input_state = cells.zero_state(batch_size, tf.float32)
    # shape: [num_layers, 2, batch_size, rnn_size]
    rnn_out, output_state = tf.nn.dynamic_rnn(cells, embed_seq, initial_state=input_state)
    # rnn_out shape: [batch_size, seq_len, rnn_size]
    # output_state shape: [num_layers, 2, batch_size, rnn_size]
    with tf.name_scope("lstm"):
        tf.summary.histogram("outputs", rnn_out)
        for c_state, h_state in output_state:
            tf.summary.histogram("c_state", c_state)
            tf.summary.histogram("h_state", h_state)

    # fully connected layer
    logits = layers.fully_connected(rnn_out, vocab_size, activation_fn=None)
    # shape: [batch_size, seq_len, vocab_size]

    # predictions
    with tf.name_scope("softmax"):
        probs = tf.nn.softmax(logits)
        # shape: [batch_size, seq_len, vocab_size]

    with tf.name_scope("sequence"):
        tf.summary.histogram("embeddings", embed_seq)
        tf.summary.histogram("logits", logits)

    model = {"logits": logits, "probs": probs,
             "input_state": input_state, "output_state": output_state,
             "p_keep": p_keep, "batch_size": batch_size, "infer_args": infer_args}
    return model


def build_eval_graph(logits, y):
    """
    builds evaluation graph
    """
    eval_args = {}
    logger.debug("building evaluation graph: %s.", eval_args)

    with tf.name_scope("loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        # shape: [batch_size, seq_len]
        seq_loss = tf.reduce_mean(loss, axis=1, name="seq_loss")
        # shape: [batch_size]
        batch_loss = tf.reduce_mean(seq_loss, name="batch_loss")
        # shape: []
        tf.summary.histogram("sequence", seq_loss)

    model = {"loss": batch_loss, "eval_args": eval_args}
    return model


def build_train_graph(loss, learning_rate=0.001, clip_norm=5.0):
    """
    builds training graph
    """
    train_args = {"learning_rate": learning_rate, "clip_norm": clip_norm}
    logger.debug("building training graph: %s.", train_args)

    learning_rate = tf.placeholder_with_default(learning_rate, [], "learning_rate")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = layers.optimize_loss(loss, global_step, learning_rate, "Adam",
                                    clip_gradients=clip_norm)

    model = {"global_step": global_step, "train_op": train_op,
             "learning_rate": learning_rate, "train_args": train_args}
    return model


def build_model(batch_size, vocab_size=VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, p_keep=1.0, learning_rate=0.001,
                clip_norm=5.0, build_eval=True, build_train=True):
    """
    builds model end-to-end, including data placeholders and saver
    """
    model_args = {"batch_size": batch_size, "vocab_size": vocab_size,
                  "embedding_size": embedding_size, "rnn_size": rnn_size,
                  "num_layers": num_layers, "p_keep": p_keep,
                  "learning_rate": learning_rate, "clip_norm": clip_norm}
    logger.info("building model: %s.", model_args)

    # data placeholders
    x = tf.placeholder(tf.int32, [None, None], "X")
    # shape: [batch_size, seq_len]
    y = tf.placeholder(tf.int32, [None, None], "Y")
    # shape: [batch_size, seq_len]

    model = {"X": x, "Y": y, "args": model_args}
    model.update(build_infer_graph(model["X"],
                                   batch_size=batch_size,
                                   vocab_size=VOCAB_SIZE,
                                   embedding_size=embedding_size,
                                   rnn_size=rnn_size,
                                   num_layers=num_layers,
                                   p_keep=p_keep))
    if build_eval or build_train:
        model.update(build_eval_graph(model["logits"], model["Y"]))
    if build_train:
        model.update(build_train_graph(model["loss"],
                                       learning_rate=learning_rate,
                                       clip_norm=clip_norm))
    # init op
    model["init_op"] = tf.global_variables_initializer()
    # tensorboard summary
    model["summary"] = tf.summary.merge_all()
    # saver
    model["saver"] = tf.train.Saver()
    return model


def load_inference_model(checkpoint_path):
    """
    builds inference model from model args saved in `checkpoint_path`
    """
    # load model args
    with open("{}.json".format(checkpoint_path)) as f:
        model_args = json.load(f)
    # edit batch_size and p_keep
    model_args.update({"batch_size": 1, "p_keep": 1.0})
    infer_model = build_model(**model_args, build_eval=False, build_train=False)
    logger.info("inference model loaded: %s.", checkpoint_path)
    return infer_model


def generate_text(model, sess, seed, length=512, top_n=10):
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
    state = sess.run(model["output_state"], feed_dict={model["X"]: x})

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # input shape: [1, 1]
        feed_dict = {model["X"]: x, model["input_state"]: state}
        probs, state = sess.run([model["probs"], model["output_state"]], feed_dict=feed_dict)
        # output shape: [1, 1, vocab_size]
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += ID2CHAR[next_index]

    logger.info("generated text: \n%s\n", generated)
    return generated


def train_main(args):
    """
    trains model specfied in args.
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
        logger.info("model restored: %s.", load_path)
    else:
        load_path = None
        model_args = {"batch_size": args.batch_size,
                      "vocab_size": VOCAB_SIZE,
                      "embedding_size": args.embedding_size,
                      "rnn_size": args.rnn_size,
                      "num_layers": args.num_layers,
                      "p_keep": 1 - args.drop_rate,
                      "learning_rate": args.learning_rate,
                      "clip_norm": args.clip_norm}

    # build train model
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = build_model(**model_args)

    with tf.Session(graph=train_graph) as train_sess:
        # restore or initialise model weights
        if load_path is not None:
            train_model["saver"].restore(train_sess, load_path)
            logger.info("model weights restored: %s.", load_path)
        else:
            train_sess.run(train_model["init_op"])

        # clear checkpoint directory
        log_dir = make_dirs(args.checkpoint_path, empty=True)
        # save model
        with open("{}.json".format(args.checkpoint_path), "w") as f:
            json.dump(train_model["args"], f, indent=2)
        checkpoint_path = train_model["saver"].save(train_sess, args.checkpoint_path)
        logger.info("model saved: %s.", checkpoint_path)
        # tensorboard logger
        summary_writer = tf.summary.FileWriter(log_dir, train_sess.graph)
        # embeddings visualisation
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "EmbedSequence/embeddings"
        embedding.metadata_path = os.path.abspath(os.path.join("data", "id2char.tsv"))
        projector.visualize_embeddings(summary_writer, config)
        logger.info("tensorboard set up.")

        # build infer model
        inference_graph = tf.Graph()
        with inference_graph.as_default():
            inference_model = load_inference_model(args.checkpoint_path)

        # training start
        num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
        data_iter = batch_generator(encode_text(text), args.batch_size, args.seq_len)
        fetches = [train_model["train_op"], train_model["output_state"],
                   train_model["loss"], train_model["summary"]]
        state = train_sess.run(train_model["input_state"])
        logger.info("start of training.")
        time_train = time.time()

        for i in range(args.num_epochs):
            epoch_losses = np.empty(num_batches)
            time_epoch = time.time()
            # training epoch
            for j in tqdm(range(num_batches), desc="epoch {}/{}".format(i + 1, args.num_epochs)):
                x, y = next(data_iter)
                feed_dict = {train_model["X"]: x, train_model["Y"]: y, train_model["input_state"]: state}
                _, state, loss, summary_log = train_sess.run(fetches, feed_dict)
                epoch_losses[j] = loss

            # logs
            duration_epoch = time.time() - time_epoch
            logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                        i + 1, duration_epoch, epoch_losses.mean())
            # tensorboard logs
            summary_writer.add_summary(summary_log, i + 1)
            summary_writer.flush()
            # checkpoint
            checkpoint_path = train_model["saver"].save(train_sess, args.checkpoint_path)
            logger.info("model saved: %s.", checkpoint_path)

            # generate text
            seed = generate_seed(text)
            with tf.Session(graph=inference_graph) as infer_sess:
                # restore weights
                inference_model["saver"].restore(infer_sess, checkpoint_path)
                generate_text(inference_model, infer_sess, seed)

        # training end
        duration_train = time.time() - time_train
        logger.info("end of training, duration: %ds.", duration_train)
        # generate text
        seed = generate_seed(text)
        with tf.Session(graph=inference_graph) as infer_sess:
            # restore weights
            inference_model["saver"].restore(infer_sess, checkpoint_path)
            generate_text(inference_model, infer_sess, seed, 1024, 3)

    return train_model


def generate_main(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    # restore model
    inference_graph = tf.Graph()
    with inference_graph.as_default():
        inference_model = load_inference_model(args.checkpoint_path)

    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    with tf.Session(graph=inference_graph) as infer_sess:
        # restore weights
        inference_model["saver"].restore(infer_sess, args.checkpoint_path)
        return generate_text(inference_model, infer_sess, seed, args.length, args.top_n)


if __name__ == "__main__":
    main("TensorFlow", train_main, generate_main)
