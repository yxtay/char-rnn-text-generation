import json
import os
import time

import numpy as np

import chainer
from chainer import (functions as F,
                     links as L,
                     ChainList, Variable)
from chainer.training import extension, extensions

from logger import get_logger
from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   make_dirs, sample_from_probs, VOCAB_SIZE)

logger = get_logger(__name__)


class Network(ChainList):
    """
    build character embeddings LSTM neural network.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_size=32,
                 rnn_size=128, num_layers=2, drop_rate=0.0):
        super(Network, self).__init__()
        self.args = {"vocab_size": vocab_size, "embedding_size": embedding_size,
                     "rnn_size": rnn_size, "num_layers": num_layers,
                     "drop_rate": drop_rate}

        self.encoder = L.EmbedID(vocab_size, embedding_size)
        self.rnn_layers = [L.LSTM(embedding_size, rnn_size)]
        self.rnn_layers.extend(L.LSTM(rnn_size, rnn_size) for _ in range(num_layers-1))
        self.decoder = L.Linear(rnn_size, vocab_size)

        self.add_link(self.encoder)
        for link in self.rnn_layers:
            self.add_link(link)
        self.add_link(self.decoder)

    def __call__(self, inputs):
        # input shape: [batch_size]
        embed_seq = F.dropout(self.encoder(inputs), self.args["drop_rate"])
        # shape: [batch_size, embedding_size]
        rnn_out = embed_seq
        for link in self.rnn_layers:
            rnn_out = F.dropout(link(rnn_out), self.args["drop_rate"])
        # shape: [batch_size, rnn_size]
        logits = self.decoder(rnn_out)
        # shape: [batch_size, vocab_size]
        return logits

    def reset_state(self):
        """
        resets rnn states.
        """
        for link in self.rnn_layers:
            link.reset_state()

    def get_state(self):
        """
        get rnn states.
        """
        return [(link.c, link.h) for link in self.rnn_layers]

    def set_state(self, state):
        """
        set rnn states
        """
        for link, (c, h) in zip(self.rnn_layers, state):
            link.set_state(c, h)


def load_model(checkpoint_path):
    """
    loads model from checkpoint_path.
    """
    with open("{}.json".format(checkpoint_path)) as f:
        model_args = json.load(f)
    net = Network(**model_args)
    model = L.Classifier(net)
    chainer.serializers.load_npz(checkpoint_path, model)
    logger.info("model loaded: %s.", checkpoint_path)
    return model


class DataIterator(chainer.dataset.Iterator):
    """
    data iterator for chainer.
    """
    def __init__(self, text, batch_size=64, seq_len=64):
        self.data_iterator = batch_generator(encode_text(text).astype(np.int32),
                                             batch_size, seq_len)
        self.num_batches = (len(text) - 1) // (batch_size * seq_len)
        self.iteration = 0
        self.epoch = 0
        self.is_new_epoch = True

    def __next__(self):
        self.iteration += 1
        self.is_new_epoch = self.iteration % self.num_batches == 0
        if self.is_new_epoch:
            self.epoch += 1

        return next(self.data_iterator)

    @property
    def epoch_detail(self):
        return self.iteration / self.num_batches

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BpttUpdater(chainer.training.StandardUpdater):
    """
    updater for backpropagation through time.
    """
    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        x, y = next(train_iter)
        seq_len = x.shape[1]

        loss = 0
        for i in range(seq_len):
            loss += optimizer.target(chainer.Variable(x[:, i]), chainer.Variable(y[:, i]))

        optimizer.target.cleargrads()  # clear gradients
        loss.backward()  # calculate gradient
        loss.unchain_backward()  # truncate
        optimizer.update()  # apply gradient update


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
    generated = seed
    encoded = encode_text(seed).astype(np.int32)
    model.predictor.reset_state()

    with chainer.using_config("train", False), chainer.no_backprop_mode():
        for idx in encoded[:-1]:
            x = Variable(np.array([idx]))
            # input shape: [1]
            # set internal states
            model.predictor(x)

        next_index = encoded[-1]
        for i in range(length):
            x = Variable(np.array([next_index], dtype=np.int32))
            # input shape: [1]
            probs = F.softmax(model.predictor(x))
            # output shape: [1, vocab_size]
            next_index = sample_from_probs(probs.data.squeeze(), top_n)
            # append to sequence
            generated += ID2CHAR[next_index]

    logger.info("generated text: \n%s\n", generated)
    return generated


class LoggerExtension(extension.Extension):
    """
    chainer Extension for logging.
    generates text at the end of each epoch.
    """
    trigger = (1, "epoch")
    priority = -200

    def __init__(self, text):
        self.text = text
        self.time_epoch = time.time()

    def __call__(self, trainer):
        duration_epoch = time.time() - self.time_epoch
        epoch = trainer.updater.epoch
        loss = trainer.observation["main/loss"].data
        logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                    epoch, duration_epoch, loss)

        # get rnn state
        model = trainer.updater.get_optimizer("main").target
        state = model.predictor.get_state()
        # generate text
        seed = generate_seed(self.text)
        generate_text(model, seed)
        # set rnn back to training state
        model.predictor.set_state(state)

        # reset time
        self.time_epoch = time.time()

    def initialize(self, _):
        self.time_epoch = time.time()


def train_main(args):
    """
    trains model specfied in args.
    main method for train subcommand.
    """
    # load text
    with open(args.text_path) as f:
        text = f.read()
    logger.info("corpus length: %s.", len(text))

    # data iterator
    data_iter = DataIterator(text, args.batch_size, args.seq_len)

    # load or build model
    if args.restore:
        logger.info("restoring model.")
        load_path = args.checkpoint_path if args.restore is True else args.restore
        model = load_model(load_path)
    else:
        net = Network(vocab_size=VOCAB_SIZE,
                      embedding_size=args.embedding_size,
                      rnn_size=args.rnn_size,
                      num_layers=args.num_layers,
                      drop_rate=args.drop_rate)
        model = L.Classifier(net)

    # make checkpoint directory
    log_dir = make_dirs(args.checkpoint_path)
    with open("{}.json".format(args.checkpoint_path), "w") as f:
        json.dump(model.predictor.args, f, indent=2)
    chainer.serializers.save_npz(args.checkpoint_path, model)
    logger.info("model saved: %s.", args.checkpoint_path)

    # optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    # clip gradient norm
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.clip_norm))

    # trainer
    updater = BpttUpdater(data_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (args.num_epochs, 'epoch'), out=log_dir)
    trainer.extend(extensions.snapshot_object(model, filename=os.path.basename(args.checkpoint_path)))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(y_keys=["main/loss"]))
    trainer.extend(LoggerExtension(text))

    # training start
    model.predictor.reset_state()
    logger.info("start of training.")
    time_train = time.time()
    trainer.run()

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
    # load model
    inference_model = load_model(args.checkpoint_path)

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
    main("Chainer", train_main, generate_main)
