import json
import time

from tqdm import tqdm

import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn, rnn
from mxnet import autograd

from logger import get_logger
from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   make_dirs, sample_from_probs, VOCAB_SIZE)

logger = get_logger(__name__)


class Model(gluon.Block):
    """
    build character embeddings LSTM text generation model.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_size=32,
                 rnn_size=128, num_layers=2, drop_rate=0.0, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.args = {"vocab_size": vocab_size, "embedding_size": embedding_size,
                     "rnn_size": rnn_size, "num_layers": num_layers,
                     "drop_rate": drop_rate}
        with self.name_scope():
            self.encoder = nn.Embedding(vocab_size, embedding_size)
            self.dropout = nn.Dropout(drop_rate)
            self.rnn = rnn.LSTM(rnn_size, num_layers, dropout=drop_rate,
                                input_size=embedding_size)
            self.decoder = nn.Dense(vocab_size, in_units=rnn_size)

    def forward(self, inputs, state):
        # input shape: [seq_len, batch_size]
        seq_len, batch_size = inputs.shape
        embed_seq = self.dropout(self.encoder(inputs))
        # shape: [seq_len, batch_size, embedding_size]
        rnn_out, state = self.rnn(embed_seq, state)
        # rnn_out shape: [seq_len, batch_size, rnn_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        rnn_out = self.dropout(rnn_out)
        # shape: [seq_len, batch_size, rnn_size]
        logits = (self.decoder(rnn_out.reshape((-1, rnn_out.shape[2])))
                  .reshape((seq_len, batch_size, -1)))
        # output shape: [seq_len, batch_size, vocab_size]
        return logits, state

    def begin_state(self, batch_size=1):
        """
        initialises rnn states.
        """
        return self.rnn.begin_state(batch_size)

    def save(self, checkpoint_path):
        """
        saves model and args to checkpoint_path.
        """
        with open("{}.json".format(checkpoint_path), "w") as f:
            json.dump(self.args, f, indent=2)
        self.save_params(checkpoint_path)
        logger.info("model saved: %s.", checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path, ctx=mx.cpu(), **kwargs):
        """
        loads model from checkpoint_path.
        """
        with open("{}.json".format(checkpoint_path)) as f:
            model_args = json.load(f)
        model = cls(**model_args, **kwargs)
        model.load_params(checkpoint_path, ctx)
        logger.info("model loaded: %s.", checkpoint_path)
        return model


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
    generated = seed
    encoded = mx.nd.array(encode_text(seed))
    seq_len = encoded.shape[0]

    x = F.expand_dims(encoded[:seq_len-1], 1)
    # input shape: [seq_len, 1]
    state = model.begin_state()
    # get rnn state due to seed sequence
    _, state = model(x, state)

    next_index = encoded[seq_len-1].asscalar()
    for i in range(length):
        x = mx.nd.array([[next_index]])
        # input shape: [1, 1]
        logit, state = model(x, state)
        # output shape: [1, vocab_size]
        probs = F.softmax(logit)
        next_index = sample_from_probs(probs.asnumpy().squeeze(), top_n)
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
        logger.info("restoring model.")
        load_path = args.checkpoint_path if args.restore is True else args.restore
        model = Model.load(load_path)
    else:
        model = Model(vocab_size=VOCAB_SIZE,
                      embedding_size=args.embedding_size,
                      rnn_size=args.rnn_size,
                      num_layers=args.num_layers,
                      drop_rate=args.drop_rate)
        model.initialize(mx.init.Xavier())
    model.hybridize()

    # make checkpoint directory
    make_dirs(args.checkpoint_path)
    model.save(args.checkpoint_path)

    # loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=1)
    # optimizer
    optimizer = mx.optimizer.Adam(learning_rate=args.learning_rate,
                                  clip_gradient=args.clip_norm)
    # trainer
    trainer = gluon.Trainer(model.collect_params(), optimizer)

    # training start
    num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    data_iter = batch_generator(encode_text(text), args.batch_size, args.seq_len)
    state = model.begin_state(args.batch_size)
    logger.info("start of training.")
    time_train = time.time()
    for i in range(args.num_epochs):
        epoch_losses = mx.nd.empty(num_batches)
        time_epoch = time.time()
        # training epoch
        for j in tqdm(range(num_batches), desc="epoch {}/{}".format(i + 1, args.num_epochs)):
            # prepare inputs
            x, y = next(data_iter)
            x = mx.nd.array(x.T)
            y = mx.nd.array(y.T)
            # reset state variables to remove their history
            state = [arr.detach() for arr in state]

            with autograd.record():
                logits, state = model(x, state)
                # calculate loss
                L = loss(logits, y)
                L = F.mean(L)
                epoch_losses[j] = L.asscalar()
                # calculate gradient
                L.backward()
            # apply gradient update
            trainer.step(1)

        # logs
        duration_epoch = time.time() - time_epoch
        logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                    i + 1, duration_epoch, F.mean(epoch_losses).asscalar())
        # checkpoint
        model.save_params(args.checkpoint_path)
        logger.info("model saved: %s.", args.checkpoint_path)
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
    # load model
    inference_model = Model.load(args.checkpoint_path, mx.cpu())

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
    main("MXNet", train_main, generate_main)
