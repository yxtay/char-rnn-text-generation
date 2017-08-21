from argparse import ArgumentParser
import sys
import time

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from logger import get_logger
from utils import (make_dirs, encode_text, generate_seed, ID2CHAR,
                   batch_generator, VOCAB_SIZE)

logger = get_logger(__name__)


class Model(nn.Module):
    """
    build character embeddings LSTM text generation model.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_size=32,
                 rnn_size=128, num_layers=2, drop_rate=0.0):
        super(Model, self).__init__()
        self.args = {"vocab_size": vocab_size, "embedding_size": embedding_size,
                     "rnn_size": rnn_size, "num_layers": num_layers,
                     "drop_rate": drop_rate}
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(drop_rate)
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers, dropout=drop_rate)
        self.decoder = nn.Linear(rnn_size, vocab_size)

    def forward(self, input, hidden):
        # input shape: [seq_len, batch_size]
        embed_seq = self.dropout(self.encoder(input))
        # shape: [seq_len, batch_size, embedding_size]
        rnn_out, hidden = self.rnn(embed_seq, hidden)
        # rnn_out shape: [seq_len, batch_size, rnn_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        rnn_out = self.dropout(rnn_out)
        # shape: [seq_len, batch_size, rnn_size]
        logits = self.decoder(rnn_out.view(-1, rnn_out.size(2)))
        # output shape: [seq_len * batch_size, vocab_size]
        return logits, hidden

    def predict(self, input, hidden):
        # input shape: [seq_len, batch_size]
        logits, hidden = self.forward(input, hidden)
        # logits shape: [seq_len * batch_size, vocab_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        probs = F.softmax(logits)
        # shape: [seq_len * batch_size, vocab_size]
        probs = probs.view(input.size(0), input.size(1), probs.size(1))
        # output shape: [seq_len, batch_size, vocab_size]
        return probs, hidden

    def init_hidden(self, batch_size=1):
        """
        initialise hidden state of rnn layer with zeroes
        """
        return (Variable(torch.zeros(self.args["num_layers"], batch_size, self.args["rnn_size"])),
                Variable(torch.zeros(self.args["num_layers"], batch_size, self.args["rnn_size"])))

    def save(self, checkpoint_path="model.ckpt"):
        checkpoint = {"args": self.args, "state_dict": self.state_dict()}
        with open(checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)
        logger.info("model saved: %s.", checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)

        model = cls(**checkpoint["args"])
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("model loaded: %s.", checkpoint_path)
        return model


def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    _, indices = torch.sort(probs)
    # set probabilities after top_n to 0
    probs[indices.data[:-top_n]] = 0
    sampled_index = torch.multinomial(probs, 1)
    return sampled_index


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
    generated = seed
    encoded = encode_text(seed)
    encoded = Variable(torch.from_numpy(encoded), volatile=True)
    model.eval()

    x = encoded[:-1].unsqueeze(1)
    # input shape: [seq_len, 1]
    state = model.init_hidden(1)
    # get rnn state due to seed sequence
    _, state = model.predict(x, state)

    next_index = encoded[-1:]
    for i in range(length):
        x = next_index.unsqueeze(1)
        # input shape: [1, 1]
        probs, state = model.predict(x, state)
        # output shape: [1, 1, vocab_size]
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += ID2CHAR[next_index.data[0]]

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

    # load or build model
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

    # make checkpoint directory
    make_dirs(args.checkpoint_path)
    model.save(args.checkpoint_path)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # training start
    num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    data_iter = batch_generator(encode_text(text), args.batch_size, args.seq_len)
    state = model.init_hidden(args.batch_size)
    logger.info("start of training.")
    time_train = time.time()

    for i in range(args.num_epochs):
        epoch = i + 1
        epoch_losses = torch.Tensor(num_batches)
        time_epoch = time.time()
        # training epoch
        for j in tqdm(range(num_batches), desc="epoch {}/{}".format(epoch, args.num_epochs)):
            # prepare inputs
            x, y = next(data_iter)
            x = Variable(torch.from_numpy(x)).t()
            y = Variable(torch.from_numpy(y)).t().contiguous()
            # reset state variables to remove their history
            state = tuple([Variable(var.data) for var in state])
            # prepare model
            model.train()
            model.zero_grad()
            # calculate loss
            logits, state = model.forward(x, state)
            loss = criterion(logits, y.view(-1))
            epoch_losses[j] = loss.data[0]
            # calculate gradients
            loss.backward()
            # clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), args.clip_norm)
            # apply gradient update
            optimizer.step()

        # logs
        duration_epoch = time.time() - time_epoch
        logger.info("epoch: %s, duration: %ds, loss: %.6g.",
                    epoch, duration_epoch, epoch_losses.mean())
        # checkpoint
        model.save(args.checkpoint_path)
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
    inference_model = Model.load(args.checkpoint_path)

    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, 3)


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
                              help="path to save or load model checkpoints; "
                                   "tensorboard logs will be saved in the same directory")
    train_parser.add_argument("--restore", nargs="?", default=False, const=True,
                              help="whether to restore from checkpoint_path "
                                   "or from another path if specified")
    train_parser.add_argument("--seq-len", type=int, default=64,
                              help="sequence length of inputs and outputs")
    train_parser.add_argument("--embedding-size", type=int, default=32,
                              help="character embedding size")
    train_parser.add_argument("--rnn-size", type=int, default=128,
                              help="size of rnn cell")
    train_parser.add_argument("--num-layers", type=int, default=2,
                              help="number of rnn layers")
    train_parser.add_argument("--drop-rate", type=float, default=0.,
                              help="dropout rate for rnn layers")
    train_parser.add_argument("--learning-rate", type=float, default=0.001,
                              help="learning rate")
    train_parser.add_argument("--clip-norm", type=float, default=5.,
                              help="max norm to clip gradient")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="training batch size")
    train_parser.add_argument("--num-epochs", type=int, default=32,
                              help="number of epochs for training")
    train_parser.set_defaults(main=train_main)

    # generate args
    generate_parser = subparsers.add_parser("generate", help="generate text from trained model")
    generate_parser.add_argument("--checkpoint-path", required=True,
                                 help="path to load model checkpoints")
    generate_parser.add_argument("--text-path", required=True,
                                 help="path of text file to generate seed")
    generate_parser.add_argument("--seed", default=None,
                                 help="seed character sequence")
    generate_parser.add_argument("--length", type=int, default=1024,
                                 help="length of character sequence to generate")
    generate_parser.set_defaults(main=generate_main)

    args = arg_parser.parse_args()
    logger.debug("call: %s", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s", args)

    try:
        args.main(args)
    except Exception as e:
        logger.exception(e)
