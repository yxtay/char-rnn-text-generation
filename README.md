# Character Embeddings Recurrent Neural Network Text Generation Models

Inspired by [Andrej Karpathy](https://github.com/karpathy/)'s 
[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

This repository attempts to replicate the models, with slight modifications, in different python deep learning frameworks.

## Frameworks

- Keras: [`keras_model.py`](keras_model.py)
- TensorFlow: [`tf_model.py`](tf_model.py)
- PyTorch: [`pytorch_model.py`](pytorch_model.py)

## Default Model Specification

| Layer Type | Output Shape  | Param # | Remarks                            |
|------------|---------------|---------|------------------------------------|
| Embedding  | (64, 64, 32)  | 3136    | vocab size: 98, embedding size: 32 |
| Dropout    | (64, 64, 32)  | 0       | dropout rate: 0.0                  |
| LSTM       | (64, 64, 128) | 82432   | output size: 128                   |
| Dropout    | (64, 64, 128) | 0       | dropout rate: 0.0                  |
| LSTM       | (64, 64, 128) | 131584  | output size: 128                   |
| Dropout    | (64, 64, 128) | 0       | dropout rate: 0.0                  |
| Dense      | (64, 64, 98)  | 12642   | output size: 98                    |

### Training Specification

- Batch size: 64
- Sequence length: 64
- Number of epochs: 32
- Learning rate: 0.001
- Max gradient norm: 5.0

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/char-rnn-text-generation.git && cd char-rnn-text-generation

# install dependencies with uv
uv sync
```

## Usage

### Training

```
usage: <framework>_model.py train [-h] --checkpoint-path CHECKPOINT_PATH 
                                  --text-path TEXT_PATH
                                  [--restore [RESTORE]]
                                  [--seq-len SEQ_LEN]
                                  [--embedding-size EMBEDDING_SIZE]
                                  [--rnn-size RNN_SIZE] 
                                  [--num-layers NUM_LAYERS]
                                  [--drop-rate DROP_RATE]
                                  [--learning-rate LEARNING_RATE]
                                  [--clip-norm CLIP_NORM] 
                                  [--batch-size BATCH_SIZE]
                                  [--num-epochs NUM_EPOCHS]
                                  [--log-path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-path CHECKPOINT_PATH
                        path to save or load model checkpoints
  --text-path TEXT_PATH
                        path of text file for training
  --restore [RESTORE]   whether to restore from checkpoint_path or from
                        another path if specified
  --seq-len SEQ_LEN     sequence length of inputs and outputs (default: 64)
  --embedding-size EMBEDDING_SIZE
                        character embedding size (default: 32)
  --rnn-size RNN_SIZE   size of rnn cell (default: 128)
  --num-layers NUM_LAYERS
                        number of rnn layers (default: 2)
  --drop-rate DROP_RATE
                        dropout rate for rnn layers (default: 0.0)
  --learning-rate LEARNING_RATE
                        learning rate (default: 0.001)
  --clip-norm CLIP_NORM
                        max norm to clip gradient (default: 5.0)
  --batch-size BATCH_SIZE
                        training batch size (default: 64)
  --num-epochs NUM_EPOCHS
                        number of epochs for training (default: 32)
  --log-path LOG_PATH   path of log file (default: main.log)
```

Example:

```bash
uv run python tf_model.py train \
    --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt \
    --text-path=data/tinyshakespeare.txt
```

### Text Generation

```
usage: <framework>_model.py generate [-h] --checkpoint-path CHECKPOINT_PATH
                                     (--text-path TEXT_PATH | --seed SEED)
                                     [--length LENGTH] [--top-n TOP_N]
                                     [--log-path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-path CHECKPOINT_PATH
                        path to load model checkpoints
  --text-path TEXT_PATH
                        path of text file to generate seed
  --seed SEED           seed character sequence
  --length LENGTH       length of character sequence to generate (default:
                        1024)
  --top-n TOP_N         number of top choices to sample (default: 3)
  --log-path LOG_PATH   path of log file (default: main.log)
```

Example:

```bash
uv run python tf_model.py generate \
    --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt \
    --seed="KING RICHARD"
```
