# Character Embedding Recurrent Neural Network Text Generation Models

Inspired by Andrej Karpathy's [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

This repository attempts to replicate the models, with slight modifications, in different python deep learning frameworks.

## Frameworks

- [Keras](#keras)
- [Tensorflow](#tensorflow)
- PyTorch
- Caffe
- MXNet
- CNTK
- Chainer

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

## Keras

### Training

Usage:

```
usage: keras_model.py train [-h] --text-path TEXT_PATH --checkpoint-path
                            CHECKPOINT_PATH [--restore [RESTORE]]
                            [--seq-len SEQ_LEN]
                            [--embedding-size EMBEDDING_SIZE]
                            [--rnn-size RNN_SIZE] [--num-layers NUM_LAYERS]
                            [--drop-rate DROP_RATE]
                            [--learning-rate LEARNING_RATE]
                            [--batch-size BATCH_SIZE]
                            [--num-epochs NUM_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --text-path TEXT_PATH
                        path of text file for training
  --checkpoint-path CHECKPOINT_PATH
                        path to save or load model checkpoints; tensorboard
                        logs will be saved in the same directory
  --restore [RESTORE]   whether to restore from checkpoint_path or from
                        another path if specified
  --seq-len SEQ_LEN     sequence length of inputs and outputs
  --embedding-size EMBEDDING_SIZE
                        character embedding size
  --rnn-size RNN_SIZE   size of rnn cell
  --num-layers NUM_LAYERS
                        number of rnn layers
  --drop-rate DROP_RATE
                        dropout rate for rnn layers
  --learning-rate LEARNING_RATE
                        learning rate
  --batch-size BATCH_SIZE
                        training batch size
  --num-epochs NUM_EPOCHS
                        number of epochs for training
```

Example:

```bash
python keras_model.py train \
    --text-path=data/tinyshakespeare.txt \
    --checkpoint-path=checkpoints/keras_tinyshakespeare/model.hdf5
```

### Text Generation

Usage:

```
usage: keras_model.py generate [-h] --checkpoint-path CHECKPOINT_PATH
                               --text-path TEXT_PATH [--seed SEED]
                               [--length LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-path CHECKPOINT_PATH
                        path to load model checkpoints
  --text-path TEXT_PATH
                        path of text file to generate seed
  --seed SEED           seed character sequence
  --length LENGTH       length of character sequence to generate
```

Example:

```bash
python keras_model.py generate \
    --text-path=data/tinyshakespeare.txt \
    --checkpoint-path=checkpoints/keras_tinyshakespeare/model.hdf5
```

## Tensorflow

### Training

Example:

```bash
python tf_model.py train \
    --text-path=data/tinyshakespeare.txt \
    --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt
```

### Text Generation

Example:

```bash
python tf_model.py generate \
    --text-path=data/tinyshakespeare.txt \
    --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt
```
