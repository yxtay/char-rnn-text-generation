# Character Embeddings Recurrent Neural Network Text Generation Models

Inspired by [Andrej Karpathy](https://github.com/karpathy/)'s 
[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

This repository attempts to replicate the models, with slight modifications, in different python deep learning frameworks.

## Frameworks

- Keras: [`keras_model.py`](keras_model.py)
- TensorFlow: [`tf_model.py`](tf_model.py)
- PyTorch: [`pytorch_model.py`](pytorch_model.py)
- Chainer: [`chainer_model.py`](chainer_model.py)
- MXNet: [`mxnet_model.py`](mxnet_model.py)
- CNTK
- Caffe

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
conda env create -f=environment.yml
```

## Usage

### Training

```
usage: <framework>_model.py train [-h] --text-path TEXT_PATH --checkpoint-path
                                  CHECKPOINT_PATH [--restore [RESTORE]]
                                  [--seq-len SEQ_LEN]
                                  [--embedding-size EMBEDDING_SIZE]
                                  [--rnn-size RNN_SIZE] [--num-layers NUM_LAYERS]
                                  [--drop-rate DROP_RATE]
                                  [--learning-rate LEARNING_RATE]
                                  [--clip-norm CLIP_NORM] 
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
```

Example:

```bash
python tf_model.py train \
    --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt \
    --text=data/tinyshakespeare.txt
```

### Text Generation

```
usage: <framework>_model.py generate [-h] --checkpoint-path CHECKPOINT_PATH
                                     (--text-path TEXT_PATH | --seed SEED)
                                     [--length LENGTH] [--top-n TOP_N]

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
```

Example:

```bash
python tf_model.py generate \
    --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt \
    --seed="KING RICHARD"
```

Sample output:

```
KING RICHARDIIIIl II I tell thee,
As I have no mark of his confection,
The people so see my son.

SEBASTIAN:
I have men's man in the common to his sounds,
And so she said of my soul, and to him,
And too marry his sun their commanded
As thou shalt be alone too means
As he should to thy sensess so far to mark of
these foul trust them fringer whom, there would he had
As the word of merrous and subject.

GLOUCESTER:
A spack, a service the counsel son and here.
What is a misin the wind and to the will
And shall not streaks of this show into all heard.

KING EDIN YORK:
I will be suppet on himself tears as the sends.

KING EDWARD IV:
No looks and them, and while, a will, when this way.

BAPTHIO:
A mortain and me to the callant our souls
And the changed and such of the son.

CORIOLANUS:
I will, so show me with the child to the could sheep
To beseence, and shall so so should but hear
Than him with her fair to be that soul,
Whishe it is no meach of my lard and
And this, and with my love and the senter'd with marked
And her should
```
