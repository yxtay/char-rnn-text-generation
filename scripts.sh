#!/bin/bash
# keras
python keras_model.py train --checkpoint=checkpoints/keras_tinyshakespeare/model.hdf5 --text=data/tinyshakespeare.txt
python keras_model.py generate --checkpoint=checkpoints/keras_tinyshakespeare/model.hdf5 --text=data/tinyshakespeare.txt

python keras_model.py train --checkpoint=checkpoints/keras_shakespeare/model.hdf5 --text=data/shakespeare.txt
python keras_model.py generate --checkpoint=checkpoints/keras_shakespeare/model.hdf5 --text=data/shakespeare.txt

# tensorflow
python tf_model.py train --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt --text=data/tinyshakespeare.txt
python tf_model.py generate --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt --text=data/tinyshakespeare.txt

python tf_model.py train --checkpoint=checkpoints/tf_shakespeare/model.ckpt --text=data/shakespeare.txt
python tf_model.py generate --checkpoint=checkpoints/tf_shakespeare/model.ckpt --text=data/shakespeare.txt

# pytorch
python pytorch_model.py train --checkpoint=checkpoints/pytorch_tinyshakespeare/model.ckpt --text=data/tinyshakespeare.txt
python pytorch_model.py generate --checkpoint=checkpoints/pytorch_tinyshakespeare/model.ckpt --text=data/tinyshakespeare.txt

python pytorch_model.py train --checkpoint=checkpoints/pytorch_shakespeare/model.ckpt --text=data/shakespeare.txt
python pytorch_model.py generate --checkpoint=checkpoints/pytorch_shakespeare/model.ckpt --text=data/shakespeare.txt

# chainer
python chainer_model.py train --checkpoint=checkpoints/chainer_tinyshakespeare/model.npz --text=data/tinyshakespeare.txt
python chainer_model.py generate --checkpoint=checkpoints/chainer_tinyshakespeare/model.npz --text=data/tinyshakespeare.txt

python chainer_model.py train --checkpoint=checkpoints/chainer_shakespeare/model.npz --text=data/shakespeare.txt
python chainer_model.py generate --checkpoint=checkpoints/chainer_shakespeare/model.npz --text=data/shakespeare.txt

# mxnet
python mxnet_model.py train --checkpoint=checkpoints/mxnet_tinyshakespeare/model.params --text=data/tinyshakespeare.txt
python mxnet_model.py generate --checkpoint=checkpoints/mxnet_tinyshakespeare/model.params --text=data/tinyshakespeare.txt

python mxnet_model.py train --checkpoint=checkpoints/mxnet_shakespeare/model.params --text=data/shakespeare.txt
python mxnet_model.py generate --checkpoint=checkpoints/mxnet_shakespeare/model.params --text=data/shakespeare.txt
