#!/bin/bash
# keras
python keras_model.py train --text=data/tinyshakespeare.txt --checkpoint=checkpoints/keras_tinyshakespeare/model.hdf5
python keras_model.py generate --text=data/tinyshakespeare.txt --checkpoint=checkpoints/keras_tinyshakespeare/model.hdf5

python keras_model.py train --text=data/shakespeare.txt --checkpoint=checkpoints/keras_shakespeare/model.hdf5
python keras_model.py generate --text=data/shakespeare.txt --checkpoint=checkpoints/keras_shakespeare/model.hdf5

# tensorflow
python tf_model.py train --text=data/tinyshakespeare.txt --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt
python tf_model.py generate --text=data/tinyshakespeare.txt --checkpoint=checkpoints/tf_tinyshakespeare/model.ckpt

python tf_model.py train --text=data/shakespeare.txt --checkpoint=checkpoints/tf_shakespeare/model.ckpt
python tf_model.py generate --text=data/shakespeare.txt --checkpoint=checkpoints/tf_shakespeare/model.ckpt

# pytorch
python pytorch_model.py train --text=data/tinyshakespeare.txt --checkpoint=checkpoints/pytorch_tinyshakespeare/model.ckpt
python pytorch_model.py generate --text=data/tinyshakespeare.txt --checkpoint=checkpoints/pytorch_tinyshakespeare/model.ckpt

python pytorch_model.py train --text=data/shakespeare.txt --checkpoint=checkpoints/pytorch_shakespeare/model.ckpt
python pytorch_model.py generate --text=data/shakespeare.txt --checkpoint=checkpoints/pytorch_shakespeare/model.ckpt
