#!/bin/bash
# using nietzsche.txt
python keras_model.py train --text=data/nietzsche.txt --checkpoint=checkpoints/keras_nietzsche.hdf5 --tensorboard=tensorboard/keras_nietzsche
python keras_model.py generate --text=data/nietzsche.txt --checkpoint=checkpoints/keras_nietzsche.hdf5

# using tinyshakespeare.txt
python keras_model.py train --text=data/tinyshakespeare.txt --checkpoint=checkpoints/keras_tinyshakespeare.hdf5 --tensorboard=tensorboard/keras_tinyshakespeare
python keras_model.py generate --text=data/tinyshakespeare.txt --checkpoint=checkpoints/keras_tinyshakespeare.hdf5

# using shakespeare.txt
python keras_model.py train --text=data/shakespeare.txt --checkpoint=checkpoints/keras_shakespeare.hdf5 --tensorboard=tensorboard/keras_shakespeare
python keras_model.py generate --text=data/shakespeare.txt --checkpoint=checkpoints/keras_shakespeare.hdf5
