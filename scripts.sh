#!/bin/bash
# keras
uv run python keras_model.py train --checkpoint-path=checkpoints/keras_tinyshakespeare/model.keras --text-path=data/tinyshakespeare.txt
uv run python keras_model.py generate --checkpoint-path=checkpoints/keras_tinyshakespeare/model.keras --text-path=data/tinyshakespeare.txt

uv run python keras_model.py train --checkpoint-path=checkpoints/keras_shakespeare/model.keras --text-path=data/shakespeare.txt
uv run python keras_model.py generate --checkpoint-path=checkpoints/keras_shakespeare/model.keras --text-path=data/shakespeare.txt

# tensorflow
uv run python tf_model.py train --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt --text-path=data/tinyshakespeare.txt
uv run python tf_model.py generate --checkpoint-path=checkpoints/tf_tinyshakespeare/model.ckpt --text-path=data/tinyshakespeare.txt

uv run python tf_model.py train --checkpoint-path=checkpoints/tf_shakespeare/model.ckpt --text-path=data/shakespeare.txt
uv run python tf_model.py generate --checkpoint-path=checkpoints/tf_shakespeare/model.ckpt --text-path=data/shakespeare.txt

# pytorch
uv run python pytorch_model.py train --checkpoint-path=checkpoints/pytorch_tinyshakespeare/model.ckpt --text-path=data/tinyshakespeare.txt
uv run python pytorch_model.py generate --checkpoint-path=checkpoints/pytorch_tinyshakespeare/model.ckpt --text-path=data/tinyshakespeare.txt

uv run python pytorch_model.py train --checkpoint-path=checkpoints/pytorch_shakespeare/model.ckpt --text-path=data/shakespeare.txt
uv run python pytorch_model.py generate --checkpoint-path=checkpoints/pytorch_shakespeare/model.ckpt --text-path=data/shakespeare.txt
