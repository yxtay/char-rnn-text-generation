# Agent Instructions

This repository contains character-level RNN text generation models implemented in several deep learning frameworks.

## Environment Management
We use `uv` for dependency management. To set up the environment, run:
```bash
uv sync
```

## Supported Frameworks
- **PyTorch**: `pytorch_model.py`
- **TensorFlow**: `tf_model.py`
- **Keras**: `keras_model.py`

Other frameworks (Chainer, MXNet, CNTK, Caffe) have been removed or are no longer supported.

## Training and Generation
Each model file can be run directly. For example:
```bash
uv run python pytorch_model.py train --checkpoint-path checkpoints/model.ckpt --text-path data/tinyshakespeare.txt
uv run python pytorch_model.py generate --checkpoint-path checkpoints/model.ckpt --text-path data/tinyshakespeare.txt
```

## Adding New Models
When adding a new framework implementation:
1. Follow the existing pattern of using `utils.py` for data processing and CLI.
2. Use `logger.py` for consistent logging.
3. Update `pyproject.toml` dependencies using `uv add`.
