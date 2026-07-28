import platform
import sys

import matplotlib
import numpy
import pandas
import PIL
import sklearn
import torch
import torchvision
import wandb
import yaml


if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "Formal runs require Python 3.11; found "
        f"{platform.python_version()}"
    )

print("Python/PyTorch environment check")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pillow:", PIL.__version__)
print("PyYAML:", yaml.__version__)
print("W&B:", wandb.__version__)
print("Environment check passed")
