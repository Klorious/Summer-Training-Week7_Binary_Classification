# Python 3.11 reproducible environment

The final formal reruns must be executed with:

- Python 3.11.x
- PyTorch 2.6
- Weights & Biases 0.28.0
- NVIDIA Tesla V100-SXM2-32GB
- Random seed 20260725

## Create the Conda environment

```bash
conda env create -f environment.yml
conda activate training-unit7-py311
python check_environment.py
```

PyTorch GPU packages depend on the host driver and CUDA runtime. If the
standard installation does not detect the GPU, use the installation command
recommended by the official PyTorch selector for the current CUDA platform,
then rerun `python check_environment.py`.

## W&B login

```bash
wandb login
```

Never commit a W&B API key, `.netrc`, `.env`, or the local `wandb/` directory.
