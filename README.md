# dl-cs-dynamic
A Python package for training unrolled neural networks for dynamic MRI reconstruction.

Written by Christopher Sandino (sandino@stanford.edu), 2021.

## Installation
To install this package, I recommend conda. If you do not already have conda installed on your system, then I recommend downloading and installing [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).

After you have installed conda, create your conda environment from the provided `environment.yaml` using the following command:
```bash
conda env create -f environment.yaml
```

Then, activate the environment:
```bash
conda activate dl-cs
```

And finally, install the dl-cs-dynamic package:
```bash
flit install -s
```

## Training
To launch a training, you must first create a config file containing the hyperparameters of the network. You can find a basic config file in `configs/basic/example.yaml`. Once you have created a config file, you can launch a training session: 

```bash
python3 ./scripts/train.py --config-file <path to config file> --devices 0
```

## Inference
Once you have a trained network, you can run inference on a CFL dataset using the following: 

```bash
python3 ./scripts/reconstruct.py --config-file <path to config file used for training> --ckpt <path to model checkpoint> --directory <path to directory containing k-space and sensitivity maps in CFL format>
```
