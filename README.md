# TV-INR

This is the official PyTorch implementation of TV-INR. 

# Installation

First, create a conda environment and activate it.

```bash
conda create --name tvinr python=3.8  --no-default-packages
conda activate tvinr
```


Then, install the requirements

```bash
pip install -r requirements_pip.txt
pip install -e .
```

## Usage

Our project uses wandb, please fill ./configs/wandb/wandb.yaml. As an example you can run experiment for Traffic dataset with L=200.

```bash
cd ./run
python main_wandb.py --cfg ./configs/traffic_tvinr_200.yaml
```