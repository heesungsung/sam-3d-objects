# Setup for RTX 50xx (Blackwell Architecture)

## Build on Ubuntu 24.04.3 w/ RTX 5090

Last updated @ Nov 21, 2025 ([Link](https://github.com/facebookresearch/sam-3d-objects/issues/15#issuecomment-3560650855))


## Version

| Package | Repository | Version/Commit |
|---------|-----------|----------------|
| sam-3d-objects | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/pytorch3d) | 0e3d254 |
| pytorch3d | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | V0.7.8 |
| kaolin | [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) | v0.18.0 |

Clone them first and be aware of the submodules.
e.g.,
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout V0.7.8
git submodule update --init --recursive

git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout v0.18.0
git submodule update --init --recursive
```

## Build


```bash
## Run on project_root
# Create environment
conda create -n sam3d-objects python=3.11
conda activate sam3d-objects

# Install torch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Update dependency
conda env update -f sam3d-objects-single.yml

# Refresh (This is necessary.)
conda deactivate
conda activate sam3d-objects

## Run on pytorch3d
# Build
python setup.py install

## Run on kaolin
# Build
pip install -e . --no-build-isolation

## Run on project_root
# Test!
python demo.py
```
