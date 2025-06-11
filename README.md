# Gaussian Splatting for Image Compression

## Overview

Our `gsplat` library is developed based on the open-source implementation from [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat).  
This is a concise and easily extensible Gaussian Splatting library, which we further adapted for image compression tasks.

## Environment & Dependencies

To set up the environment, make sure the following Python packages are installed:

- `numpy`  
- `pandas`  
- `pillow`  
- `pytorch-msssim`  
- `PyYAML`  
- `tqdm`  
- `vector-quantize-pytorch`  

You can install them via pip:

```bash
pip install numpy/pandas/ pillow/ pytorch-msssim/ PyYAML/ tqdm/ vector-quantize-pytorch/
```
## Image Representation

To perform an overfitting-based hyperspectral image representation using different variants of the “GS-HSI” model, run the following script:

### Houston Dataset

```bash
sh ./scripts/gaussianimage_cholesky/kodak.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/kodak.sh /path/to/your/dataset
sh ./scripts/3dgs/kodak.sh /path/to/your/dataset'
```
### Botswana Dataset
```bash
sh ./scripts/gaussianimage_cholesky/div2k.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/div2k.sh /path/to/your/dataset
sh ./scripts/3dgs/div2k.sh /path/to/your/dataset
```
## Image Compression
After overfitting the image, we load the checkpoints from image representation and apply quantization-aware training to obtain the image compression results of GaussianImage models.

### Houston Dataset
```bash
sh ./scripts/gaussianimage_cholesky/kodak_comp.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/kodak_comp.sh /path/to/your/dataset
```
### Botswana Dataset
```bash
DIV2K Dataset
sh ./scripts/gaussianimage_cholesky/div2k_comp.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/div2k_comp.sh /path/to/your/dataset
```
Acknowledgments
Our code is developed based on gsplat, a lightweight and modular Gaussian Splatting library.
We also thank vector-quantize-pytorch for providing the framework to implement residual vector quantization.





