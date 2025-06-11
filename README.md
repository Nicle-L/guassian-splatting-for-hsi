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
sh ./scripts/tran.sh /path/to/your/dataset                #for 2DGS reference[14]
sh ./scripts/tran_only_color.sh /path/to/your/dataset     #for 2DGS + color weight W
sh ./scripts/tran_hsi.sh /path/to/your/dataset            #for 2DGS + color weight W + adaptive resampling module
sh ./scripts/tran_only_color.sh /path/to/your/dataset     #for 2DGS + color weight W + reusing cross-band information
sh ./scripts/train_hsi_inter.sh /path/to/your/dataset     #for 2DGS + color weight W + adaptive resampling module + reusing cross-band information(GS-HSI)
```
### Botswana Dataset
```bash
sh ./scripts/tran.sh /path/to/your/dataset                #for 2DGS reference[14]
sh ./scripts/tran_only_color.sh /path/to/your/dataset     #for 2DGS + color weight W
sh ./scripts/tran_hsi.sh /path/to/your/dataset            #for 2DGS + color weight W + adaptive resampling module
sh ./scripts/tran_only_color.sh /path/to/your/dataset     #for 2DGS + color weight W + reusing cross-band information
sh ./scripts/train_hsi_inter.sh /path/to/your/dataset     #for 2DGS + color weight W + adaptive resampling module + reusing cross-band information(GS-HSI)
```
## HSI Compression
After overfitting the image, we load the checkpoints from the image representation and apply quantization-aware training to obtain the image compression results of GaussianImage models.

### Houston Dataset
```bash
sh ./scripts/train_quant.sh        /path/to/your/dataset         #for 2DGS reference[14] + attribute-aware quantization module
sh ./scripts/train_quant_color.sh  /path/to/your/dataset         #for 2DGS + color weight W + attribute-aware quantization module
sh ./scripts/train_quant_hsi.sh    /path/to/your/dataset         #for 2DGS + color weight W + adaptive resampling module + attribute-aware quantization module
sh ./scripts/train_quant_band.sh   /path/to/your/dataset         #for 2DGS + color weight W+ reusing cross-band information + attribute-aware quantization module
sh ./scripts/train_quant_inter.sh  /path/to/your/dataset         #for 2DGS + color weight W+ reusing cross-band information + adaptive resampling module + attribute-aware quantization module

``` 
### Botswana Dataset
```bash
sh ./scripts/train_quant.sh        /path/to/your/dataset         #for 2DGS reference[14] + attribute-aware quantization module
sh ./scripts/train_quant_color.sh  /path/to/your/dataset         #for 2DGS + color weight W + attribute-aware quantization module
sh ./scripts/train_quant_hsi.sh    /path/to/your/dataset         #for 2DGS + color weight W + adaptive resampling module + attribute-aware quantization module
sh ./scripts/train_quant_band.sh   /path/to/your/dataset         #for 2DGS + color weight W+ reusing cross-band information + attribute-aware quantization module
sh ./scripts/train_quant_inter.sh  /path/to/your/dataset         #for 2DGS + color weight W+ reusing cross-band information + adaptive resampling module + attribute-aware quantization module
```
Acknowledgments
Our code is developed based on gsplat, a lightweight and modular Gaussian Splatting library.
We also thank vector-quantize-pytorch for providing the framework to implement residual vector quantization.





