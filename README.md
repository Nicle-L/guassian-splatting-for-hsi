# Gaussian Splatting for Image Compression

## Overview

Our `gsplat` library is developed based on the open-source implementation from [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat).  
This is a concise and easily extensible Gaussian Splatting library, which we further adapted for image compression tasks.

## Dataset
File shared via Netflix: dataset_gs 
Link: https://pan.baidu.com/s/1npBgk8BQm2C-nBmpqzHTQw?pwd=iaca Extract code: iaca

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
## HSI Representation

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

### BD-Rate、BD-PSNR(dB)、BD-MS-SSIM(dB)

Table 1: BD-RATE (%), BD-PSNR (dB), and BD-MS-SSIM (dB) Performance Comparison on the Botswana Dataset
| Method         | BD-Rate (%) | BD-PSNR (dB) | BD-MS-SSIM (dB) |
|----------------|-------------|--------------|-----------------|
| PCA_JPEG2000   | +0.26%      | -1.39        | -0.56           |
| BPG            | +3.01%      | +1.46        | +3.03           |
| Mean_scale     | +41.27%     | -10.19       | +0.60           |
| Context        | +18.26%     | -13.85       | +0.23           |
| Cheng          | +23.47%     | -7.19        | +3.02           |
| CNN_HSI        | +15.18%     | -7.83        | +0.75           |
| SSCNet         | +41.93%     | -6.48        | +4.27           |
| CENet          | -9.07%      | +1.37        | +2.27           |
| FHNerf         | -16.29%     | +1.76        | +2.66           |
| 2DGS           | -0.86%      | +2.24        | +3.09           |
| GS-HSI (Ours)  | -0.89%      | +2.65        | +3.26           |

Table 2: BD-RATE (%), BD-PSNR (dB), and BD-MS-SSIM (dB) Performance Comparison on the Houston Dataset
| Method         | BD-Rate (%) | BD-PSNR (dB) | BD-MS-SSIM (dB) |
|----------------|-------------|--------------|-----------------|
| PCA_JPEG2000   | +1.65%      | -0.22        | +0.67           |
| BPG            | -15.91%     | +0.84        | +1.56           |
| Mean_scale     | -30.13%     | -5.40        | -0.04           |
| Context        | -4.21%      | -4.23        | +1.25           |
| Cheng          | -11.23%     | -3.39        | +2.36           |
| CNN_HSI        | -3.56%      | -4.70        | +0.48           |
| SSCNet         | +13.34%     | -7.24        | +3.63           |
| CENet          | +8.85%      | +0.71        | +2.20           |
| FHNerf         | -16.29%     | +1.36        | +2.24           |
| 2DGS           | -21.01%     | +1.76        | +2.77           |
| GS-HSI (Ours)  | -20.83%     | +2.41        | +3.70           |


## Acknowledgments
Our code is developed based on gsplat, a lightweight and modular Gaussian Splatting library.
We also thank vector-quantize-pytorch for providing the framework to implement residual vector quantization.





