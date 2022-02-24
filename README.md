
# A general approach for automatic segmentation of pneumonia, pulmonary nodule and tuberculosis on CT images

[![standard-readme compliant](https://img.shields.io/badge/Readme-standard-brightgreen.svg?style=flat-square)](https://github.com/JD910/ESLN/blob/main/README.md)
![](https://img.shields.io/badge/Pytorch-1.7.1-brightgreen.svg?style=flat-square)

<div align=left><img width="610" height="365" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/flowchartV1.jpg"/></div><br />

### Introduction of the *GSAL* model for automatic segmentation of multiple types of lung lesions.

* The training procedure is presented in this repository.<br />

* main.py is used to start the proposed model.<br />

* GDModels folder stores the generator network G and the discriminator network D:
  > Dmodel: Definition of the discriminator network with self-supervised roration loss.
  > 
  > Gmodel: Definition of the generator network with cascaded CPFE and dual attention modules.
  > 
  > Attention: Definition of the spatial attention network and the channel-wise attention network.

* Hyper-parameters in the main function:
  > input_dir_train: Path of your training dataset.
  > 
  > input_dir_test: Path of your test dataset.
  > 
  > gpu_ids: Number of your GPUs.
  > 
  > d_iter: The number of times the D network is trained per batch.
  > 
  > gpu_ids: Number of your GPUs.
  > 
  > Height and Width: Definition of the size of input images.
