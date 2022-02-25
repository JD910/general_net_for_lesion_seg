
# A general approach for automatic segmentation of pneumonia, pulmonary nodule and tuberculosis

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
  > Height and Width: Definition of the size of input images.


## *Supp_S1*
<div align=left><img width="800" height="258" src="https://github.com/JD910/LNHG/blob/main/Segmentation/Images/Fig_F1.jpg"/></div><br />

**Fig. F1. Explanation of the Workflow of the Proposed LNHG Model. The input is the original CT image, and *Block1–Block8* denote the network layers (with the same color) presented in Figure 3 in the manuscript. The feature map output by each block showed in the figure is the mean of all the feature maps.**<br />

<div align=left><img width="610" height="338" src="https://github.com/JD910/LNHG/blob/main/Segmentation/Images/Fig-github.jpg"/></div><br />

**Fig. F2. Example illustrating how the final lung nodule image is produced based on the output of WGAN-GP and Faster R-CNN branches. The false-positive candidates produced by the WGAN-GP branch is marked in a red oval for better display. By fusing the output of the two branches, the false-positive candidates are eliminated, and the final lung nodule image is produced.**<br />

<div align=left><img width="358" height="380" src="https://github.com/JD910/LNHG/blob/main/Segmentation/Images/Fig_F3.jpg"/></div><br />

**Fig. F3. Example illustrating how the intra-nodular heterogeneity images are generated from the output of the *LNHG* model. Three nodules are presented: (a) original CT images, (b) output of the *LNHG* model, and (c) intra-nodular heterogeneity images converted from the output of the *LNHG* model. The original CT image is superimposed on the output image of the *LNHG* model. Therefore, the original CT image is shown on (c), except for the generated nodule area. The images in (c) are used for the reader study.**<br />
