
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


## *Supp_Materials*
<div align=left><img width="800" height="479" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/Fig_S1_covid-bilateral.jpg"/></div><br />

**Fig. S1. Segmentation of a COVID-19 lesion with the CT manifestation of bilateral GGO. (a) Original images and segmentation obtained by (b) the proposed model and (c) radiologists.**<br />

<div align=left><img width="800" height="479" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/Fig_S2_non-covid-bilateralGGN.jpg"/></div><br />

**Fig. S2. Segmentation of a non-COVID-19 lesion in which the CT manifestation is bilateral GGO. (a) Original images and segmentation obtained by (b) the proposed model and (c) radiologists.**<br />

<div align=left><img width="800" height="479" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/Fig_S3_Non-covid-Single.jpg"/></div><br />

**Fig. S3. Segmentation of a non-COVID-19 lesion in which the CT manifestation are unilateral mixed GGO and consolidation. (a) Original images and segmentation obtained by (b) the proposed model and (c) radiologists.**<br />

<div align=left><img width="800" height="479" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/Fig_S4_JP.jpg"/></div><br />

**Fig. S4. Juxta-pleural nodule segmentation results. (a) Original images and segmentation obtained by (b) the proposed model and (c) radiologists. Nodule is marked with arrows.**<br />

<div align=left><img width="800" height="479" src="https://github.com/JD910/general_net_for_lesion_seg/blob/main/GDModels/Fig_S5_JV.jpg"/></div><br />

**Fig. S5. Juxta-vascular nodule results. (a) Original images and segmentation obtained by (b) the proposed model and (c) radiologists. False positives that can be recognized by the naked eye in b(3) are marked with arrow. However, two other nodules that are not delineated by the radiologist in b(5) are recognized by the proposed model (with arrow)**<br />
