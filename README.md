<!-- <div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), developed by [Ultralytics](https://ultralytics.com),
is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces
new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image
classification tasks.

To request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

<div align="center">
    <a href="https://github.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-producthunt.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-facebook.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
  </div>
</div> -->
<!-- width = "600" height = "450"  -->

## <div align="center">YOLOngv8: From Imbalanced to Accurate Object Detection in Long Tailed iSAID Dataset</div>

## Requirements
YOLOv8 requires ultralytics package which requires Python>=3.7 and Pytorch>=1.7. Setup Environment with \
``` 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install ultralytics 
```

Run validation of model by specifying the checkpoint for ```PATH/TO/CHECKPOINT``` in 

``` python validate.py --checkpoint PATH/TO/CHECKPOINT ```

## Model Architecture</div>
<img src="https://github.com/salwaalkhatib/ultralytics/blob/main/readme_figs/yolov8-arch.png" alt="" align=center /> 


## Prototype based Contrastive loss</div>
<img src="https://github.com/salwaalkhatib/ultralytics/blob/main/readme_figs/contr_loss.png" alt="" align=center /> 


### tSNE plots of features</div>
<img src="https://github.com/salwaalkhatib/ultralytics/blob/main/readme_figs/tsne.png" width = "600" height = "450" alt="" align=center /> 

tSNE plot of features before and contrastive learning. 