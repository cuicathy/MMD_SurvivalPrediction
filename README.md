# MMD_SurvivalPrediction

The repository is the official PyTorch implementation of the paper " 
Survival Prediction of Brain Cancer with Incomplete Radiology, Pathology, Genomics, and Demographic Data". 
The paper link is https://arxiv.org/pdf/2203.04419.pdf


## Prerequisites
* NVIDIA GPU + CUDA + cuDNN + python + pytorch
* We used python >= 3.6, CUDA 11.4, pytorch >= 1.7.0 

## Usage
* Unimodal embeddings and data splits have been provided https://drive.google.com/drive/folders/1dsG7Ab4dNG7IdRZEax6ABIVCC2J8x-w8?usp=share_link
* Run python main.py to do the multimodal survival prediction   
* Key component of the radiomics feature extraction has been uploaded (Radiomics.py, 2DRadiomics_params.py)
* 
## 
If you use this code, please cite our work, the reference is
```
@article{cui2022survival,
  title={Survival Prediction of Brain Cancer with Incomplete Radiology, Pathology, Genomics, and Demographic Data},
  author={Cui, Can and Liu, Han and Liu, Quan and Deng, Ruining and Asad, Zuhayr and Zhao, Yaohong WangShilin and Yang, Haichun and Landman, Bennett A and Huo, Yuankai},
  journal={arXiv preprint arXiv:2203.04419},
  year={2022}
}

If you have any problem, feel free to contact can.cui.1@vanderbilt.edu.
