# Saliency_BackwardForwardFeatures_VGG16

Saliency Detection by Forward and Backward Cues in Deep-CNN, IEEE ICIP 2017

These codes here are revised version of the top-down saliency part of the work "Saliency Detection by Forward and Backward Cues in Deep-CNN,  IEEE ICIP 2017", which can be accessed from [here](https://arxiv.org/abs/1703.00152). 

The saliency computation code here can be run just CPU without the need of GPU; however, it is also tested by Nvidia GTX 970 GPU with Nvidia Driver 375.82 and Cuda-8.0 on Ubuntu 14 LTS environment. Chainer 1.21.0 version is used as the deep learning tool for this application. 

### Before running the code:

* please change the line 24 for "imagefolder" directory variable in main_Saliency.py file for the images that you want to test.

* the results will be saved to same path with the main_Saliency.py. So, if you want to save the results to specific existing directory, please chage the line 25 for "resultfolder" directory variable in main_Saliency.py file for the images that you want to test.

* Pretrained VGG16 chainer model can be obtained through the git [chainer-imagenet-vgg repository of Shunta Saito](https://github.com/mitmul/chainer-imagenet-vgg) under download_chainermodel.sh.
     which links the VGG16 model [here](https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0)
     or 
     ```
     $ wget https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0 -O VGG.model
     ```
* if you want to check saliency without center bias, just remove the "mask" variable multiplication in line 131      

## Run the code:

```
$ python main_Saliency.py
```


DISCLAIMER: The codes here are provided only for evaluation of the algorithm. Neither the authors of the codes, nor affiliations of the authors can be held responsible for any damages arising out of using this code in any manner. Please use the code at your own risk.
