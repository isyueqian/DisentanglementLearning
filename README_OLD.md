# Spatial Decomposition Network (SDNet)

**Tensorflow implementation of SDNet**

For details refer to the paper:

> Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A. (2019). 
> *Factorised Representation Learning in Cardiac Image Analysis*. arXiv preprint arXiv:1903.09467.

If you find it useful, please also consider citing my work, as:
> Valvano, G., Chartsias, A., Leo, A., & Tsaftaris, S. A.,
> Temporal Consistency Objectives Regularize the Learning of Disentangled Representations, MICCAI Workshop on Domain Adaptation and Representation Transfer, 2019.

An implementation using Keras can be found at: https://github.com/agis85/anatomy_modality_decomposition


**Data:**

Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual
segmentations of the heart cavity, myocardium and right ventricle are provided.

Database at: [*acdc_challenge*](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).\
An atlas of the heart in each projection at: [*atlas*](http://tuttops.altervista.org/ecocardiografia_base.html).

# How to use it

**Overview:**
The code is organised with the following structure:

|    File/Directory            |Content                               
|---------------|----------------------------------------|
|architectures	| This directory contains the architectures used to build each module of the SDTNet (Encoders, Decoder, Segmentor, Discriminator, Transformer)|
|data			| folder with the data						|
|data_interfaces| dataset interfaces (dataset iterators)	|
|idas			| package with tensorflow utilities for deep learning (smaller version of [idas](https://github.com/gvalvano/idas)) |
results		| folder with the results (tensorboard logs, checkpoints, images, etc.)| 
|*config_file.py*| configuration file and network hyperparameters 
|*model.py*| file defining the general model and the training pipeline (architecture, dataset iterators, loss, train ops, etc.)
|*model_test.py*| simplified version of *model.py* , for test (lighter and faster)
|*split_data.py*| file for splitting the data in train/validation/test sets
|*prepare_dataset.py*| file for data pre-processing
|*train.py*| file for training



**Train:**
1. Put yourself in the project directory
2. Download the ACDC data set and put it under:  *./data/acdc_data/*
3. Split the data in train, validation and test set folders. You can either do this manually or you can run: ```python split_data.py```
4. Run ```python prepare_dataset.py``` to pre-process the data. The code will pre-process the data offline and you will be able to train the neural network without this additional CPU overload at training time (there are expensive operations such as interpolations). The pre-processing consists in the following operations:
    - data are rescaled to the same resolution
    - the slice index is placed on the first axis
    - slices are resized to the desired dimension (i.e. 128x128)
    - the segmentation masks will be one-hot encoded
5. Run ```python train.py``` to train the model.

Furthermore, you can monitor the training results using TensorBoard after running the following command in your bash:
```bash
tensorboard --logdir=results/graphs
```
**Test:**

The file *model.py* contains the definition of the SDTNet architecture but also that of the training pipeline, tensorboard logs, etc.. Compiling all this stuff may be quite slow for a quick test: for this reason, we share a lighter version of *model.py*, namely *model_test.py* that avoids defining variables not used at test time. You can use this file to run a test, coding something like:

```python
from model_test import Model
...
RUN_ID = 'SDTNet'
ckp_dir = project_root + 'results/checkpoints/' + RUN_ID

# Given: x = a batch of slices to test
model = Model()
model.define_model()
soft_anat, hard_anat, masks, reconstr = model.test(input_data=x, checkpoint_dir=ckp_dir)
```

Remember that, due to architectural constraints of the SDNet [see *Chartsias et al. (2019)*], the batch size that you used during training remains fixed at test time. 

# Results:

Example of anatomical factors extracted by the SDNet from the image on the left:

<img src="https://github.com/gvalvano/sdnet/blob/master/results/images/example.png" alt="example" width="600"/>

The last two images are: the reconstruction and the predicted segmentation mask, respectively. 

---------------------

For problems, bugs, etc. feel free to contact me at the following email address:

  *gabriele.valvano@imtlucca.it* 
  
Enjoy the code! :)

**Gabriele**
