# <center>Prompt Learning With Bounding Box Constraints for Medical Image Segmentation</center>

This is the repository for the paper **"[Prompt Learning With Bounding Box Constraints
for Medical Image Segmentation](https://arxiv.org/pdf/2507.02743)"** published at *IEEE Transactions on Biomedical Imaging* (2025).

## Overview
This work proposes a novel framework that combines the strengths
of foundation models with the cost-efficiency of weakly supervised learning. Our approach automates and adapts foundation
models by training a dedicated prompt generator module using only bounding box annotations. The proposed multi-loss optimization scheme integrates the segmentation predictions from the prompted foundation model with box-based spatial constraints and consistency regularization. 

## Framework

<p float="left">
  <img src="images/SAMPromptLearningWithBoxes.png" width="100%" />
</p>


## Usage
To use our code, follow these steps:

### 1) Set-up environment
Create an environment with the required packages using the following command:

```
$ conda create -n py310 python=3.10 pip
$ conda activate py310
$ conda install pytorch torchvision -c pytorch -c nvidia

$ conda install numpy matplotlib scikit-learn scikit-image h5py
$ pip install nibabel comet_ml flatten-dict pytorch_lightning
$ pip install transformers monai opencv-python
```

Then set-up the segment-anything library:
```
git subtree add --prefix segment-anything --squash https://github.com/facebookresearch/segment-anything.git main
cd segment-anything
pip install -e .
cd ..
```

Modify original `segment-anything/segment_anything/modeling/mask_decoder.py` file. <br/>
Replace (line 126)
```
src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
``` 
by
```
if image_embeddings.shape[0] != tokens.shape[0]:
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
else:
    src = image_embeddings
```

### 2) Get model checkpoints
Model versions of the SAM model are available with different backbone sizes. In our experiments, we use [SAM ViT-huge](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).



Create a directory `models_dir` and download the SAM model checkpoint</br>
```
$ mkdir <models-dir>/sam
$ wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O <models-dir>/sam/sam_vit_b_01ec64.pth
```


### 3) Prepare data

We used the following open-source datasets for our experiments:
- [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) ([link](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc) to train & test sets) 
- [CAMUS challenge](https://www.creatis.insa-lyon.fr/Challenge/camus/) ([link](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) to data)
- [HC Challenge](https://hc18.grand-challenge.org/) ([link](https://zenodo.org/records/1327317) to data)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) ([link](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar) to Spleen data, and [link](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar) to Liver data)
<br/> 
<br/> 

Create a directory `data_dir` containing all datasets, and with the following structure:

```
<data-dir>
├── ACDC 
│ └── raw                                           # Put the raw data here
│ └── preprocessed_sam                              # Subfolder created after preprocessing
|   └── train_2d_images 
|       └── patient001_frame01_slice2.nii.gz
|       └── ...
|   └── train_2d_masks
|       └── patient001_frame01_slice2.nii.gz
|       └── ...
|   └── val_2d_images     
|   └── val_2d_masks          
|   └── test_2d_images     
|   └── test_2d_masks               
|   └── image_embeddings                            # Pre-computed image embeddings
|       └── sam_vit_h_4b8939-pth                # Model checkpoint name 
|           └── train_2d_images
|               └── patient001_frame01_slice2.h5
|               └── ...
|           └── val_2d_images
|           └── test_2d_images
├── CAMUS_public
├── HC18
```

#### Preprocessing
Examples of the preprocessing applied are shown in the notebooks located in `JupyterNotebook` folder:
- `Preprocessing_ACDC.ipynb`
- `Preprocessing_CAMUS.ipynb`
- `Preprocessing_HC.ipynb`

These notebooks create a `preprocessed_sam` subfolder for every dataset and generate train/val/test sets that can be used by our dataloader.
<br/> 


In addition run the notebook `Save_SAM_image_embeddings.ipynb` for every dataset. </br>
This will create the subfolder `preprocessed_sam/image_embeddings`, with the pre-computed image embeddings from the specified model checkpoint.
</br>



### 4) Run code 

The code is located in the `src` folder. </br>
Modify the input arguments `--data_dir`, `--models_dir` and `--output_dir` in  `main.py`. Also, if using Comet-ml logger, fill in api-key and workspace name in `Configs/logger_config.yaml`

### Main training function

To train the model, run `src/main.py`:
```
$ python src/main.py  --data_dir <data-dir> --models_dir <models-dir> --output_dir <output-dir> --data_config data_config/ACDC_256.yaml  --data__class_to_segment 1 --train_config train_config/train_config_200_100_00001.yaml --loss_config loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_weak_W0001.yaml --seed 0
```

There are several input arguments, but most of them get filled in when using the bash script:
```
# These are the paths to the data and output folder
--data_dir          # path to directory where we saved our (raw and preprocessed) data
--models_dir        # path to directory where we saved backbone model checkpoints
--output_dir        # path to directory where we want to save our output model and experiments

# These are config file names located in src/Config
--data_config   
--model_config      
--train_config
--logger_config 
--loss_config       # can be several file paths, all located in Configs/loss_config

# Additional training configs (seed and gpu)
--seed
--gpu_idx           # gpu index, if we want to use a specific gpu

# Training hyper-parameters that we should change according to the dataset
# Note: arguments start with 'train__' if modifying train_config, 
# and with 'model__'  if modifying model_config, etc.
# Name is determined by hierarchical keywords to value in config, each separated by '__'
--data__class_to_segment           # index of class to segment in dataset
--data__compute_sam_embeddings     # whether to use compute embeddings during training (if not, will use precomputed embeddings). Default is False

--train__train_indices             # indices of labelled training data. Default is all.
--train__val_indices
--train__clip_gradient_norm_value  # value to clip gradient norm. Default is 1.
```

### Bash script
The bash script runs the `main.py` file that trains the prompt generator module with our proposed losses, across different seeds and training sets.

```
$ bash src/Bash_scripts/run_experiments.sh <data-dir> <models-dir> <output-dir> <path-to-data-config> <class-id> <num-samples> <path-to-train-config> <gpu-idx>
```

For example: 
```
bash src/Bash_scripts/run_experiments.sh <data-dir> <models-dir> <output-dir> data_config/ACDC_256.yaml 1 20 train_config/train_config_200_100_00001.yaml 0
```
(and modify `<path-to-data-config>` and `<class-id>` to run our main experiments with different tasks).</br>


Options:
- `<data-dir>`: path to directory containing all raw and preprocessed datasets.
- `<models-dir>`: path to directory where model checkpoints are saved.
- `<output-dir>`: path to directory where output run will be saved.
- `<path-to-data-config>`: file located in `Configs/data_config`. Path from `Configs` folder. </br>
For example: data_config/ACDC_256.yaml, data_config/CAMUS_512.yaml or data_config/HC_640.yaml
- `<class-id>`: starting from 1. We used 1 for all datasets except for ACDC where we experiment with classes 1 and 3.
- `<num-samples>`: 0 (all samples), 10 or 20.
- `<path-to-train-config>`: file located in `Configs/train_config`. Path from `Configs` folder. For example, train_config/train_config_200_100_00001.yaml (for 10 or 20 samples), or train_config/train_config_20_10_00001.yaml (for all samples)
- `<gpu-idx>`: depending on server
<br/>


