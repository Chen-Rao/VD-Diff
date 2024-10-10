# VD-Diff
Official implementation of Paper "Rethinking Video Deblurring with Wavelet-Aware Dynamic Transformer and Diffusion Model" (ECCV 2024)

# Requirements
You can create a new environment named VD-Diff by running
```
conda create -n VD-Diff python==3.10
conda activate VD-Diff
pip install -r requirements.txt
```

# Datesets
First, download [GoPro dataset](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view) and put it in "dataset" folder 
as the form below: <br>
*   dataset<br>
    *  GoPro
        *   test
            *   blur
            *   sharp
        *   train
            *   blur
            *   sharp

# Train
## Training Stage One
To train your own S1 model on the GoPro dataset, simply use the following command:
```
python basicsr/train.py -opt options/train/train_GoPro_S1.yml
```
Your S1 model weights will be saved in the directory: "experiments/GOPRO_S1/models/".
## Training Stage Two
In this stage, update the "pretrain_network_S1" entry in "options/train/train_GoPro_S2.yml" with the path to your S1 model weights. Then, execute the following command:
```
python basicsr/train.py -opt options/train/train_GoPro_S2.yml
```
## Training Stage Three
For Stage Three, modify the "pretrain_network_S2" entry in "options/train/train_GoPro_S3.yml" to point to your S2 model weights. Next, run:
```
python basicsr/train.py -opt options/train/train_GoPro_S3.yml
```

# Test
First, modify the "pretrain_network_g" entry in "options/test/test_GoPro_S3.yml" to your S3 model weights. Then run:
```
python basicsr/test.py -opt options/test/test_GoPro_S3.yml
```
You can also test your S1 and S2 model by changing the ".yml" option file path accordingly.

# Pretrained Models
Pretrained models will be available in a few days.