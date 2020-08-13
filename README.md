# hsid-cnn
Tensorflow 2.0 Implementation of HSID-CNN for denoising hyperspectral images.

- Dataset Preparation
 - Prepare two file `train.txt` & `valid.txt` with the paths to hyperspctral image.
 - Install the requirements with `pip install -r requirements.txt`
 
- Training
  - Hyperparameters for training can be tweaked inside of `train.py` file.
  - Just run `python train.py` for running the script
  - For using pre-trained model; you can modify the `checkpoint_path` variable inside `train.py`
  - **Argparse would be introduced soon for better running**
  
- Summaries
 - For having a look at the various plots; run `tensorboard --logdir='./summaries'`
 
- Pretrained weights would be made available soon.
 
 
- Original Paper: [HSID-CNN](https://arxiv.org/pdf/1806.00183.pdf)
