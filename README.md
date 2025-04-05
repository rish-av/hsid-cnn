# hsid-cnn
Tensorflow 2.0 Implementation of HSID-CNN for denoising hyperspectral images.

- Dataset Preparation & Environment Preparation
  - Prepare two file `train.txt` & `valid.txt` with the paths to hyperspectral images (see data folder)
  - Add the paths of the files into a config file, `sample_config.yaml` is an example.
  - Install the requirements with `pip install -r requirements.txt`
 
- Training
   - Hyperparameters for training can be tweaked in the config file or you can use the default ones from `sample_config.yaml`.
   - Run `python train.py --config-file ./sample_config.yaml` for running the script
   - For using a pre-trained model: modify the `checkpoint_path` variable inside your config file
  
- Summaries
  - For having a look at the various error plots: run `tensorboard --logdir='./summaries'`
 
 
- Original Paper: [HSID-CNN](https://arxiv.org/pdf/1806.00183.pdf)


Archived: I do not maintain this repo anymore, you can still shoot me an email for any query but cannot guarantee as response.
