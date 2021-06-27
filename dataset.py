import tensorflow as tf
import scipy.io as io
import numpy as np

#hyperparameter K; the number of spectral bands to consider

class dataset:
    def __init__(self,batch_size,config,training=True):

        train_file_path = config.train_file
        valid_file_path = config.valid_file
        with open(train_file_path,"r") as fp:
            self.train_files = list(fp.read().splitlines())

        with open(valid_file_path,"r") as fp:
            self.valid_file = list(fp.read().splitlines())
        self.training = training
        self.batch_size = batch_size

        self.K = config.K

    def _get_data(self,file):
        for img_path in range(len(file)):
                image = io.loadmat(img_path)
            
                bands = image.shape[-1]

                for i in range(0,bands-K):
                    spatial_image = image[:,:,i]
                    spectral_volume = image[:,:,i+self.K]
                    spectral_image = image[:,:,np.newaxis]

                    return spatial_image, spectral_volume

    def _aviris_generator(self):
        if self.training:
            spatial_image, spectral_volume = self._get_data(self.train_files)
        else:
            spatial_image, spectral_volume = self._get_data(self.valid_files)

        yield spatial_image, spectral_volume

    
    def _get_aviris(self):
        data = tf.data.Dataset.from_generator(self._aviris_generator)
        data = data.batch(self.batch_size)
        data = data.cache()
        data = data.prefetch(2)
        data = data.repeat()
        return data



