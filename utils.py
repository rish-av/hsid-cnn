import tensorflow as tf

def _simulate_noise(volume,noise_level):
    shape = tf.shape(volume)
    noise = tf.random.normal(shape,mean=0.0,stddev=noise_level)
    noise_simulated_data = volume + noise
    return noise_simulated_data

def _loss(x,y_out):
    loss = tf.norm((y_out-x))
    return loss

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self