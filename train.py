import tensorflow as tf
from model import Network
from tensorflow.python.keras import backend
from utils import _loss, _simulate_noise

with backend.get_graph().as_default():
    net = Network(20,20,20)


num_epochs = 1000
learning_rate = 0.0001
noise_level = 1.0

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0.),loss = tf.Variable(1e4),
optimizer=optimizer,model=net)

ckpt_manager = tf.train.CheckpointManager(checkpoint,directory="./checkpoints",max_to_keep=5)

@tf.function
def train_step(spatial_band,spectral_volume):
    spatial_band_noised = _simulate_noise(spatial_band,noise_level)
    spectral_volume_noised = _simulate_noise(spectral_volume,noise_level)
    with tf.GradientTape() as tape:
        out = net(spatial_band_noised,spectral_volume_noised)
        loss_val = _loss(spatial_band,out)
    grads = tape.gradient(loss_val,net.trainable_weights)
    optimizer.apply(zip(grads,net.trainable_weights))
    
    return loss_val, out


@tf.function
def valid_step(spatial_band,spectral_volume):
    spatial_band_noised = _simulate_noise(spatial_band,noise_level)
    spectral_volume_noised = _simulate_noise(spectral_volume,noise_level)
    out = net(spatial_band_noised,spectral_volume_noised)
    loss_val = _loss(spatial_band,out)

    return loss_val, out

init_epoch = checkpoint.epoch.numpy()+1 

for epoch in range(init_epoch,init_epoch+num_epochs):

    loss_val = tf.constant(0.)


