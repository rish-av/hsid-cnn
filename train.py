import tensorflow as tf
from model import Network
from tensorflow.python.keras import backend
from utils import _loss, _simulate_noise
from dataset import dataset


with backend.get_graph().as_default():
    net = Network(20,20,20)


num_epochs = 1000
learning_rate = 0.0001
noise_level = 1.0

valid_data = dataset(2,False)
train_data = dataset(2,True)

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

    loss_train = tf.constant(0.)
    loss_val = tf.constant(0.)
    
    sample_count=1
    for i,(spatial_image, spectral_volume) in enumerate(train_data._get_aviris()):

        output, loss = train_step(spatial_image,spectral_volume)
        loss_train += loss
        sample_count+=1

        print("Train loss for the sample %d is %.2f"%(i+1,loss))

    print("Average train loss for epoch %d is %.2f"%(epoch,loss_train/sample_count))

    sample_count=1
    for i,(spatial_image, spectral_volume) in enumerate(valid_data._get_aviris()):

        output, loss = valid_step(spatial_image, spectral_volume)
        loss_val += loss
        sample_count+=1

    print("Average valid loss for epoch %d is %.2f"%(epoch,loss_val/sample_count))



