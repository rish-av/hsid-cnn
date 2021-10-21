import tensorflow as tf

#gathering the spectral information
def spectral_conv(spectral_volume,num_filters):
    def conv3d(filters,kernel_size,input_shape):
        return tf.keras.layers.Conv3D(filters,kernel_size,
        input_shape= input_shape,padding='same')

    spectral_volume_e = tf.expand_dims(spectral_volume,axis=0)
    input_shape = spectral_volume_e.shape[1:]

    conv1 = tf.squeeze(conv3d(num_filters,3,input_shape)(spectral_volume_e),axis=0)
    conv2 = tf.squeeze(conv3d(num_filters,5,input_shape)(spectral_volume_e),axis=0)
    conv3 = tf.squeeze(conv3d(num_filters,7,input_shape)(spectral_volume_e),axis=0)

    output_volume = tf.concat([conv1,conv2,conv3],axis=-1)

    return tf.keras.Model(inputs=spectral_volume,outputs=output_volume,name='spectral_conv')

#gathering the spatial information
def spatial_conv(spatial_band,num_filters):
    def conv2d(filters,kernel_size,stride=1,padding='same'):
        return tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
    
    spatial_band_e = tf.expand_dims(spatial_band,axis=-1)
    conv1 = conv2d(num_filters,3)(spatial_band_e)
    conv2 = conv2d(num_filters,5)(spatial_band_e)
    conv3 = conv2d(num_filters,7)(spatial_band_e)
    
    output_volume = tf.concat([conv1,conv2,conv3],axis=-1)

    return tf.keras.Model(inputs=spatial_band,outputs=output_volume,name='spatial_conv')

def convolution_block(volume,num_filters):
    def conv2d(filters,kernel_size,stride=1,padding='same'):
        return tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
    
    conv1 = conv2d(num_filters,3)(volume)
    conv2 = conv2d(num_filters,3)(conv1)
    conv3 = conv2d(num_filters,3)(conv2)
    conv4 = conv2d(num_filters,3)(conv3)
    conv5 = conv2d(num_filters,3)(conv4)
    conv6 = conv2d(num_filters,3)(conv5)
    conv7 = conv2d(num_filters,3)(conv6)
    conv8 = conv2d(num_filters,3)(conv7)
    conv9 = conv2d(num_filters,3)(conv8)

    final_volume = tf.concat([conv3,conv5,conv5,conv9],axis=-1)
    clean_band = conv2d(1,3)(final_volume)

    return tf.keras.Model(inputs=volume,outputs=clean_band,name='convolution_block')

class Network(tf.keras.Model):
    def __init__(self,num_3d_filters,num_2d_filters,num_conv_filters,K=24):
        super(Network,self).__init__()
        self.spectral_conv = spectral_conv(tf.keras.Input(shape=(None,None,None,K)),num_3d_filters)
        self.spatial_conv = spatial_conv(tf.keras.Input(shape=(None,None,1)),num_2d_filters)
        self.convolution_block = convolution_block(tf.keras.Input(shape=(None,None,num_3d_filters*3 + num_2d_filters*3)),num_conv_filters)
    
    def call(self,spatial_band,spectral_volume):
        spatial_vol = self.spatial_conv(spatial_band)
        spectral_vol = self.spectral_conv(spectral_volume)

        for_conv_block = tf.concat([spatial_vol[:,:,:,0,:],spectral_vol],axis=-1)
        residue = self.convolution_block(for_conv_block)

        return residue + spatial_band