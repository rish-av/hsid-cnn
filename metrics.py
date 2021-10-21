import tensorflow as tf

def _psnr(x,y_out):
    max_val = tf.math.reduce_max(x)
    psnr_val = tf.image.psnr(x,y_out,max_val=max_val)
    return psnr_val

def _ssim(x,y_out):
    max_val = tf.math.reduce_max(x)
    ssim_val = tf.image.ssim(x,y_out,max_val=max_val)
    return ssim_val