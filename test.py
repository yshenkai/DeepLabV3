import numpy as np
from PIL import Image
import tensorflow as tf
y_pred=tf.ones(shape=(8,220,220,21))
y_true=tf.zeros(shape=(8,220,220))


print(y_pred[:,:,:,:-1].shape)