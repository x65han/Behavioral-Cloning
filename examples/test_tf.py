import tensorflow as tf

with tf.Session() as sess:
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

