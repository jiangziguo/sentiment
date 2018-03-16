#Train in Python
import tensorflow as tf
# good idea
# https://stackoverflow.com/documentation/tensorflow/10718/save-tensorflow-model-in-python-and-load-with-java#t=201709030336395954421
tf.reset_default_graph()

# DO MODEL STUFF
# Pretrained weighting of 2.0
W = tf.get_variable('w', initializer=tf.constant(2.0), dtype=tf.float32)
# Model input x
x = tf.placeholder(tf.float32, name='x')
# Model output y = W*x
y = tf.multiply(W, x, name='y')

# DO SESSION STUFF
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# SAVE THE MODEL
builder = tf.saved_model.builder.SavedModelBuilder("E:\\tensorflowtest\\model" )
builder.add_meta_graph_and_variables(
  sess,
  [tf.saved_model.tag_constants.SERVING]
)
builder.save()