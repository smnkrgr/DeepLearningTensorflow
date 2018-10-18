import tensorflow as tf, numpy as np
import numpy as np  
from mnist import MNIST
mndata = MNIST('mnist_data/')
trainx, trainy = mndata.load_training()
testx, testy = mndata.load_testing()

np.save('mnist.npy', (np.array(trainx) / 255.0, np.array(trainy), np.array(testx) / 255.0, np.array(testy)))

# 1
trainx, trainy, testx, testy = np.load('mnist.npy')
print 'train x shape is {}'.format(trainx.shape)
print 'train y shape is {}'.format(trainy.shape)
print 'test x shape is {}'.format(testx.shape)
print 'tesy y shape is {}'.format(testy.shape)

# Additional 1
learning_rate = 0.1
minibatch_size = 64
layer_size_h1 = 400
layer_size_h2 = 400
standard_deviation_of_initialization = 0.1
number_of_training_iterations = 100000
model_evaluation_frequency = 100
dropout_value = 0.5

# 2
def get_train_batch():
  indices = np.random.randint(low=0, high=60000, size=[minibatch_size])
  return trainx[indices], trainy[indices]


g = tf.Graph()
with g.as_default():
  
  # 3
  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.int64, [None])
  
  p_dropout = tf.placeholder(tf.float32)
  
  # 4
  def fc(tensor, outdim, name):
    w = tf.get_variable(name=name + 'w', shape=(tensor.get_shape().as_list()[1], outdim), initializer=tf.truncated_normal_initializer(stddev=standard_deviation_of_initialization))
    b = tf.get_variable(name=name + 'b', shape=(outdim), initializer=tf.constant_initializer(0.0))
    
    out = tf.matmul(tensor, w)
    out = out + b
    
    return out
  
  # 5 
  h1 = tf.nn.relu(fc(x, layer_size_h1, "h1"))
  h2 = tf.nn.relu(fc(h1, layer_size_h2, "h2"))
  
  # Additional 8 different activation function (slow and bad)
  # h1 = tf.nn.sigmoid(fc(x, layer_size_h1, "h1"))
  # h2 = tf.nn.sigmoid(fc(h1, layer_size_h2, "h2"))
  
  # Additional 6 dropout
  drop_out = tf.nn.dropout(h2, p_dropout)
  
  # 6
  z = fc(drop_out, 10, "z")
  
  # 13
  missclassified = (tf.to_float(tf.count_nonzero(y - tf.argmax(tf.nn.softmax(z), axis=1))) / tf.to_float(tf.shape(y)[0])) * 100
  
  # Additional 5 confusion matrix (not done)
  confusion_matrix = tf.confusion_matrix(y, tf.argmax(tf.nn.softmax(z), axis=1))
  
  # 7
  p = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y))
  
  # 8
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  minimize_op = optimizer.minimize(p, var_list=tf.trainable_variables())
  
  # 9
  sess = tf.Session()
  
  # 10
  sess.run(tf.global_variables_initializer())
  
  average_training_loss = 0
  # 11
  for i in range(number_of_training_iterations):
    batch = get_train_batch()
    sess.run(minimize_op, feed_dict={x: batch[0], y: batch[1], p_dropout: dropout_value})
    
    if (i % model_evaluation_frequency == 0):
      print "Iteration number", i
      print "------------------------------------------------------------------------------------------"
      # Additional 3 average training loss
      training_batch_loss = sess.run(p, feed_dict={x: batch[0], y: batch[1], p_dropout: 1.0})
      times_evaluated = i/model_evaluation_frequency + 1
      average_training_loss = (((times_evaluated - 1) * average_training_loss) + training_batch_loss) / times_evaluated 
      # 12
      print "Current training set loss:", training_batch_loss, "and average training set loss:", average_training_loss
      print "Confusion matrix:"
      print sess.run(confusion_matrix, feed_dict={x: testx, y: testy, p_dropout: 1.0})
        
      # 13
      print "Test set loss:", sess.run(p, feed_dict={x: testx, y: testy, p_dropout: 1.0})
      print "Missclassification rate:", sess.run(missclassified, feed_dict={x: testx, y: testy, p_dropout: 1.0}), "%"
      print ""