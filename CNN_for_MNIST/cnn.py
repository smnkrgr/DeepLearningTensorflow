import tensorflow as tf, numpy as np
import numpy as np  
from mnist import MNIST
mndata = MNIST('mnist_data/')
trainx, trainy = mndata.load_training()
testx, testy = mndata.load_testing()

np.save('mnist.npy', (np.array(trainx) / 255.0, np.array(trainy),
                      np.array(testx) / 255.0, np.array(testy)))

# Test for GPU support
print tf.test.gpu_device_name()

# Numpy arrays for test and training data
trainx, trainy, testx, testy = np.load('mnist.npy')

# Reshape flat pictures into 28 x 28 pixels
trainx = np.reshape(trainx, [-1, 28, 28, 1])
testx = np.reshape(testx, [-1, 28, 28, 1])

# Print shapes of data
print 'train x shape is {}'.format(trainx.shape)
print 'train y shape is {}'.format(trainy.shape)
print 'test x shape is {}'.format(testx.shape)
print 'tesy y shape is {}'.format(testy.shape)

# Parameters for the model
batch_size = 128
training_iterations = 500000
model_evaluation_frequency = 250
dropout_prob = 0.5
initial_learning_rate = 0.001

# Returns a random batch of training data with size batch_size
def get_train_batch():
  indices = np.random.randint(low=0, high=trainx.shape[0], size=[batch_size])
  return trainx[indices], trainy[indices]

# Convolutional layer with batch normalization and max pooling
def conv_layer_with_max_pooling(input_tensor, filter_size):
  conv = tf.layers.conv2d(inputs=input_tensor, filters=filter_size,
                          kernel_size=[5, 5], padding="same",
                          activation=tf.nn.relu)
  normal = tf.contrib.layers.batch_norm(conv)
  pool = tf.layers.max_pooling2d(inputs=normal, pool_size=[2, 2], strides=2)
  return pool

# Tensorflow graph
g = tf.Graph()
with g.as_default():
  
  # Defining placeholders for features x and their labes y
  x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
  y = tf.placeholder(dtype=tf.int64, shape=[None,])
  
  # Defining a placeholder for the dropout probability
  p_dropout = tf.placeholder(dtype=tf.float32)
  
  # Defining a placeholder for adaptive learning rate
  lr = tf.placeholder(dtype=tf.float32)
  
  # First convolutional layer
  conv_1 = conv_layer_with_max_pooling(input_tensor=x, filter_size=32)
  
  # Second convolutional layer
  conv_2 = conv_layer_with_max_pooling(input_tensor=conv_1, filter_size=64)
  
  # Reshape tensor flat for fully connected layers
  conv_2_flat = tf.reshape(conv_2, [-1, 7 * 7 * 64])
  
  # Fully connected hidden layer with RELU activatin function and 1024 neurons
  fc_1 = tf.layers.dense(inputs=conv_2_flat, units=1024, activation=tf.nn.relu)
  
  # Dropout
  dropout = tf.nn.dropout(x=fc_1, keep_prob=p_dropout)
  
  # Logits for the 10 labes
  logits = tf.layers.dense(inputs=dropout, units=10)
  
  # Defining the loss function (cross entropy) and the optimizer
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  
  minimize_op = optimizer.minimize(loss, var_list=tf.trainable_variables())
  
  # Running a session to feed the graph
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # Calculations for the evaluation of the model
  missclassified = (tf.to_float(tf.count_nonzero(y - tf.argmax(tf.nn.softmax(logits), axis=1))) / tf.to_float(tf.shape(y)[0])) * 100
  confusion_matrix = tf.confusion_matrix(y, tf.argmax(tf.nn.softmax(logits), axis=1))
  average_training_loss = 0
  lowest_missclassification_rate = 100.0
  
  # Training iteration loop
  for i in range(training_iterations):
    
    # Feed data to the graph in batches of batch_size
    train_features, train_labels = get_train_batch()
    
    # Train the model and adapt learning rate in relation to number of iterations
    if lowest_missclassification_rate < 101.0:
      sess.run(minimize_op, feed_dict={x: train_features, y: train_labels, p_dropout: dropout_prob, lr: initial_learning_rate})
    if lowest_missclassification_rate < 0.6:
      sess.run(minimize_op, feed_dict={x: train_features, y: train_labels, p_dropout: dropout_prob, lr: initial_learning_rate/10})
    if lowest_missclassification_rate < 0.45:
      sess.run(minimize_op, feed_dict={x: train_features, y: train_labels, p_dropout: dropout_prob, lr: initial_learning_rate/50})
    if lowest_missclassification_rate < 0.41:
      sess.run(minimize_op, feed_dict={x: train_features, y: train_labels, p_dropout: dropout_prob, lr: initial_learning_rate/100})
      
    # Evaluate state of the model every model_evaluation_frequency iterations
    if i % model_evaluation_frequency == 0:
      print "Iteration number", i
      print "------------------------------------------------------------------------------------------"
      
      # Calculating loss and average loss over the training batch
      training_batch_loss = sess.run(loss, feed_dict={x: train_features, y: train_labels, p_dropout: 1.0})
      times_evaluated = i/model_evaluation_frequency + 1
      average_training_loss = (((times_evaluated - 1) * average_training_loss) + training_batch_loss) / times_evaluated
      print "Current training set loss:", training_batch_loss, "and average training set loss:", average_training_loss
      
      # Printing the confusion matrix for the test data
      #print "Confusion matrix:"
      #print sess.run(confusion_matrix, feed_dict={x: testx, y: testy})
        
      # Printing the test data loss and the missclassification rate
      #print "Test set loss:", sess.run(loss, feed_dict={x: testx, y: testy})
      missclassification_rate = sess.run(missclassified, feed_dict={x: testx, y: testy, p_dropout: 1.0})
      if lowest_missclassification_rate > missclassification_rate:
        lowest_missclassification_rate = missclassification_rate
      print "Missclassification rate:",missclassification_rate , "%, while lowest:", lowest_missclassification_rate, "%"
      print ""