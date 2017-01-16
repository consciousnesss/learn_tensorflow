import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def mnist_simple():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([28*28, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y)
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    for i in range(1000):
        batch = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch[0], labels: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(session.run(accuracy, feed_dict={x: mnist.test.images, labels: mnist.test.labels}))


def mnist_conv():

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    W_conv1 = weight_variable(shape=(5, 5, 1, 32))
    b_conv1 = bias_variable(shape=(32,))

    x_image = tf.reshape(x, (-1, 28, 28, 1))

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable(shape=(5, 5, 32, 64))
    b_conv2 = bias_variable(shape=(64,))

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable(shape=(7*7*64, 1024))
    b_fc1 = bias_variable(shape=(1024,))

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_probabilities = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_probabilities)

    W_fc2 = weight_variable(shape=(1024, 10))
    b_fc2 = bias_variable(shape=(10,))

    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv)
    )

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(labels, axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    for i in range(20000):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = session.run(
                accuracy,
                feed_dict={x:batch[0], labels: batch[1], keep_probabilities: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        session.run(train_step, feed_dict={x: batch[0], labels: batch[1], keep_probabilities: 0.5})

    print("test accuracy %g" % session.run(accuracy, feed_dict={
        x: mnist.test.images, labels: mnist.test.labels, keep_probabilities: 1.0}))


if __name__ == '__main__':
    import time
    st = time.time()
    mnist_conv()

    print(time.time() - st)
    '''
    CPU:
    test accuracy 0.9926
    1818.77772021

    GPU:
    
    '''
