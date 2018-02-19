from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

stddev = 0.1
bias_weights_init = 0.050
dir_path = "C:\\Daten\\workspaces\\ml\\data"
mnist = input_data.read_data_sets(dir_path + '/data/', one_hot=True)

def neural_network(layer_nodes):
    x = tf.placeholder(dtype=tf.float32, shape=[None, features], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    W = weights(layer_nodes)
    b = biases(layer_nodes)
    return model(x, W, b)

def model(x, W, b):
    layer_variable = x
    for i in range(len(layer_nodes)-1):
        layer_variable = tf.add(tf.matmul(layer_variable, W[i]), b[i])
        layer_variable = tf.nn.relu(layer_variable)
    return layer_variable

def weights(layer_nodes):
    W = [0] * len(layer_nodes)
    for x in range(len(layer_nodes)-1):
        W[x] = tf.Variable(tf.truncated_normal([layer_nodes[x], 
                                                layer_nodes[x+1]], 
                                                stddev=stddev), name='W' + str(x+1))

    return W

def biases(layer_nodes):
    b = [0] * len(layer_nodes)
    for x in range(len(layer_nodes)-1):
         b[x] = tf.Variable(tf.constant(bias_weights_init, shape=[layer_nodes[x+1]]), name='b' + str(x+1))

    return b


if __name__ == "__main__":
    features = 784
    classes = 10
    layer_nodes = [features, 400, 200, classes]
    model = neural_network(layer_nodes)
    print(model)