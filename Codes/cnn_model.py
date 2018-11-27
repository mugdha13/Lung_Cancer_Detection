import tensorflow as tf
import util as tu




def cnn(x):
    
    """ CNN model to detect lung cancer
    
            Args:
                x: tensor of shape [batch_size, width, height, channels]
        
            Returns:
                pool2: tensor with all convolutions, pooling applied
    """
    
    with tf.name_scope('cnn') as scope:
        with tf.name_scope('conv1') as inner_scope:
            wcnn1 = tu.weight([3, 3, 1, 64], name='wcnn1')
            bcnn1 = tu.bias(1.0, [64], name='bcnn1')
            conv1 = tf.add(tu.conv2d(x, wcnn1, stride=(1, 1), padding='SAME'), bcnn1)
            conv1 = tu.relu(conv1)
            # (?, 192, 192, 64)
            
        with tf.name_scope('conv2') as inner_scope:
            wcnn2 = tu.weight([3, 3, 64, 64], name='wcnn2')
            bcnn2 = tu.bias(1.0, [64], name='bcnn2')
            conv2 = tf.add(tu.conv2d(conv1, wcnn2, stride=(1, 1), padding='SAME'), bcnn2)
            conv2 = tu.relu(conv2)
            #(?, 192, 192, 64)
            
        with tf.name_scope('max_pool') as inner_scope:
            pool1 = tu.max_pool2d(conv2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME') 
            # (?, 96, 96, 64)
            
        with tf.name_scope('conv3') as inner_scope:
            wcnn3 = tu.weight([3, 3, 64, 64], name='wcnn3')
            bcnn3 = tu.bias(1.0, [64], name='bcnn3')
            conv3 = tf.add(tu.conv2d(pool1, wcnn3, stride=(1, 1), padding='SAME'), bcnn3)
            conv3 = tu.relu(conv3)
            # (?, 96, 96, 64)
            
        with tf.name_scope('conv4') as inner_scope:
            wcnn4 = tu.weight([3, 3, 64, 64], name='wcnn4')
            bcnn4 = tu.bias(1.0, [64], name='bcnn4')
            conv4 = tf.add(tu.conv2d(conv3, wcnn4, stride=(1, 1), padding='SAME'), bcnn4)
            conv4 = tu.relu(conv4)
            # (?, 96, 96, 64)
            
        with tf.name_scope('conv5') as inner_scope:
            wcnn5 = tu.weight([3, 3, 64, 64], name='wcnn5')
            bcnn5 = tu.bias(1.0, [64], name='bcnn5')
            conv5 = tf.add(tu.conv2d(conv4, wcnn5, stride=(1, 1), padding='SAME'), bcnn5)
            conv5 = tu.relu(conv5)
            # (?, 96, 96, 64)
            
        with tf.name_scope('max_pool') as inner_scope:
            pool2 = tu.max_pool2d(conv5, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME') 
            # (?, 48, 48, 64)
            
        return pool2




def classifier(x, dropout):
    
    """cnn fully connected layers definition
    
            Args:
                x: tensor of shape [batch_size, width, height, channels]
                dropout: probability of non dropping out units
                
            Returns:
                fc3: 2 linear tensor taken just before applying the softmax operation
                    it is needed to feed it to tf.softmax_cross_entropy_with_logits()
                softmax: 2 linear tensor representing the output probabilities of the image to classify
    """
    
    pool2 = cnn(x)
    
    dim = pool2.get_shape().as_list()
    flat_dim = dim[1] * dim[2] * dim[3] # 48 * 48 * 64
    flat = tf.reshape(pool2, [-1, flat_dim])
    
    with tf.name_scope('classifier') as scope:
        with tf.name_scope('fullyconected1') as inner_scope:
            wfc1 = tu.weight([flat_dim, 500], name='wfc1')
            bfc1 = tu.bias(1.0, [500], name='bfc1')
            fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
            fc1 = tu.relu(fc1)
            fc1 = tf.nn.dropout(fc1, dropout)
            
        with tf.name_scope('fullyconected2') as inner_scope:
            wfc2 = tu.weight([500, 100], name='wfc2')
            bfc2 = tu.bias(1.0, [100], name='bfc2')
            fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
            fc2 = tu.relu(fc2)
            fc2 = tf.nn.dropout(fc2, dropout)
            
        with tf.name_scope('classifier_output') as inner_scope:
            wfc3 = tu.weight([100, 2], name='wfc3')
            bfc3 = tu.bias(1.0, [2], name='bfc3')
            fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)
            softmax = tf.nn.softmax(fc3)
            
    return fc3, softmax
