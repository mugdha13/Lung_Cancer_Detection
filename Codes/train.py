import sys
import os.path
import cnn_model as model
import tensorflow as tf
import util as tu
import numpy as np




def train(
        epochs, 
        batch_size, 
        learning_rate, 
        dropout,  
        image_path,
        label_path,
        ):
    
    """ Procedure to train the model for detecting lung cancer
        
            Args:
                image_path: path where CT image folder locates 
                label_path: path where label file locates      
    """
    
    img = tu.read_image(image_path)
    label = tu.read_label(label_path)
    x, y, x_test, y_test = tu.divide_trainset_testset(img,label)
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    num_batches = (int)(x.shape[0]/batch_size)
    num_test_batches = (int)(x_test.shape[0]/batch_size)
    
    x_width = x.shape[1]
    x_height = x.shape[2]
    x_channel = x.shape[3]
    
    x_b = tf.placeholder(tf.float32, shape = [batch_size,x_width,x_height,x_channel])
    y_b = tf.placeholder(tf.float32, shape = [batch_size,2])  #2 is number of result

    pred_b, _ = model.classifier(x_b, keep_prob)
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_b, labels=y_b, name='cross-entropy'))
        
    with tf.name_scope('loss'):
        loss = cross_entropy
                
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(pred_b, 1), tf.argmax(y_b, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
    with tf.name_scope('analysis'):
        y_pred = tf.argmax(pred_b, 1)
        y_real = tf.argmax(y_b, 1)
        tp = tf.count_nonzero(y_pred * y_real, dtype=tf.float32)
        tn = tf.count_nonzero((y_pred - 1) * (y_real - 1), dtype=tf.float32)
        fp = tf.count_nonzero(y_pred * (y_real - 1), dtype=tf.float32)
        fn = tf.count_nonzero((y_pred - 1) * y_real, dtype=tf.float32)
        
    with tf.name_scope('optimizer'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
        
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for e in range(epochs):
            for i in range(num_batches-1):
                step = i*batch_size
                x_batch = x[step:step+batch_size] 
                y_batch = y[step:step+batch_size] 
                
                sess.run(optimizer,
                         feed_dict={x_b: x_batch, y_b: y_batch, lr: learning_rate, keep_prob: dropout})
            
            cost_sum = 0
            avg_acc = 0
            for j in range(num_test_batches-1):
                step = j*batch_size
                x_batch = x_test[step:step+batch_size] 
                y_batch = y_test[step:step+batch_size] 
                c, a = sess.run([loss, accuracy], feed_dict={x_b: x_batch, y_b: y_batch, lr: learning_rate, keep_prob: 1.0})
                cost_sum += c
                avg_acc += a/(num_test_batches-1)
                
            print ('Epoch: {:03d} \nLoss: {:.7f}\nAccuracy: {:.4f}\n'.format(e+1, cost_sum, avg_acc))
        
        TP = 0 
        TN = 0
        FP = 0
        FN = 0
        cost_sum = 0
        avg_acc = 0
        for j in range(num_test_batches-1):
            step = j*batch_size
            x_batch = x_test[step:step+batch_size] 
            y_batch = y_test[step:step+batch_size] 
            tp_temp,tn_temp,fp_temp,fn_temp,c,a = sess.run([tp,tn,fp,fn,loss,accuracy],
                                                           feed_dict={x_b: x_batch, y_b: y_batch, lr: learning_rate, keep_prob: 1.0})
            TP += tp_temp
            TN += tn_temp
            FP += fp_temp
            FN += fn_temp
            cost_sum += c
            avg_acc += a/(num_test_batches-1)
        
        precision = TP / (TP + FP)
        recall = TP / (TP+ FN)
        f1 = 2 * precision * recall / (precision + recall)
        print ("\n\nLoss: {:.7f}\nAccuracy: {:.4f}\n".format(cost_sum, avg_acc))
        print("TP: ",TP)
        print("TN: ",TN)
        print("FP: ",FP)
        print("FN: ",FN)
        print("\nPrecision: ",precision)
        print("Recall: ",recall)
        print("f1: ", f1)




if __name__ == '__main__':
    
    EPOCHS = 10
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-04
    DROPOUT = 0.6
    IMAGE_PATH = 'G:/NN Project/subset0/subset0'
    LABEL_PATH = 'G:/NN Project/label.xlsx'
    train(
        EPOCHS, 
        BATCH_SIZE, 
        LEARNING_RATE, 
        DROPOUT, 
        IMAGE_PATH, 
        LABEL_PATH
    )
