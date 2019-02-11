import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pandas import DataFrame
from pandas import Series,DataFrame
from sklearn import preprocessing

data=pd.read_excel("C:\\Users\\siat\\desktop\\dwn\\siatpaper\\8variables.xlsx")
dwn = np.array(data)
print(type(dwn))
dwnn = preprocessing.scale(dwn)
#XXX = tf.convert_to_tensor(XX,dtype=tf.float32)
#XXXX = tf.reshape(XX,shape=[-1])

# tf.Session() as sess:
    # (sess.run(XXX))
    
learning_rate = 0.001
training_epochs = 1  
batch_size = 600  

n_input = 10

X = tf.placeholder(tf.float32, [None, n_input])  

n_hidden_1 = 500 # 第一编码层,neural num  
n_hidden_2 = 100 # 第二编码层,neural num 
n_hidden_3 = 1 # 第三编码层,neural num 

weights = {  
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),  
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),  
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),  
}  
biases = {  
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),  
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),  
}  

def encoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                   biases['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                   biases['encoder_b2']))  
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3

def decoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3



#model construction
encoder_op = encoder(X)  
decoder_op = decoder(encoder_op)

#prediction
y_pred = decoder_op
y_true = X

#cost function and optimizer
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#define next_batch
def batch_iter(sourceData, batch_size, num_epochs, shuffle=True):
    data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]

#run



with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(10):
        for batch_num in range(6):
        
            batch_xs=batch_iter(dwnn, 600, 10, shuffle=True)
            batch_xss=next(batch_xs)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xss})
            
    


    #total_batch=int(3054/batch_size)
    #for epoch in range(training_epochs):
        #for i in range(total_batch):
            #batch_xs=tf.train.batch(tensors=XXXXX,batch_size=256,enqueue_many=True)
        
            #_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        #if epoch % display_step == 0:  
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))  
    #print("Optimization Finished!")
        
    encode_decode=sess.run(y_pred,feed_dict={X: dwnn})

    outt=sess.run(encoder_op,feed_dict={X:dwnn})
    print(outt)
    print(sess.run(cost,feed_dict={X:dwnn}))
    print(type(outt))

    df=pd.DataFrame(outt)
    df.to_csv('226-1.8variables.csv')
    


