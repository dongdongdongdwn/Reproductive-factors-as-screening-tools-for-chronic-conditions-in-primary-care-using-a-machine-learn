import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xlrd
from xlutils.copy import copy

data = pd.read_excel("C:\\Users\\Researcher\\Desktop\\SIAT paper\\paper1\\10variables.xlsx")
dwn = np.array(data)
dwnn = preprocessing.scale(dwn)



def write_excel_xls_append(path, value):
    index = len(value)  # get the No of lines of the data to be write
    workbook = xlrd.open_workbook(path)  # open the file
    sheets = workbook.sheet_names()  # get all the sheets
    worksheet = workbook.sheet_by_name(sheets[0])  # get the first sheet
    rows_old = worksheet.nrows  # get the lines No already in the sheet
    new_workbook = copy(workbook)  # transfer xlrd to xlwt
    new_worksheet = new_workbook.get_sheet(0)  # get the first sheet of transfered sheet
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # write from No i+rows_old line
    new_workbook.save(path)
    print("write to excel done！")


n_input = 10
n_hidden_3 = 1

X = tf.placeholder(tf.float32, [None, n_input])

for learning_rate in (0.001, 0.005,0.01, 0.05, 0.1, 0.5):
    for batch_size_n in (100,300, 600):
        for n_hidden_1 in (9, 8, 7, 6, 5, 4, 3):
            for n_hidden_2 in (8, 7, 6, 5, 4, 3, 2):
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


                # model construction
                encoder_op = encoder(X)
                decoder_op = decoder(encoder_op)

                # prediction
                y_pred = decoder_op
                y_true = X

                # cost function and optimizer
                cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
                # could try different Optimizer here, i.e.
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

                
                # define batch
                def batch_iter(sourcedata, batch_size, num_epochs, shuffle=True):
                    data = np.array(sourcedata)  # save sourceData as array
                    data_size = len(sourcedata)
                    num_batches_per_epoch = int(len(sourcedata) / batch_size) + 1
                    for epoch in range(num_epochs):
                        # Shuffle the data at each epoch
                        if shuffle:
                            shuffle_indices = np.random.permutation(np.arange(data_size))
                            shuffled_data = sourcedata[shuffle_indices]
                        else:
                            shuffled_data = sourcedata

                        for batch_num in range(num_batches_per_epoch):
                            start_index = batch_num * batch_size
                            end_index = min((batch_num + 1) * batch_size, data_size)

                            yield shuffled_data[start_index:end_index]


                # run

                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    for epoch in range(10):
                        for batch_num in range(6):
                            batch_xs = batch_iter(dwnn, batch_size_n, 30, shuffle=True)
                            batch_xss = next(batch_xs)
                            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xss})


                    encode_decode = sess.run(y_pred, feed_dict={X: dwnn})

                    outt = sess.run(encoder_op, feed_dict={X: dwnn})
                    cost_final = sess.run(cost, feed_dict={X: dwnn})
                    print(cost_final)
                    cost_search = np.array([[learning_rate,batch_size_n,n_hidden_1,n_hidden_2,cost_final],])
                    excel_name = 'Hyper_para_search_weights_RFs.xls'
                    write_excel_xls_append(excel_name, cost_search)

                    #df = pd.DataFrame(outt)
                    #df.to_csv('226-1.8variables.csv')



    
    
    
