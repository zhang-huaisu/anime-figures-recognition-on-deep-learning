# -*- coding: utf-8 -*-
"""
Created on Sat May  5 13:20:08 2018

@author: 怀素
"""
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt   
import tensorflow as tf  
pixel =2304
images_amount = 4800
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

homepage="G:/graduation/datasets/"
#homepage ="E:/zhanghuaisu/datasets/"
address_array =['testing/angry/','testing/happy/','testing/surprise/', \
                'training/angry/','training/happy/','training/surprise/',]
address=[homepage+ x for x in address_array]

def images2array():
    labels=[]    
    images=[]
    for i in range(len(address)):
        if i<3:
            dataset_amount=300
        else: dataset_amount=1300
        for j in range(dataset_amount):
            image_ =Image.open(address[i] +str(j) +'.jpg')            
            r, g, b =image_.split()  #split the channels of per image           
            r_arr = np.array(r).reshape(pixel)
            g_arr = np.array(g).reshape(pixel)
            b_arr = np.array(b).reshape(pixel)
            image_arr = np.concatenate((r_arr, g_arr, b_arr))            
            images.append(image_arr)            
            if i==0 or i==3 or i==6 or i==9:
                labels.append('0')
            elif i==1 or i== 4 or i==7 or 10:
                labels.append('3')
            elif i==2 or i==5 or i==8 or 11:
                labels.append('5')
            
    return images, labels

images,labels =images2array()
emotion_data = np.zeros((len(images), 3*pixel)) 
emotion_label = np.zeros((len(labels), 7), dtype=int) 

for i in range(len(images)):
    x = images[i]          
    x_max = x.max()  
    x = x / (x_max + 0.0001)  
    emotion_data[i] = x
    emotion_label[i, int(labels[i])] = 1  
    if i <10:  
        print('i: %d \t '%(i), emotion_label[i])

train_num = 1300  
test_num = 300  
train_x = emotion_data[0:train_num, :]  
train_y = emotion_label[0:train_num, :] 
print(len(train_y))
test_x = emotion_data[train_num: train_num + test_num, :]  
test_y = emotion_label[train_num: train_num + test_num, :]
print("All is well")  
batch_size = 50
train_batch_num = train_num / batch_size  
test_batch_num = test_num / batch_size  
train_epoch = 5
learning_rate = 0.001  
# Network Parameters  
n_input = 3*pixel  # data input (img shape: 48*48)  
n_classes = 7  # total classes
dropout = 0.5  # Dropout, probability to keep units  
# tf Graph input  
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])  
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)  
  
  
# Create some wrappers for simplicity  
def conv2d(x, W, b, strides=1):  
    # Conv2D wrapper, with bias and relu activation  
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')  
    x = tf.nn.bias_add(x, b)      
    return tf.nn.relu(x)  
  
  
def maxpool2d(x, k=2):  
    # MaxPool2D wrapper  
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],  
                          padding='VALID')  
  
  
# Create model  
def conv_net(x, weights, biases, dropout):  
    # Reshape input picture  
    x = tf.reshape(x, shape=[-1, 48, 48, 3])
    print(x)
    # Convolution Layer  
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) 
    print(conv1)
    # Max Pooling (down-sampling)  
    conv1 = maxpool2d(conv1, k=2)  
    # Convolution Layer  
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  
    # Max Pooling (down-sampling)  
    conv2 = maxpool2d(conv2, k=2)  
    # Convolution Layer  
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])  
    # Max Pooling (down-sampling)  
    conv3 = maxpool2d(conv3, k=2)  
    # Fully connected layer  
    # Reshape conv2 output to fit fully connected layer input  
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])  
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  
    fc1 = tf.nn.relu(fc1)  
    # Apply Dropout  
    fc1 = tf.nn.dropout(fc1, dropout)  
    # Output, class prediction  
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])  
    return out  
  
  
# Store layers weight & bias  
weights = {  
    # 3x3 conv, 1 input, 128 outputs  
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 128])),  
    # 3x3 conv, 128 inputs, 64 outputs  
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 64])),  
    # 3x3 conv, 64 inputs, 32 outputs  
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),  
    # fully connected,  
    'wd1': tf.Variable(tf.random_normal([6 * 6 * 32, 200])),  
    # 1024 inputs, 10 outputs (class prediction)  
    'out': tf.Variable(tf.random_normal([200, n_classes]))  
}  
  
biases = {  
    'bc1': tf.Variable(tf.random_normal([128])),  
    'bc2': tf.Variable(tf.random_normal([64])),  
    'bc3': tf.Variable(tf.random_normal([32])),  
    'bd1': tf.Variable(tf.random_normal([200])),  
    'out': tf.Variable(tf.random_normal([n_classes]))  
}  
# Construct model  
pred = conv_net(x, weights, biases, keep_prob)  
# Define loss and optimizer  
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
# Evaluate model  
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
# Initializing the variables  
init = tf.global_variables_initializer()  
Train_ind = np.arange(train_num)  
Test_ind = np.arange(test_num)  
with tf.Session() as sess:  
    sess.run(init) 
    test_loss_of_epochs=[]
    test_acc_of_epochs=[]
    for epoch in range(0, train_epoch):  
        Total_test_loss =0
        Total_test_acc = 0
        for train_batch in range(0, int(train_batch_num)):  
            sample_ind = Train_ind[train_batch * batch_size:(train_batch + 1) * batch_size]  
            print("sichuan shifandaxue ")
            print(sample_ind)
            print("zhanghuaisu")
            batch_x = train_x[sample_ind, :]
            print(batch_x)
            batch_y = train_y[sample_ind, :] 
            # Run optimization op (backprop)            
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})  
            if train_batch % batch_size == 0:  
                # Calculate loss and accuracy  
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,  
                                                                  y: batch_y,  
                                                                  keep_prob: 1.})  
                print("Epoch: " + str(epoch + 1) + ", Batch: " + str(train_batch) + ", Loss= " + "{:.3f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
        # Calculate test loss and test accuracy  
        for test_batch in range(0, int(test_batch_num)):  
            sample_ind = Test_ind[test_batch * batch_size:(test_batch + 1) * batch_size]  
            batch_x = test_x[sample_ind, :]
            batch_y = test_y[sample_ind, :]
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x,  
                                                                        y: batch_y,  
                                                                        keep_prob: 1.})  
            Total_test_lost = Total_test_loss + test_loss  
            Total_test_acc = Total_test_acc + test_acc  
  
        Total_test_acc = Total_test_acc / test_batch_num  
        Total_test_loss = Total_test_lost / test_batch_num  
        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + "{:.3f}".format(Total_test_loss) + ", Test Accuracy= " + "{:.3f}".format(Total_test_acc))
        test_loss_of_epochs.append(Total_test_loss)
        test_acc_of_epochs.append(Total_test_acc)    
    saver = tf.train.Saver()
    model_path = "E:\zhanghuaisu\model\model.ckpt"
    save_path = saver.save(sess, model_path)  

print("---"*20)
print(test_loss_of_epochs)
print("---"*20)
print(test_acc_of_epochs)
#Total_test_acc
plt.subplot(2, 1, 1)  
plt.ylabel('Test loss')  
plt.plot(test_loss_of_epochs, 'r')  
plt.subplot(2, 1, 2)  
plt.ylabel('Test Accuracy')  
plt.plot(test_acc_of_epochs, 'r')  
  
print("All is well")  
plt.show()
