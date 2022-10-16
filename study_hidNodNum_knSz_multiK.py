# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:05:01 2019

@author: Zhengda Qin
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import model_class.train_delta as td 
import sys  
sys.path.append(r'F:\程序\data_and_formating\pyData')  
from data_formating import data_formating 
import scipy.special as ss

dataName = 'Ailerons'
dgenerate = data_formating(data_normal = 1)
input_train, desire_train, input_test, desire_test = dgenerate.data_format(name=dataName, MC=0)
#%%

tf.keras.backend.set_floatx('float64')   
NLLscale = 0.1*np.mean(np.std(input_train,axis=1))
def neg_log_likelihood(y_obs, y_pred, sigma=NLLscale):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return -dist.log_prob(y_obs)

def MSE_loss(y_obs, y_pred):
    err = y_obs-y_pred
    return err*err
    

def loss(model, inputs, targets, loss_ratio, tloss):
    hid_out = model(inputs)
    hid_hf1 = hid_out[0:int(hid_out.shape[0]/2),0:]
    hid_hf2 = hid_out[int(hid_out.shape[0]/2):,0:]
    weight = out_weight(hid_hf1,targets[0:int(hid_out.shape[0]/2),0:])
    prediction = hid_hf2@weight
    if tloss == 'NLL':
        loss_value = neg_log_likelihood(targets[int(hid_out.shape[0]/2):,0:], prediction)
    elif tloss == 'MSE':
        loss_value = MSE_loss(targets[int(hid_out.shape[0]/2):,0:], prediction)
    MSE = K.mean(loss_value)
    MAP = sum(model.losses)
    return MSE+loss_ratio*MAP, MSE, MAP

def out_weight(hid_out,y_obs):
    weight_new = tf.linalg.inv(tf.transpose(hid_out)@hid_out)@tf.transpose(hid_out)@y_obs
    return weight_new


def RFF_LS(weight, X, y, X_test, y_test, rgl_fact):
    hid_out = X@weight
    act_hid_out = np.hstack((np.sin(hid_out),np.cos(hid_out)))
    w_out = np.linalg.inv(act_hid_out.T@act_hid_out+rgl_fact*np.identity(np.shape(act_hid_out)[1]))@act_hid_out.T@y
    err = np.reshape(y,(train_size,1))-act_hid_out@w_out
    MSE = tf.transpose(err)@err/train_size
    hid_out_test = np.dot(X_test,weight)
    y_test_pred = np.hstack((np.sin(hid_out_test),np.cos(hid_out_test)))@w_out
    err_test = np.reshape(y_test,(test_size,1))-y_test_pred
    MSE_test = tf.transpose(err_test)@err_test/test_size
#    print("MSE: {:.3f}, testMSE: {:.3f}".format(MSE[0,0], MSE_test[0,0]))
    return MSE, MSE_test

class earlyStp(object):
    def __init__(self, patience=0):
        self.patience = patience
        self.best_weights = None
        self.wait = 0
        self.best = np.Inf
        self.prev = np.Inf
        self.breakSign = 0
        
    def end_epch(self, model, logs=None):
        self.log = logs
        if self.log>self.best:
            self.wait += 1
        else:
            self.wait = 0
            self.best = self.log
            self.best_weights = model.get_weights()
        if self.wait>self.patience:
            self.breakSign = 1
        return self.best_weights, self.breakSign, self.best

def ORF_weight(otn_matrix_size, kernel_num):
    w_init = np.random.randn(otn_matrix_size,otn_matrix_size)
    Q, R = np.linalg.qr(w_init)
    S = np.diag(np.linalg.norm(w_init, axis=1))
    weightORF = S@Q
    for i in range(1,kernel_num):
        w_init = np.random.randn(otn_matrix_size,otn_matrix_size)
        Q, R = np.linalg.qr(w_init)
        S = np.diag(np.linalg.norm(w_init, axis=1))
        weight_term = S@Q
        weightORF = np.hstack([weightORF, weight_term])
    return weightORF
        
    
# %%

train_size = dgenerate.train_num
test_size = dgenerate.test_num
input_dim = dgenerate.data_dim-1
batch_size = train_size
num_batches = train_size / batch_size
prior_rho = np.array([[3.,0.,-3.]])
kernel_num = np.shape(prior_rho)[1]
rho_all = (prior_rho*np.ones([input_dim, kernel_num])).reshape(-1)
ratio_prior = np.array([[1.,1.,1.]])
ratio_all = ratio_prior
#prior_rho = prior_r ho*np.ones([input_dim, np.shape(prior_rho)[0]])
#prior_rho = 0
rho_number = 10
kernel_num = np.shape(prior_rho)[1]
loss_ratio = 0.001

rst1 = np.zeros([rho_number,3])
rst2 = np.zeros([rho_number,3])
rst3 = np.zeros([rho_number,3])
for ii in range(1,rho_number+1):
    
    @tf.function
    def train(model, x, y_obs, loss_ratio, tloss):
        with tf.GradientTape() as tape:   #persistent=True
            loss_value, NLL, KL = loss(model, x, y_obs, loss_ratio, tloss)
        grads = tape.gradient(loss_value, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Compare predicted label to actual label
        hid_out = model(x)
        weight_new = out_weight(hid_out, y_obs)
        prediction = hid_out@weight_new
        epoch_accuracy(y_obs, prediction)
        epoch_loss_avg(loss_value)
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        return loss_value, NLL, KL
    hid_layer_num = input_dim*ii
    otn_matrix_size = input_dim*ii
    x_in = Input(shape=(input_dim,))
#    x = td.rdnW_mulK_multiDimDelta(output_dim=otn_matrix_size, prior_sigma=prior_sigma, kernel_num=kernel_num, activation=None)(x_in)
    x = td.rdnW_multiK_multiDimDelta(output_dim=hid_layer_num, prior_rho=prior_rho, ratio_prior=ratio_prior, kernel_num=kernel_num, activation=None)(x_in)
    #x = td.rdnW_oneK_oneDimDelta(output_dim = hid_layer_num, prior_sigma_init=prior_sigma_init, activation=None)(x_in)
    model = Model(x_in, x)
    model.summary()
    
    
    #%%
    
    train_loss_results = []
    train_accuracy_results = []
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()
    inputs = tf.constant(input_train)
    labels = tf.constant(desire_train)
    dataset = tf.data.Dataset.from_tensor_slices((inputs,labels))
    dataset = dataset.shuffle(4000).batch(4000).repeat(2000)
    optimizer = optimizers.Adam(learning_rate=0.03)  
    #one_dimension
    #Ailerons: dataset.shuffle(7000).batch(6750).repeat(2000), learning_rate=0.1, patience = 400
    #multi_dimension
    #Ailerons: dataset.shuffle(7000).batch(2000).repeat(2000), learning_rate=0.03, patience = 800, prior_sigma_init = 0
    #bank8FM: dataset.shuffle(4000).batch(4000).repeat(2000),  learning_rate=0.07, patience=200
    #bank32nh: dataset.shuffle(4000).batch(2000).repeat(5000),  learning_rate=0.02, patience=3000， prior = 0, delta = 10, rgl1 = 1, rgl2 = 10
    #CPU_small: dataset.shuffle(4000).batch(4000).repeat(5000), learning_rate=0.02, patience=3000
    #CPU_big: dataset.shuffle(4000).batch(4000).repeat(2000), learning_rate=0.03, patience=200
    #elevators: dataset.shuffle(8000).batch(8000).repeat(2000), learning_rate=0.02, patience=500
    #cal: dataset.shuffle(10000).batch(10000).repeat(2000), learning_rate=0.01, patience=500, delta=3, rgl=rgl2=0
    epoch = 0
    weight = 0
    earlystp = earlyStp(patience = 1000)
    for x, y_obs in dataset:
        epoch = epoch+1
        loss_all, loss_NLL, loss_KL = train(model, x, y_obs, loss_ratio, 'NLL')
        best_weight, breakSign, bestLoss = earlyStp.end_epch(earlystp, model, logs = loss_all)
    #    print("best:{:.5f}".format(best.numpy()))
        if epoch % 50 == 0:
            print("Epoch {:03d}: likeliLoss: {:.3f}, NLL: {:.3f}, KL:{:.3f}, ratio:{:.3f}".format(epoch,
                                                  epoch_loss_avg.result(),
                                                  loss_NLL.numpy(),
                                                  loss_KL.numpy(),
                                                  model.trainable_variables[1].numpy()[0,0]))
        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        if breakSign:
            break;
        
    #%%
    
    
    rho = best_weight[0]
    rho_all = np.vstack([rho_all,rho.reshape(-1)])
    sigma = tf.math.softplus(rho)
    ratio_post = model.trainable_variables[1].numpy()
    ratio_all = np.vstack([ratio_all,ratio_post.reshape(-1)])
    MSE1 = np.zeros([50,1])
    MSE_test1 = np.zeros([50,1])
    MSE_test2 = np.zeros([50,1])
    MSE_test3 = np.zeros([50,1])
    MSE2 = np.zeros([50,1])
    MSE3 = np.zeros([50,1])
    for j in range(50):
        weightORF = ORF_weight(otn_matrix_size, 1)
        probs = ss.softmax(ratio_post)
#        ratio_post[ratio_post<0]=0.
#        ratio_term = np.zeros([1,kernel_num])
#        ratio_term[0,-1] = 1.
#        ratio = ratio_term+np.hstack([ratio_post,np.sum(-ratio_post,1).reshape((1,1))])
        sign_sigma = np.random.multinomial(1,probs[0],size=hid_layer_num)
        sigma_mtrx = sigma.numpy()@sign_sigma.T
        weight = sigma_mtrx*weightORF[0:input_dim,:]
#        w_init = np.random.randn(otn_matrix_size,otn_matrix_size)
#        Q, R = np.linalg.qr(w_init)
#        S = np.diag(np.linalg.norm(w_init, axis=1))
#        weightORF = S@Q
#        weight[0:x.shape[1],0:otn_matrix_size] = sigma.numpy()[0:,0:1]*weightORF[0:x.shape[1],0:otn_matrix_size]
#        for i in range(1,kernel_num):
#            weight[0:x.shape[1],i*otn_matrix_size：(i+1)*otn_matrix_size] = sigma.numpy()[0:,0:1]*weightORF[0:x.shape[1],0:otn_matrix_size]
#        weight = sigma.numpy().T*weightORF[0:input_dim,:]
        
        rgl_fact = 0
        MSE1[j,0], MSE_test1[j,0] = RFF_LS(weight, input_train, desire_train, input_test, desire_test, rgl_fact)
        
        delta = 5
        rgl_fact2 = 0
        weightRFF = 1/delta*np.random.randn(np.shape(weight)[0], np.shape(weight)[1])
        MSE2[j,0], MSE_test2[j,0] = RFF_LS(weightRFF, input_train, desire_train, input_test, desire_test, rgl_fact2)
    
        ORF = 1/delta*weightORF[0:input_dim,:]
        MSE3[j,0], MSE_test3[j,0] = RFF_LS(ORF, input_train, desire_train, input_test, desire_test, rgl_fact2)
    
    print("MSE1: {:.3f}, testMSE1: {:.3f}, testMSE_std1: {:.3f}".format(np.mean(MSE1), np.mean(MSE_test1), np.std(MSE_test1)))
    print("MSE2: {:.3f}, testMSE2: {:.3f}, testMSE_std2: {:.3f}".format(np.mean(MSE2), np.mean(MSE_test2), np.std(MSE_test2)))
    print("MSE3: {:.3f}, testMSE3: {:.3f}, testMSE_std3: {:.3f}".format(np.mean(MSE3), np.mean(MSE_test3), np.std(MSE_test3)))
    rst1[ii-1,:] = [np.mean(MSE1), np.mean(MSE_test1), np.std(MSE_test1)]
    rst2[ii-1,:] = [np.mean(MSE2), np.mean(MSE_test2), np.std(MSE_test2)]
    rst3[ii-1,:] = [np.mean(MSE3), np.mean(MSE_test3), np.std(MSE_test3)]
#%%
   
index = np.arange(1,rho_number+1)
plt.figure()
plt.plot(index, rst1[:,1], label='milti-delta testing MSE')
plt.fill_between(index, 
                 rst1[:,1] + rst1[:,2], 
                 rst1[:,1] - rst1[:,2], 
                 alpha=0.5, label='milti-delta std value')
plt.plot(index, rst3[:,1], label='ORF testing MSE')
plt.fill_between(index, 
                 rst3[:,1] + rst3[:,2], 
                 rst3[:,1] - rst3[:,2], 
                 alpha=0.5, label='ORF std value')
plt.xlabel('D/d')
plt.ylabel('MSE')
#plt.title('Noisy training data and ground truth')
plt.legend();
plt.savefig(str(dataName)+'.pdf')
plt.show()
#%%

plt.figure()
for dim in range(0,kernel_num):
    plt.plot(ratio_all[1:,dim])
    #%%
plt.figure()
for dim in range(0,kernel_num*input_dim):
    plt.plot(rho_all[1:,dim])
    #%%

np.savez(dataName, rho_all, ratio_all, rst1, rst2,rst3, delta, rgl_fact, rgl_fact2)

##%%
#para = np.load(str(dataName)+'.npz')
#rho_all = para.f.arr_0[1:,:]


