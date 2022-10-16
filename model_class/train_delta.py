# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:56:00 2019

@author: Zhengda Qin
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow_probability as tfp

#%%

class rdnW_multiK_multiDimDelta(Layer):
    def __init__(self, output_dim, prior_rho, ratio_prior, kernel_num, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_rho = prior_rho
        self.kernel_num = kernel_num
        self.ratio_prior = ratio_prior
        super().__init__(**kwargs)
        
        super().__init__(**kwargs)

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=(input_shape[1], self.kernel_num),    #shape=(input_shape[1],),
                                         initializer=initializers.constant( np.array([[0,0,0]])*
                                                np.ones([input_shape[1], self.kernel_num])),
                                         trainable=True)
        self.ratio_post = self.add_weight(name='ratio_post', 
                                         shape=(1, self.kernel_num),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(np.array([[-.5,0,.5]])),
                                         trainable=True)
#        self.bn.build(input_shape)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        
        self.kernel_sigma = tf.math.softplus(self.kernel_rho)
        self.prior_sigma = tf.math.softplus(tf.constant(self.prior_rho))
        self.ratio = tf.math.softmax(self.ratio_post)
        multinoulli_distribution = tfp.distributions.Multinomial(total_count=1, probs=self.ratio)
        sign_sigma = multinoulli_distribution.sample(self.output_dim)
        sigma_mtrx = K.dot(self.kernel_sigma,tf.transpose(tf.reduce_sum(sign_sigma,1)))
        input_weight = tf.math.multiply(sigma_mtrx,tf.random.normal([x.shape[1],
                                self.output_dim], dtype = tf.dtypes.float64))
        p_loss = self.mixGauss_prob(input_weight, self.kernel_sigma, self.ratio)
        q_loss = self.mixGauss_prob(input_weight, self.prior_sigma, self.ratio_prior)
        KL_loss = p_loss*(K.log(p_loss/K.sum(p_loss))-K.log(q_loss/K.sum(q_loss)))
        self.add_loss(K.sum(KL_loss))
#        self.KL_loss(input_weight)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)
    
#    def KL_loss(self, input_weight):
#        for i in range(self.kernel_num-1):
#            part_loss = self.log_prior_minus_post(input_weight, i)
#            self.add_loss(part_loss)
#        part_loss = self.log_prior_minus_post(input_weight, -1)
#        self.add_loss(part_loss)
        
    def mixGauss_prob(self, w, sigma, ratio):
        comp_dist = tfp.distributions.Normal(0.0, sigma[0:,0:1])
        part_loss = ratio[0:,0:1]*comp_dist.prob(w)
        for i in range(1,self.kernel_num):
            comp_dist = tfp.distributions.Normal(0.0, sigma[0:,i:i+1])
            part_loss = part_loss+ratio[0:,i:i+1]*comp_dist.prob(w)
        return part_loss



class rdnW_multiK_oneDimDelta(Layer):
    def __init__(self, output_dim, prior_rho, ratio_prior, kernel_num, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_rho = prior_rho
        self.kernel_num = kernel_num
        self.ratio_prior = ratio_prior
        super().__init__(**kwargs)
        

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=(1, self.kernel_num),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(self.prior_rho),
                                         trainable=True)
        self.ratio_post = self.add_weight(name='ratio_post', 
                                         shape=(1, self.kernel_num-1),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(self.ratio_prior),
                                         trainable=True)
#        self.bn.build(input_shape)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        
        self.kernel_sigma = tf.math.softplus(self.kernel_rho)
        self.prior_sigma = tf.math.softplus(tf.constant(self.prior_rho))
        ratio_term = np.zeros([1,self.kernel_num])
        ratio_term[0,-1] = 1.
        self.ratio_update = tf.constant(ratio_term,dtype='float64')+tf.concat([self.ratio_post,tf.reshape(tf.reduce_sum(-self.ratio_post,1),shape=[1,1])],axis=1)
        multinoulli_distribution = tfp.distributions.Multinomial(total_count=1, probs=self.ratio_update)
        sign_sigma = multinoulli_distribution.sample(x.shape[1])
        sigma_mtrx = sigma_mtrx = K.dot(tf.reduce_sum(sign_sigma,1),tf.transpose(self.kernel_sigma))
        input_weight = tf.math.multiply(sigma_mtrx,tf.random.normal([x.shape[1],
                                self.output_dim], dtype = tf.dtypes.float64))
        self.KL_loss(input_weight)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)
    
    def KL_loss(self, input_weight):
        for i in range(self.kernel_num-1):
            part_loss = self.log_prior_minus_post(input_weight, i)
            self.add_loss(part_loss)
        part_loss = self.log_prior_minus_post(input_weight, -1)
        self.add_loss(part_loss)
        
    def log_prior_minus_post(self, w, i):
        comp_1_dist = tfp.distributions.Normal(0.0, self.kernel_sigma[0][i])
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma[0][i])
        return K.mean(self.ratio_update[0][i]*comp_1_dist.log_prob(w) - tf.constant(self.ratio_prior)[i]*comp_2_dist.log_prob(w))
    
class rdnW_mulK_oneDimDelta_test(Layer):
    def __init__(self, output_dim, prior_sigma, kernel_num, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_sigma = prior_sigma
        self.kernel_num = kernel_num
        super().__init__(**kwargs)
        
        super().__init__(**kwargs)

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=(1, self.kernel_num),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(self.prior_sigma),
                                         trainable=True)
#        self.bn.build(input_shape)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
#        w_init = tf.random.normal([self.output_dim,self.output_dim], dtype = tf.dtypes.float64)
#        Q, R = tf.linalg.qr(w_init)
#        S = tf.linalg.diag(tf.linalg.norm(w_init, axis=1))
#        weightORF = tf.matmul(S,Q)
#        input_weight = kernel_sigma[0:,0:1]*weightORF[0:x.shape[1],:]
        input_weight = kernel_sigma[0:,0:1]*tf.random.normal([x.shape[1],self.output_dim],
                                   dtype = tf.dtypes.float64)
        part_loss = self.MAP_loss(input_weight, kernel_sigma[0,0])
        self.add_loss(part_loss)
        for i in range(1,self.kernel_num):
#            weight_term = kernel_sigma[0:,i:i+1]*weightORF[0:x.shape[1],:]
            weight_term = kernel_sigma[0:,i:i+1]*tf.random.normal([x.shape[1],self.output_dim],
                                   dtype = tf.dtypes.float64)
            input_weight = tf.concat([input_weight, weight_term], 1)
            part_loss = self.MAP_loss(weight_term, kernel_sigma[0,i])
            self.add_loss(part_loss)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)
    
    def MAP_loss(self, w, sigma):
        prior_dist = tfp.distributions.Normal(0, sigma)
        return -K.sum(prior_dist.log_prob(w))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class rdnW_mulK_multiDimDelta(Layer):
    def __init__(self, output_dim, prior_sigma, kernel_num, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_sigma = prior_sigma
        self.kernel_num = kernel_num
        super().__init__(**kwargs)

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=([input_shape[1], self.kernel_num]),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(self.prior_sigma),
                                         trainable=True)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
#        w_init = tf.random.normal([self.output_dim,self.output_dim], dtype = tf.dtypes.float64)
#        Q, R = tf.linalg.qr(w_init)
#        S = tf.linalg.diag(tf.linalg.norm(w_init, axis=1))
#        weightORF = tf.matmul(S,Q)
#        input_weight = kernel_sigma[0:,0:1]*weightORF[0:x.shape[1],:]
        input_weight = kernel_sigma[0:,0:1]*tf.random.normal([x.shape[1],self.output_dim],
                                   dtype = tf.dtypes.float64)
        part_loss = self.MAP_loss(input_weight, kernel_sigma[0,0])
        self.add_loss(part_loss)
        for i in range(1,self.kernel_num):
#            weight_term = kernel_sigma[0:,i:i+1]*weightORF[0:x.shape[1],:]
            weight_term = kernel_sigma[0:,i:i+1]*tf.random.normal([x.shape[1],self.output_dim],
                                   dtype = tf.dtypes.float64)
            input_weight = tf.concat([input_weight, weight_term], 1)
            part_loss = self.MAP_loss(weight_term, kernel_sigma[0,i])
            self.add_loss(part_loss)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)
    
    def MAP_loss(self, w, sigma):
        prior_dist = tfp.distributions.Normal(0, sigma)
        return -K.sum(prior_dist.log_prob(w))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class rdnW_oneK_multiDimDelta(Layer):
    def __init__(self, output_dim, prior_sigma_init, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_sigma_init = prior_sigma_init
        super().__init__(**kwargs)

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=(input_shape[1],1),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(
                                            self.prior_sigma_init*np.ones([input_shape[1], 1])),
                                         trainable=True)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
#        w_init = tf.random.normal([self.output_dim,self.output_dim], dtype = tf.dtypes.float64)
#        Q, R = tf.linalg.qr(w_init)
#        S = tf.linalg.diag(tf.linalg.norm(w_init, axis=1))
#        weightORF = tf.matmul(S,Q)
#        input_weight = kernel_sigma*weightORF[0:x.shape[1],:]
        input_weight = kernel_sigma*tf.random.normal([x.shape[1],self.output_dim],
                                   dtype = tf.dtypes.float64)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    
class rdnW_oneK_oneDimDelta(Layer):
    def __init__(self, output_dim, prior_sigma_init, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.prior_sigma_init = prior_sigma_init
        super().__init__(**kwargs)

    def build(self, input_shape):  

        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                         shape=(1, 1),    #shape=(input_shape[1],),
                                         initializer=initializers.constant(self.prior_sigma_init),
                                         trainable=True)
#        self.input_weight = self.add_weight(name='input_weight', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=initializers.RandomNormal(stddev=prior),
#                                       trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        input_weight = tf.random.normal([x.shape[1],self.output_dim],stddev=kernel_sigma,dtype = tf.dtypes.float64)
        hid_out_sin = tf.math.sin(K.dot(x, input_weight))
        hid_out_cos = tf.math.cos(K.dot(x, input_weight))
        return tf.concat([hid_out_sin, hid_out_cos], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)