from __future__ import division
from __future__ import print_function 
import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os.path
import os  
import random
import copy
import codecs
import collections
import math
from evaluation import evaluate_process, evaluate_admm
import gram_schmidt
import time
from scipy import stats
import sys


###################################
###Parameters are defined here#####
###################################
user_list_len = 80 # it is 80 for sst data, 400 for imdb data.
item_list_len = 50
binary_len = 64
scale_r = binary_len/2
num_users = 7000  ## the volumns of the users for ml
num_items = 4000 ##  the volumns of the items for ml 
# 
num_neg_sample = 1
batch_size = 256 
hidden_unit = binary_len  ## the hidden units of the LSTM

mu_max = 1000  # to prevent mu become too large
rho = 1.1   ## the rate to control the convergence

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def batch_norm(x, n_out, scope='bn'):
 
	a=  True
	phase_train = tf.cast(a, dtype= tf.bool)
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		# mean = (ema.average(batch_mean))
		# var = (ema.average(batch_var))
		mean, var = tf.cond(phase_train,
							mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed
class Abbre_Repair(object):
	"""docstring for VAE"""
	def __init__(self):	
		self.params = []
		self.x_user_item = tf.placeholder(tf.int32, [None, item_list_len], name="input_x_item")
		self.x_item_user = tf.placeholder(tf.int32, [None, user_list_len], name="input_x_user")
		self.embedding_user = tf.Variable(tf.random_uniform([num_users, 10], -1.0, 1.0), name='embedding_init_for_item', trainable=True)
		self.embedding_item = tf.Variable(tf.random_uniform([num_items, 10], -1.0, 1.0), name='embedding_init_for_user', trainable=True)
		self.y = tf.placeholder(tf.float32, [None], name="input_y") 
		self.l2_loss = tf.constant(0.0)
		# self.words = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), Max_num, 1.0, 0.0), [-1, seq_words_len, RM_dim, RM_dim])   
		self.E = tf.placeholder(tf.float32, [None, binary_len], name = "E")
		self.F = tf.placeholder(tf.float32, [None, binary_len], name = "F")
		# self.B = tf.placeholder(tf.float32, [None, binary_len], name = "B")
		# self.D = tf.placeholder(tf.float32, [None, binary_len], name = "D")
		self.P = tf.placeholder(tf.float32, [None, binary_len], name = "P")
		self.Q = tf.placeholder(tf.float32, [None, binary_len], name = "Q")
		self.mu = tf.placeholder(tf.float32, name = "mu")
		# B_i = self.LSTM_user_tf(self.x_user_item)  
		# D_j = self.LSTM_item_tf(self.x_item_user)  
		B_i = self.CNN_user(self.x_user_item)
		D_j = self.CNN_item(self.x_item_user)
		# B_i = self.MF_user(self.x_user_item)
		# D_j = self.MF_item(self.x_item_user)
		alpha = 20
		B_i = tf.nn.softsign(alpha*B_i)
		D_j = tf.nn.softsign(alpha*D_j)
		

		self.B_i = B_i
		self.D_j = D_j
		
		self.J1 =  tf.reduce_mean(tf.nn.l2_loss(self.y - tf.reduce_sum(tf.multiply(B_i, D_j), reduction_indices = 1))) ## useful for ml
		
		self.J2 = tf.reduce_mean((self.mu/2.0)*((tf.nn.l2_loss(self.B_i - (self.P + (1.0/self.mu)*self.E))) + tf.nn.l2_loss(self.D_j - (self.Q + 1.0/self.mu*self.F))))
		
		self.J3 = tf.nn.l2_loss(tf.trace(binary_len*tf.eye(batch_size) - tf.matmul(B_i, tf.transpose(B_i, perm = [0,1])))) + tf.nn.l2_loss(tf.trace(binary_len*tf.eye(batch_size) - tf.matmul(D_j, tf.transpose(D_j, perm = [0,1]))))
		
		self.loss_lstm = (self.J1 + self.J2 + self.J3)

		self.acc = self.loss_lstm

	def MF_user(self, x):
		# embedding_init = tf.Variable(tf.random_uniform([num_items, 10], -1.0, 1.0), name='embedding_init_user', trainable=True)
		user_embedding = tf.nn.embedding_lookup(self.embedding_item, x)
		
		user_embedding = tf.expand_dims(user_embedding, -1)
		user_embedding = tf.reshape(user_embedding,[batch_size, item_list_len*10])
		W_user = weight_variable([item_list_len*10, binary_len]) 
		b_user = bias_variable([binary_len])
		self.params.append(W_user)
		self.params.append(b_user)
		self.l2_loss += tf.nn.l2_loss(W_user)
		out = tf.nn.softsign(tf.nn.xw_plus_b(user_embedding, W_user, b_user))
	

		return out
	def MF_item(self, x):
		# embedding_init = tf.Variable(tf.random_uniform([num_users, 10], -1.0, 1.0), name='embedding_init_item', trainable=True)
		item_embedding = tf.nn.embedding_lookup(self.embedding_user, x)
		item_embedding = tf.reshape(item_embedding,[batch_size, user_list_len*10])
		W_item = weight_variable([user_list_len*10, binary_len]) 
		b_item = bias_variable([binary_len])
		self.params.append(W_item)
		self.params.append(b_item)
		self.l2_loss += tf.nn.l2_loss(W_item)
		out = tf.nn.softsign(tf.nn.xw_plus_b(item_embedding, W_item, b_item))
	

		return out
	def CNN_user(self, x):
		with tf.name_scope("CNN_user"):
			filter_sizes = [2,3,4,5,7,9,11, 13]
			num_filters = [200,200,300,200, 200, 100,200, 200]  
			user_embedding = tf.nn.embedding_lookup(self.embedding_item, x)
			# user_embedding = tf.nn.dropout(user_embedding, self.dropout_keep_prob)
			user_embedding = tf.expand_dims(user_embedding, -1)

			pooled_outputs = []
			 
			for filter_size, num_filter in zip(filter_sizes, num_filters):
				with tf.name_scope("Senti-conv"):
					filter_shape = [filter_size, 10, 1, num_filter]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01),name="W_r", trainable=True) 
					b = tf.Variable(tf.constant(0.01, shape=[num_filter]), name="b_r", trainable=True)
					user_conv = tf.nn.conv2d(user_embedding, W, strides=[1,1,1,1], padding="VALID", name="conv") 
					self.params.extend([W, b])
					h = tf.nn.relu(tf.nn.bias_add(user_conv, b), name="relu_r")
					h = batch_norm(h, num_filter, scope='user_conv-bn1')

					pooled1 =tf.nn.avg_pool(h, ksize=[1, item_list_len + 1 - filter_size,1,1],strides=[1,1,1,1], padding="VALID", name="pooled")
					pooled_outputs.append(pooled1)
					self.l2_loss += tf.nn.l2_loss(W)
					self.l2_loss += tf.nn.l2_loss(b)
			num_filter = sum(num_filters) 
			pooled = tf.concat(pooled_outputs,3)
			pooled_flat = tf.reshape(pooled, [-1, num_filter])  
			W_bin_user = weight_variable([num_filter, binary_len])
			b_bin_user = bias_variable([binary_len])
			self.params.append(W_bin_user)
			self.params.append(b_bin_user)
			self.l2_loss += tf.nn.l2_loss(W_bin_user)
			out = (tf.nn.xw_plus_b(pooled_flat, W_bin_user, b_bin_user))
			return out
	def CNN_item(self, x):
		with tf.name_scope("CNN_item"):
			filter_sizes = [2,3,4,5,7,9,11, 13]
			num_filters = [200,200,300,200, 200, 100,200, 200]  
			item_embedding = tf.nn.embedding_lookup(self.embedding_user, x) 
			item_embedding = tf.expand_dims(item_embedding, -1)

			pooled_outputs = []
			 
			for filter_size, num_filter in zip(filter_sizes, num_filters):
				with tf.name_scope("Senti-conv"):
					filter_shape = [filter_size, 10, 1, num_filter]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01),name="W_r", trainable=True) 
					b = tf.Variable(tf.constant(0.01, shape=[num_filter]), name="b_r", trainable=True)
					item_conv = tf.nn.conv2d(item_embedding, W, strides=[1,1,1,1], padding="VALID", name="conv") 
					self.params.extend([W, b])
					h = tf.nn.relu(tf.nn.bias_add(item_conv, b), name="relu_r")
					h = batch_norm(h, num_filter, scope='item_conv-bn1')

					pooled1 =tf.nn.avg_pool(h, ksize=[1, user_list_len + 1 - filter_size,1,1],strides=[1,1,1,1], padding="VALID", name="pooled")
					pooled_outputs.append(pooled1)
					self.l2_loss += tf.nn.l2_loss(W)
					self.l2_loss += tf.nn.l2_loss(b)
			num_filter = sum(num_filters) 
			pooled = tf.concat(pooled_outputs,3)
			pooled_flat = tf.reshape(pooled, [-1, num_filter])  
			W_bin_item = weight_variable([num_filter, binary_len])
			b_bin_item = bias_variable([binary_len])
			self.params.append(W_bin_item)
			self.params.append(b_bin_item)
			self.l2_loss += tf.nn.l2_loss(W_bin_item)
			out = (tf.nn.xw_plus_b(pooled_flat, W_bin_item, b_bin_item))
			return out

	def LSTM_user_tf(self, x):
		# x = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(x, [-1])), num_items, 1.0, 0.0), [-1, num_items, item_list_len])
		num_hidden = 10
		with tf.variable_scope("LSTM_user_tf"):
			cell_fw = tf.contrib.rnn.GRUCell(num_hidden)
			cell_bw = tf.contrib.rnn.GRUCell(num_hidden)
			# cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
			# cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
			inputs = tf.nn.embedding_lookup(self.embedding_item, x, validate_indices=False)
			# inputs = tf.one_hot(x, depth = num_items, on_value=1.0, off_value=0.0)
			val, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = inputs, dtype=tf.float32)
			val_fw = tf.transpose(val[0], [1, 0, 2])
			val_fw = tf.gather(val_fw, int(val_fw.get_shape()[0]) - 1)
			val_bw = tf.transpose(val[1], [1, 0, 2])
			val_bw = tf.gather(val_bw, int(val_bw.get_shape()[0]) - 1)
			val = tf.concat([val_fw, val_bw], 1)
			W_encoder = weight_variable([num_hidden*2, hidden_unit])
			b_encoder = bias_variable([hidden_unit])
			self.params.append(W_encoder)
			self.params.append(b_encoder)
			self.l2_loss += tf.nn.l2_loss(W_encoder)
	 
			out = (tf.nn.xw_plus_b(val, W_encoder, b_encoder))
		return out
	def LSTM_item_tf(self, x):
		# x = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(x, [-1])), num_users, 1.0, 0.0), [-1, num_users, user_list_len])
		num_hidden = 30
		with tf.variable_scope("LSTM_item_tf"):
			cell_fw = tf.contrib.rnn.GRUCell(num_hidden)
			cell_bw = tf.contrib.rnn.GRUCell(num_hidden)
			# cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
			# cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
			inputs = tf.nn.embedding_lookup(self.embedding_user, x, validate_indices=False)
			# inputs = tf.one_hot(x, depth=num_users, on_value = 1.0, off_value = 0.0)
			val, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = inputs, dtype=tf.float32)
			val_fw = tf.transpose(val[0], [1, 0, 2])
			val_fw = tf.gather(val_fw, int(val_fw.get_shape()[0]) - 1)
			val_bw = tf.transpose(val[1], [1, 0, 2])
			val_bw = tf.gather(val_bw, int(val_bw.get_shape()[0]) - 1)
			val = tf.concat([val_fw, val_bw], 1)
			W_encoder = weight_variable([num_hidden*2, hidden_unit])
			b_encoder = bias_variable([hidden_unit])
			self.params.append(W_encoder)
			self.params.append(b_encoder)
			self.l2_loss += tf.nn.l2_loss(W_encoder)
			out = (tf.nn.xw_plus_b(val, W_encoder, b_encoder))
		return out

	def LSTM_user(self, x):
		lstm_user_unit = self.lstm_user_unit(self.params) 
		lstm_out_raw = tf.zeros([batch_size, hidden_unit]) 
		states = tf.stack([lstm_out_raw, lstm_out_raw])
		def _dec_recurrence(i, states): 
			x_t = x[:,i]
			x_t = tf.one_hot(indices = x_t, depth = num_items) 

			states = lstm_user_unit(x_t, states) 
			i = i + 1
			return i, states

		with tf.name_scope("lstm_user"):
			_, final_states = tf.while_loop(
				cond = lambda i, _1: i < item_list_len ,
				body = _dec_recurrence,
				loop_vars = (tf.constant(0, dtype=tf.int32), states))
		hidden, out = tf.unstack(final_states)
		return out

	def LSTM_item(self, x):
		lstm_item_unit = self.lstm_item_unit(self.params) 
		lstm_out_raw = tf.zeros([batch_size, hidden_unit]) 
		states = tf.stack([lstm_out_raw, lstm_out_raw])
		def _dec_recurrence(j, states): 
			x_t = x[:,j]
			x_t = tf.one_hot(indices = x_t, depth = num_users)
			# x_t = tf.contrib.layers.batch_norm(x_t, center=True)
			states = lstm_item_unit(x_t, states) 
			j = j + 1
			return j, states

		with tf.name_scope("lstm_item"):
			_, final_states = control_flow_ops.while_loop(
				cond = lambda j, _1: j < user_list_len ,
				body = _dec_recurrence,
				loop_vars = (tf.constant(0, dtype=tf.int32), states))
		hidden, out = tf.unstack(final_states)
		return out
 
 
	def lstm_user_unit(self, params): 
		with tf.name_scope("lstm-user-unit"):  
			self.Wi = weight_variable([num_items, hidden_unit]) 
			self.Ui = weight_variable([hidden_unit, hidden_unit])
			self.bi = bias_variable([hidden_unit])

			self.Wf = weight_variable([num_items, hidden_unit]) 
			self.Uf = weight_variable([hidden_unit, hidden_unit])
			self.bf = bias_variable([hidden_unit])

			self.Wog = weight_variable([num_items, hidden_unit]) 
			self.Uog = weight_variable([hidden_unit, hidden_unit])
			self.bog = bias_variable([hidden_unit])

			self.Wc = weight_variable([num_items, hidden_unit]) 
			self.Uc = weight_variable([hidden_unit, hidden_unit])
			self.bc = bias_variable([hidden_unit])
 
  
			params.extend([
				 self.Wi, self.Ui, self.bi, 
				 self.Wf, self.Uf, self.bf, 
				 self.Wog, self.Uog, self.bog, 
				 self.Wc, self.Uc, self.bc])

			def unit(x, hidden_memory_tm1):

				previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
				# Input Gate
				i = tf.sigmoid(
					tf.matmul(x, self.Wi) + 
					tf.matmul(previous_hidden_state, self.Ui) + self.bi
				)

				# Forget Gate
				f = tf.sigmoid(
					tf.matmul(x, self.Wf) + 
					tf.matmul(previous_hidden_state, self.Uf) + self.bf
				)

				# Output Gate
				o = tf.sigmoid(
					tf.matmul(x, self.Wog) +  
					tf.matmul(previous_hidden_state, self.Uog) + self.bog
				)

				# New Memory Cell
				c_ = tf.nn.tanh(
					tf.matmul(x, self.Wc) + 
					tf.matmul(previous_hidden_state, self.Uc) + self.bc
				)

				# Final Memory cell
				c = f * c_prev + i * c_ 

				# Current Hidden state
				current_hidden_state = o * tf.nn.tanh(c)

				return tf.stack([current_hidden_state, c])

		return unit


	def lstm_item_unit(self, params):
		# Weights and Bias for input and hidden tensor
		with tf.name_scope("lstm-item-unit"):  
			self.Wi = weight_variable([num_users, hidden_unit]) 
			self.Ui = weight_variable([hidden_unit, hidden_unit])
			self.bi = bias_variable([hidden_unit])

			self.Wf = weight_variable([num_users, hidden_unit]) 
			self.Uf = weight_variable([hidden_unit, hidden_unit])
			self.bf = bias_variable([hidden_unit])

			self.Wog = weight_variable([num_users, hidden_unit]) 
			self.Uog = weight_variable([hidden_unit, hidden_unit])
			self.bog = bias_variable([hidden_unit])

			self.Wc = weight_variable([num_users, hidden_unit]) 
			self.Uc = weight_variable([hidden_unit, hidden_unit])
			self.bc = bias_variable([hidden_unit])
 
  
			params.extend([
				 self.Wi, self.Ui, self.bi, 
				 self.Wf, self.Uf, self.bf, 
				 self.Wog, self.Uog, self.bog, 
				 self.Wc, self.Uc, self.bc])

			def unit(x, hidden_memory_tm1):

				previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
				# Input Gate
				i = tf.sigmoid(
					tf.matmul(x, self.Wi) + 
					tf.matmul(previous_hidden_state, self.Ui) + self.bi
				)

				# Forget Gate
				f = tf.sigmoid(
					tf.matmul(x, self.Wf) + 
					tf.matmul(previous_hidden_state, self.Uf) + self.bf
				)

				# Output Gate
				o = tf.sigmoid(
					tf.matmul(x, self.Wog) +  
					tf.matmul(previous_hidden_state, self.Uog) + self.bog
				)

				# New Memory Cell
				c_ = tf.nn.tanh(
					tf.matmul(x, self.Wc) + 
					tf.matmul(previous_hidden_state, self.Uc) + self.bc
				)

				# Final Memory cell
				c = f * c_prev + i * c_ 

				# Current Hidden state
				current_hidden_state = o * tf.nn.tanh(c)

				return tf.stack([current_hidden_state, c])

		return unit 
def scale_rate(rate):
	rate = float(rate)
	if (rate)>=1:
		rate = binary_len#(rate-1)*(binary_len/4)
		# rate = binary_len
	else:
		rate = -binary_len#- binary_len
	return rate 
 
def load_data(file_train, file_test):   

	user_item_list = {}## the item which the user have bought
	item_user_list = {}## the people who bought the item
	user_item_rate = {} ## the rate given by user on the item 
	with codecs.open(file_train) as f_train:
		for line in f_train:
			user, item ,rate, time = line.strip().split('\t')
			user = int(user)
			item = int(item)
			rate = scale_rate(rate)
			if user not in user_item_list:
				user_item_list[user]= [item]
			else:
				user_item_list[user].append(item) 

			if item not in item_user_list:
				item_user_list[item] = [user]
			else:
				item_user_list[item].append(user)
			rate_obj = (user, item)
			if rate_obj not in user_item_rate:
				user_item_rate[rate_obj] = int(rate)
	user_item_list_test = copy.deepcopy(user_item_list)## the item which the user have bought
	item_user_list_test = copy.deepcopy(item_user_list)## the people who bought the item
	user_item_rate_test = copy.deepcopy(user_item_rate) ## the rate given by user on the item 

	with codecs.open(file_test) as f_test:
		for line in f_test:
			user, item ,rate, time = line.strip().split('\t')
			user = int(user)
			item = int(item)
			rate = scale_rate(rate)
			if user not in user_item_list:
				user_item_list_test[user]= [item]
			else:
				user_item_list_test[user].append(item) 

			if item not in item_user_list:
				item_user_list_test[item] = [user]
			else:
				item_user_list_test[item].append(user)
			rate_obj = (user, item)
			if rate_obj not in user_item_rate:
				user_item_rate_test[rate_obj] = int(rate)

	def padding(dicts_, fix_len, maxnum):

		for key in dicts_:
			values = dicts_[key]
			if len(values)<fix_len:
				for kk in range(fix_len - len(values)):
					# values.append(-1)
					values.insert(0, (maxnum - 1)) 
			elif len(values)> fix_len:
				values = values[len(values)-fix_len:]
			dicts_[key] = values
		return dicts_

	user_item_list = padding(user_item_list, item_list_len, num_items)
	item_user_list = padding(item_user_list, user_list_len, num_users)
	# for kk in range(len(item_user_list)):
	#	print (item_user_list[kk])
	user_item_list_test = padding(user_item_list_test, item_list_len, num_items)
	item_user_list_test = padding(item_user_list_test, user_list_len, num_users)



	train_set = (user_item_list, item_user_list, user_item_rate)
	test_set = (user_item_rate_test, item_user_list_test, user_item_rate_test)
	return train_set, test_set

def load_evaluation(filename):
	### the key is the user and the values is the items, the first item in the items is the right one, others are the non-hit ones
	evalueation_set = {}
	with codecs.open(filename,'r') as f:
		for line in f:
			user_item = line.strip().split('\t')[0]
			user, hit_item = user_item.strip().split('(')[1].split(')')[0].split(',')
			user = int(user)
			hit_item = int(hit_item)

			negative_items = [int(num) for num in line.strip().split('\t')[1:]]
			negative_items.insert(0, hit_item)
			evalueation_set[user] = negative_items
	return evalueation_set




def main(gpu_num, exp, percent): 
	model = Abbre_Repair()
	params = model.params  
	train_step = tf.train.AdamOptimizer(5e-4).minimize(model.loss_lstm)  

	### loading the data from the files,
	file_train = './data/amazon.train.rating'
	file_test = './data/amazon.test.rating'
	file_negative = './data/amazon.test.negative'

	train_set, test_set= load_data(file_train, file_test)
	user_item_list, item_user_list, user_item_rate = train_set
	user_item_list_test, item_user_list_test, user_item_rate_test = test_set
	evalueation_set = load_evaluation(file_negative)

	saver = tf.train.Saver()
 
	config_gpu = tf.ConfigProto() 
	config_gpu.gpu_options.per_process_gpu_memory_fraction = percent
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

	train_data_user = []
	train_data_item = []
	train_data_y = []
	train_user_index = []
	train_item_index = []
	index_ii = 0
	for user in user_item_list:  ### user_item_list is a dict
		item_list = user_item_list[user]    
		
		for item in item_list:   #### item_list is a list
			# train_user_index.append(user)
			# train_item_index.append(item)
			user_list = item_user_list[item]
			rate = user_item_rate[(user, item)]

			train_data_user.append(item_list)
			train_data_item.append(user_list)
			train_data_y.append(rate)				
			train_user_index.append(user)
			train_item_index.append(item)
	



	B = np.zeros([num_users, binary_len])    #### init the B Matrix(user binary)
	D = np.zeros([num_items, binary_len])    #### init the D matrix(item binary)
	# P = np.zeros([num_users, binary_len])    #### init the E Matrix(user binary)
	# Q = np.zeros([num_items, binary_len])    #### init the F matrix(item binary)
	E = np.random.uniform(0, 1, size = [num_users, binary_len])    #### init the E (for user)
	F = np.random.uniform(0, 1, size = [num_items, binary_len])    #### init the F (for item)
	P = np.random.uniform(-1.0, 1.0, size = [num_users, binary_len])    #### init the P (for user)
	Q = np.random.uniform(-1.0, 1.0, size = [num_items, binary_len])    #### init the Q (for item)
	mu = 1e-3   ##### init the paprameter mu
	train_batches = int(len(train_data_y)/batch_size)
	print ("##There are %s batches"%train_batches)
	args_trained = {}
	args_trained["B"] = B
	args_trained["D"] = D
	args_trained["E"] = E
	args_trained["F"] = F
	args_trained["P"] = P
	args_trained["Q"] = Q
	args_trained["mu"] = mu
	with tf.Session(config=config_gpu) as sess: #config=config_gpu
	 
		sess.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state('./'+exp)
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successful restore from previous models")
		else:
			print ("There is no pretrained model")
		summary_writer_train = tf.summary.FileWriter('./'+exp+'/train',graph=sess.graph)
		summary_writer_test = tf.summary.FileWriter('./'+exp+'/test') 
		fw = open('./'+exp+'/log.txt','w')
		
		def train_process(k, args_trained, fw): 

			### the training LSTM of the frameworks;
			### Update the weight of the LSTM
			if k==0:
				ite = 1
			else:
				ite = 1
			time0 = time.time()
			for ii in range(ite):
				time1 = time.time()
				args_trained, train_Loss, train_J1, train_J2 =  Lstm_updates(args_trained)
				time2 = time.time()
				print("Epoches {0}| Train_loss: {1} | J1: {2} | J2: {3} in {4} seconds".format(k , np.mean(train_Loss), np.mean(train_J1), np.mean(train_J2), time2-time1)) 
				fw.write("Epoches {0}| Train_loss: {1} | J1: {2} | J2: {3}".format(k , np.mean(train_Loss), np.mean(train_J1), np.mean(train_J2)))
			time3 = time.time()


			args_trained = PQEF_updates(args_trained)


			time4 = time.time()
			print ("Time: LSTM %.3f, PQE %.3f"%(time3-time0, time4-time3))
			save_path = saver.save(sess, "./"+exp+"/model.ckpt") 
			return args_trained

		def Lstm_updates(args_trained): 
			train_steps = int(len(train_data_y)/batch_size)
			train_used = random.sample(range(train_steps), int(0.1*train_batches))
			train_Loss = []
			train_acc = [] 
			train_J1 = []
			train_J2 = []
			time1 = time.time()
			for step in range(train_steps): 
				time_lstm_s = time.time()
				batch_x_item_user = train_data_item[step*batch_size : (step+1)*batch_size]
				batch_x_user_item = train_data_user[step*batch_size : (step+1)*batch_size] 
				batch_user_index = train_user_index[step*batch_size : (step+1)*batch_size]
				batch_item_index = train_item_index[step*batch_size : (step+1)*batch_size]

				batch_y = train_data_y[step*batch_size : (step+1)*batch_size]
				batch_P = [args_trained["P"][p_i] for p_i in batch_user_index]
				batch_Q = [args_trained["Q"][q_j] for q_j in batch_item_index]
				batch_E = [args_trained["E"][e_i] for e_i in batch_user_index]
				batch_F = [args_trained["F"][f_j] for f_j in batch_item_index]
				feed_dict = {model.x_user_item: batch_x_user_item, model.x_item_user: batch_x_item_user, model.y: batch_y,
							 model.E: batch_E, model.F: batch_F, model.P: batch_P, model.Q: batch_Q, model.mu: args_trained["mu"]}
				_, loss, batch_B, batch_D, J1, J2= sess.run([train_step, model.loss_lstm, model.B_i, model.D_j, model.J1, model.J2], feed_dict=feed_dict)


				# B_u, B_s, B_v = np.linalg.svd(batch_B, full_matrices = False)
				# B = np.array(B_u.dot(B_v.T)>=0, dtype=float)-0.5
				batch_P = np.sign(batch_B)
				# D_u, D_s, D_v = np.linalg.svd(batch_D, full_matrices = False)
				# D = np.array(D_u.dot(D_v.T)>=0, dtype=float)-0.5
				batch_Q = np.sign(batch_D)
				train_Loss.append(loss)
				train_J1.append(J1)
				train_J2.append(J2)
				 
				### update the B and D
				for kk in range(batch_size):
					user_index = batch_user_index[kk]
					item_index = batch_item_index[kk]
					args_trained["B"][user_index] = batch_B[kk]
					args_trained["D"][item_index] = batch_D[kk]
					args_trained["P"][user_index] = batch_P[kk]
					args_trained["Q"][item_index] = batch_Q[kk]
				time_lstm_e = time.time()
				# if step%1000==0:
				#	print (step, J1, J2, time_lstm_e-time_lstm_s)
			return args_trained, train_Loss, train_J1, train_J2
		def PQEF_updates(args_trained):
			# Update P 
			_B = args_trained["B"]
			_E = args_trained["E"]
			_P = args_trained["P"]
			_Q = args_trained["Q"]
			_D = args_trained["D"]
			_F = args_trained["F"] 



			_E_new = _E + args_trained["mu"]*(_P - _B)
			_F_new = _F + args_trained["mu"]*(_Q - _D)
			mu = min([rho*args_trained["mu"], mu_max])
			args_trained["E"] = _E_new
			args_trained["F"] = _F_new
			args_trained["mu"] = mu
			# time5 = time.time()
			return args_trained
		def obj_eva(args_trained):
			_B = args_trained["B"]
			_D = args_trained["D"]
			_P = args_trained["P"]
			_Q = args_trained["Q"]
			obj1 = np.sum(np.abs(np.sum(_P, axis=1)))
			obj2 = np.sum(np.abs(np.sum(_Q, axis=1)))
			# obj1 = np.linalg.norm((_B-_P),'fro')
			# obj2 = np.linalg.norm((_D-_Q),'fro')
			return obj2+obj1
		epos = 2000

		args_trained = PQEF_updates(args_trained)
		for i in range(epos):
			args_trained = train_process(i, args_trained, fw) 

		
if __name__ == "__main__":
	gpu_num = 0#sys.argv[1]
	exp = 'exp1' #sys.argv[2] #'exp1'
	percent = 0.97#float(sys.argv[3])
	main(gpu_num, exp, percent)
