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
from evaluation import evaluate_admm
import gram_schmidt
import time
from scipy import stats
import sys
user_list_len = 80 # it is 80 for sst data, 400 for imdb data.
item_list_len = 50
binary_len = 64
scale_r = binary_len/2
num_users = 36328  ## the volumns of the users for amazon
num_items = 30776  ## the volumns of the items for amazon
 
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

		self.P = tf.placeholder(tf.float32, [None, binary_len], name = "P")
		self.Q = tf.placeholder(tf.float32, [None, binary_len], name = "Q")
		self.mu = tf.placeholder(tf.float32, name = "mu")


		B_i = self.MF_user(self.x_user_item)
		D_j = self.MF_item(self.x_item_user)
	
		B_i = tf.nn.softsign(B_i)
		D_j = tf.nn.softsign(D_j)


		self.B_i = B_i
		self.D_j = D_j


		self.J1 =  (tf.nn.l2_loss(self.y - tf.reduce_sum(tf.multiply(B_i, D_j), reduction_indices = 1))) ## useful for ml

		self.J2 = ((self.mu/2.0)*((tf.nn.l2_loss(self.B_i - (self.P + (1.0/self.mu)*self.E))) + tf.nn.l2_loss(self.D_j - (self.Q + 1.0/self.mu*self.F))))

		self.loss_lstm = (self.J1 + self.J2)

		self.acc = self.loss_lstm



 

	def MF_user(self, x):

		user_embedding = tf.nn.embedding_lookup(self.embedding_item, x)
		
		user_embedding = tf.expand_dims(user_embedding, -1)
		user_embedding = tf.reshape(user_embedding,[batch_size, item_list_len*10])
		W_user = weight_variable([item_list_len*10,128]) 
		b_user = bias_variable([128])
		self.params.append(W_user)
		self.params.append(b_user)
		self.l2_loss += tf.nn.l2_loss(W_user)
		out = tf.nn.tanh(tf.nn.xw_plus_b(user_embedding, W_user, b_user))


		W_user_out = weight_variable([128, binary_len]) 
		b_user_out = bias_variable([binary_len])
		self.params.append(W_user_out)
		self.params.append(b_user_out)
		self.l2_loss += tf.nn.l2_loss(W_user_out)
		out = tf.nn.tanh(tf.nn.xw_plus_b(out, W_user_out, b_user_out))


		W_user_linear = weight_variable([binary_len, binary_len]) 
		b_user_linear = bias_variable([binary_len])
		self.params.append(W_user_linear)
		self.params.append(b_user_linear)
		self.l2_loss += tf.nn.l2_loss(W_user_linear)
		out = (tf.nn.xw_plus_b(out, W_user_linear, b_user_linear))
		return out
	def MF_item(self, x):
		# embedding_init = tf.Variable(tf.random_uniform([num_users, 10], -1.0, 1.0), name='embedding_init_item', trainable=True)
		item_embedding = tf.nn.embedding_lookup(self.embedding_user, x)
		item_embedding = tf.reshape(item_embedding,[batch_size, user_list_len*10])
		W_item = weight_variable([user_list_len*10, 128]) 
		b_item = bias_variable([128])
		self.params.append(W_item)
		self.params.append(b_item)
		self.l2_loss += tf.nn.l2_loss(W_item)
		out = tf.nn.tanh(tf.nn.xw_plus_b(item_embedding, W_item, b_item))


		W_item_out = weight_variable([128, binary_len]) 
		b_item_out = bias_variable([binary_len])
		self.params.append(W_item_out)
		self.params.append(b_item_out)
		self.l2_loss += tf.nn.l2_loss(W_item_out)
		out = tf.nn.tanh(tf.nn.xw_plus_b(out, W_item_out, b_item_out))


		W_user_linear = weight_variable([binary_len, binary_len]) 
		b_user_linear = bias_variable([binary_len])
		self.params.append(W_user_linear)
		self.params.append(b_user_linear)
		self.l2_loss += tf.nn.l2_loss(W_user_linear)
		out = (tf.nn.xw_plus_b(out, W_user_linear, b_user_linear))


		return out



 
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


def p_value(obs):
	length = len(obs)
	exp = [1.]*length
	p = stats.chisquare(obs, f_exp = exp)[1]
	return float(p)



def main(gpu_num, exp, percent): 
	model = Abbre_Repair()
	params = model.params  
	train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss_lstm)  

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
			if item in item_user_list:
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
				args_trained, train_Loss, train_J1, train_J2 =  mlp_updates(args_trained)
				time2 = time.time()
				print("Epoches {0}| Train_loss: {1} | J1: {2} | J2: {3} in {4} seconds".format(k , np.mean(train_Loss), np.mean(train_J1), np.mean(train_J2), time2-time1)) 
				fw.write("Epoches {0}| Train_loss: {1} | J1: {2} | J2: {3}".format(k , np.mean(train_Loss), np.mean(train_J1), np.mean(train_J2)))
			time3 = time.time()

			args_trained = PQEF_updates(args_trained)

			time4 = time.time()
			print ("Time: LSTM %.3f, PQE %.3f"%(time3-time0, time4-time3))
			save_path = saver.save(sess, "./"+exp+"/model.ckpt") 
			return args_trained

		def mlp_updates(args_trained): 
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
				batch_B = np.sign(batch_B)
				# D_u, D_s, D_v = np.linalg.svd(batch_D, full_matrices = False)
				# D = np.array(D_u.dot(D_v.T)>=0, dtype=float)-0.5
				batch_D = np.sign(batch_D)
				train_Loss.append(loss)
				train_J1.append(J1)
				train_J2.append(J2)
				 
				### update the B and D
				for kk in range(batch_size):
					user_index = batch_user_index[kk]
					item_index = batch_item_index[kk]
					args_trained["B"][user_index] = batch_B[kk]
					args_trained["D"][item_index] = batch_D[kk]
				time_lstm_e = time.time()
				# if step%1000==0:
				#	print (step, J1, J2, time_lstm_e-time_lstm_s)
			return args_trained, train_Loss, train_J1, train_J2
		def PQEF_updates(args_trained):
			# Update P 
			_B = args_trained["B"]
			_E = args_trained["E"]
			T1 = _B - 1.0/args_trained["mu"]*_E
			# T1_marginal = np.mean(T1, axis=1)
			# T1_new = [T1[i]-T1_marginal[i] for i in range(len(T1))]
			T1_u, T1_s, T1_v = np.linalg.svd(T1, full_matrices = False)
			_T1 = T1_u.dot(T1_v.T)
			T1_new = np.insert(_T1, [0], np.ones([1, binary_len]), axis = 0)
			_P = np.sqrt(num_users)*gram_schmidt.gs(T1_new)[1:]
			# P_u, P_s, P_v = np.linalg.svd(_P, full_matrices = False)
			# _P = P_u.dot(P_v.T)
			args_trained["P"] = _P
			# time3 = time.time()
			# Update Q
			_D = args_trained["D"]
			_F = args_trained["F"]
			## build the T2
			T2 = _D - 1.0/args_trained["mu"]*_F
			# T2_marginal = np.mean(T2, axis=1)
			# T2_new = [T2[i]-T2_marginal[i] for i in range(len(T2))]
			## decomposition using svd
			T2_u, T2_s, T2_v = np.linalg.svd(T2, full_matrices = False)
			_T2 = T2_u.dot(T2_v.T)
			T2_new = np.insert(_T2, [0], np.ones([1, binary_len]), axis = 0)
			_Q = np.sqrt(num_items)*gram_schmidt.gs(T2_new)[1:]
			# Q_u, Q_s, Q_v = np.linalg.svd(_Q, full_matrices = False)
			# _Q = Q_u.dot(Q_v.T)
			args_trained["Q"] = _Q
			# time4 = time.time()
			# Update E, F, mu
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


		epos = 50

		args_trained = PQEF_updates(args_trained)
		for i in range(epos):
			args_trained = train_process(i, args_trained, fw) 

			if i%1==0:
				hit_5, ndcg_5, hit_10, ndcg_10, hit_15, ndcg_15, hit_20, ndcg_20 = evaluate_admm(evalueation_set, args_trained["B"], args_trained["D"])
				# print ("Hit5: %.4f, and ndcg5: %.4f Hit10: %.4f, and ndcg10: %.4f Hit15: %.4f, and ndcg15: %.4f Hit20: %.4f, and ndcg20: %.4f\n"%(np.mean(hit_5), np.mean(ndcg_5),np.mean(hit_10), np.mean(ndcg_10),np.mean(hit_15), np.mean(ndcg_15),np.mean(hit_20), np.mean(ndcg_20)))
				fw.write("Hit5: %.4f, and ndcg5: %.4f Hit10: %.4f, and ndcg10: %.4f Hit15: %.4f, and ndcg15: %.4f Hit20: %.4f, and ndcg20: %.4f\n"%(np.mean(hit_5), np.mean(ndcg_5),np.mean(hit_10), np.mean(ndcg_10),np.mean(hit_15), np.mean(ndcg_15),np.mean(hit_20), np.mean(ndcg_20)))

gpu_num = sys.argv[1]
exp = sys.argv[2] #'exp1'
percent = float(sys.argv[3])
main(gpu_num, exp, percent)
