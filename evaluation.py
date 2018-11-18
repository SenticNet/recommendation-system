import math 
import numpy as np 
import random
_model = None
_batch_size = None 
_sess = None
_binary_len = None


 
def evaluate_admm(evaluation_set, B, D):
	hit_5 = []
	ndcg_5 = [] 
	hit_10 = []
	ndcg_10 = []
	hit_15 = []
	ndcg_15 = []
	hit_20 = []
	ndcg_20 = []
	for user in evaluation_set:  ## evaluation_set is a dict, key is user and item is negative items
		B_i = B[user]
		negative_item_for_evaluation = evaluation_set[user]
 
		sim = {}
		# index = 0
		# tg_index = random.randint(0,100)
		tg_item = negative_item_for_evaluation[0]
		for item in negative_item_for_evaluation:
			D_j = D[item] 
			# if index == 0:
			#	sim[tg_index] = similarity(B_i, D_j)
			# elif index == tg_index:
			#	sim[0] = similarity(B_i, D_j)
			# else:
			sim[item] = similarity(B_i, D_j)
			# index += 1

		hit, ndcg = get_scores(sim, 5, tg_item)
		hit_5.append(hit)
		ndcg_5.append(ndcg)
		hit, ndcg = get_scores(sim, 10, tg_item)
		hit_10.append(hit)
		ndcg_10.append(ndcg)
		hit, ndcg = get_scores(sim, 15, tg_item)
		hit_15.append(hit)
		ndcg_15.append(ndcg)
		hit, ndcg = get_scores(sim, 20, tg_item)
		hit_20.append(hit)
		ndcg_20.append(ndcg)
	return (hit_5, ndcg_5, hit_10, ndcg_10, hit_15, ndcg_15, hit_20, ndcg_20)
 


def get_scores(score, topk, tg_item):
	# sorted_score = sorted(score.items(), key= lambda x,y:cmp(x[1], y[1]), reverse=True) #descendent if reverse is true 
	sorted_score = sorted(score.items(), key = lambda x:x[1], reverse=True)
	hit = 0
	ndcg = 0
	for index in range(topk):
		score_pair = sorted_score[index] ## socre pair is a tuple with index as key and score as value
			# print ("index is %s, and pair is %s"%(index,score_pair))
		if score_pair[0]==tg_item:
			repeat_num = sum([tu[1]==score_pair[1] for tu in sorted_score[:10]])
			repeat_num_all = sum([tu[1]==score_pair[1] for tu in sorted_score])
			if repeat_num_all>=90:
				hit = 0
				ndcg = 0
			elif repeat_num==10 and repeat_num_all>=30 and repeat_num_all<90:
				hit = 1.0#/(repeat_num*1.0)
							# ndcg = 1.0/(np.log2(index+1)+1)
				ndcg = math.log(2)#/math.log(index+2)/repeat_num
				# hit_repeat = hit/repeat_num
			else:
				hit = 1.0
				ndcg = math.log(2)/math.log(index+2)

	return (hit, ndcg)
 

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))  
def similarity(s1, s2):
	assert len(s1) == len(s2)
	return np.sum(np.array(s1)*np.array(s2))/(2.0*len(s1))+0.5

