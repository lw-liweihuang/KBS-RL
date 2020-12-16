# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:40:18 2017

@author: fms
"""
import logging
import random
import sys
from multiprocessing.pool import Pool

import tensorflow as tf
import numpy as np
import time
import heapq

from InteractiveModel import *
import pickle
from tqdm import tqdm
import os


def inter_diversity(lists):
    user_nb = len(lists)
    sumH = 0.0
    print (user_nb)
    for i in range(user_nb):
        for j in range(user_nb):
            if i==j:
                continue

            H_ij = 1.0 - float(len(set(lists[i]) & set(lists[j]))) / len(lists[i])
            sumH += H_ij

    Hu = sumH / (user_nb * (user_nb - 1))
    return Hu


def read_data(file):
    f = open(file, 'r')
    users = {}
    user_dict = {}
    item_dict = {}

    user_count = 0
    item_count = 1
    for line in f:
        data = line.split('::')
        user = data[0]
        item = data[1]
        rating = data[2]
        time_stmp = data[3][:-1]
        if int(user) not in user_dict:
            user_dict[int(user)] = user_count
            user_count += 1
        if int(item) not in item_dict:
            item_dict[int(item)] = item_count
            item_count += 1

        user = user_dict[int(user)]
        item = item_dict[int(item)]

        if int(user) not in users:
            users[int(user)] = []

        users[int(user)].append((int(user), int(item), float(rating), int(time_stmp)))

    f.close()
    new_users = {}
    user_historicals = {}

    for user in users:
        new_users[user] = sorted(users[user], key=lambda a: a[-1])
        user_historicals[user] = [d[1] for d in new_users[user]]

    return user_historicals, user_count, item_count




def gen_1m_train_test(path,fold):
    user_historicals, user_count, item_count = read_data(path)

    ftr = open('1m_train_user_'+str(fold), 'wb')
    fte = open('1m_test_user_'+str(fold), 'wb')
    train_users = []
    test_users = []
    for user in user_historicals:

        if random.random() < 0.9:
            train_users.append(user)
        else:
            test_users.append(user)

    pickle.dump(train_users, ftr)
    ftr.close()
    pickle.dump(test_users, fte)
    fte.close()

def gen_100k_train_test(path,fold):
    user_historicals, user_count, item_count = read_data(path)

    ftr = open('train_user_'+str(fold), 'wb')
    fte = open('test_user_'+str(fold), 'wb')
    train_users = []
    test_users = []
    for user in user_historicals:

        if random.random() < 0.9:
            train_users.append(user)
        else:
            test_users.append(user)

    pickle.dump(train_users, ftr)
    ftr.close()
    pickle.dump(test_users, fte)
    fte.close()



def compute_ndcg(labels, true_labels):
    dcg_labels = np.array(labels)
    dcg = np.sum(dcg_labels / np.log2(np.arange(2, dcg_labels.size + 2)))

    idcg_labels = np.array(true_labels)
    idcg = np.sum(idcg_labels / np.log2(np.arange(2, idcg_labels.size + 2)))
    if not idcg:
        return 0.

    return dcg / idcg


def get_topk(action, k):
    selection = np.argsort(action)[::-1][:k]
    return selection


def warm_train(file, test_num, warm_split, k):

    user_history, user_size, item_size = read_data(file)

    if file == '1m':
        with open('1m_train_user_' + test_num, 'rb') as f:
            train_users = pickle.load(f)
        with open('1m_test_user_' + test_num, 'rb') as f:
            test_users = pickle.load(f)
    elif file == '100k':
        with open('train_user_' + test_num, 'rb') as f:
            train_users = pickle.load(f)
        with open('test_user_' + test_num, 'rb') as f:
            test_users = pickle.load(f)


    config = tf.ConfigProto(allow_soft_placement=True, device_count={"CPU": 4}, 
                            inter_op_parallelism_threads=4,
                            intra_op_parallelism_threads=4,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True


    sess = tf.Session(config=config)
    initNMF = None
    lr = 0.001 
    rnn_size = 100
    layer_size = 1
    embedding_dim = 100
    nb_epcoch = 100
    slice_length = 20

    swith_epoch = 4 


    print("file:%s, test_num:%s, k:%s, warm_split:%s" % (file, test_num, str(k), warm_split))
    print ('train_users: ', len(train_users))
    print ('Totall users: ', len(test_users)+len(train_users))
    print ('avg length:', np.mean([len(user_history[u]) for u in train_users]))
    print ('item_num ', item_size)

    im = EMInteractiveModel(sess, rnn_size, layer_size, item_size, embedding_dim, k, lr)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())


    for epoch in range(nb_epcoch):
        train_mean_user_hit = []

        
        for j, user in enumerate(train_users):
            final_state = None
            user_seleceted_items = user_history[user]
            inference_length = len(user_seleceted_items)
            overall_length = inference_length

            warm_slice = int(overall_length * warm_split)
            warm_items = user_seleceted_items[:warm_slice]
            user_seleceted_items = user_seleceted_items  
            overall_length = len(user_seleceted_items)

            start_slice = 0
            end_slice = start_slice + slice_length

            mem = get_cum_interesting(warm_items, item_size)
            interest = get_cum_interesting(user_seleceted_items, item_size)
            masking = get_initial_masking(item_size)


            current_hit = []
            s_token = 0
            s_hit = 1.0

            while end_slice <= overall_length and start_slice < overall_length:
                slice_items = user_seleceted_items[start_slice: end_slice]
                inference_length = len(slice_items)
                if inference_length == 0:
                    break
                if epoch > swith_epoch:
                    _, rein, train_hit, final_state, masking, samples = im.reinforcement_learn(interest, masking,
                                                                                               mem,
                                                                                               inference_length,
                                                                                               final_state, s_token,
                                                                                               s_hit)
                else:
                    _, sup, train_hit, final_state, masking, samples = im.supervised_learn(interest, masking, mem,
                                                                                           slice_items,
                                                                                           inference_length,
                                                                                           final_state, s_token,
                                                                                           s_hit)

                start_slice = end_slice
                end_slice = start_slice + slice_length
                samples = samples.reshape(-1)
                train_hit = train_hit.reshape(-1)
                s_token = samples[-1]
                s_hit = -1.0

                if train_hit[-1] > 0:
                    s_hit = 1.0
                current_hit.extend(train_hit)
            train_mean_user_hit.append(np.sum(current_hit) / overall_length)

        test_mean_user_hit = []
        test_user_ndcg = []
        test_inter_diversity_list = []

        max_common_length = min([ len(user_history[u]) - int(len(user_history[u]) * warm_split) for u in test_users])        

        for user in test_users:
            final_state = None
            user_seleceted_items = user_history[user]
            inference_length = len(user_seleceted_items)

            warm_slice = int(inference_length * warm_split)
            warm_items = user_seleceted_items[:warm_slice]

            user_seleceted_items = user_seleceted_items[warm_slice:]
            inference_length = len(user_seleceted_items)

            mem = get_cum_interesting(warm_items, item_size)


            interest = get_cum_interesting(user_seleceted_items, item_size)
            masking = get_masking(item_size, warm_items) 

            s_token = 0
            (user_item_probs, samples, user_final_masking, user_hit, user_imediate_reward,
             user_cumsum_reward) = im.inference(interest, masking, mem,
                                                inference_length, final_state, s_t=s_token)

            user_item_probs = np.squeeze(user_item_probs, axis=0)  

            current_ndcg = []

            current_masking = np.reshape(masking, (item_size,))
            user_hit = np.reshape(user_hit, (inference_length,))

            user_item_probs = np.reshape(user_item_probs, (inference_length, item_size))
            user_selected_items = np.reshape(samples, (inference_length,))

            interest_item_num = len(user_seleceted_items)
            inference_length = inference_length
            for s in range(inference_length):

                if s == (max_common_length-1):
                    prob = user_item_probs[s] * current_masking
                    topk_items = get_topk(prob, k).tolist()
                    test_inter_diversity_list.append(topk_items)
                    
                true_labels = [0.0 for _ in range(k)]
                for ii in range(interest_item_num):
                    if ii >= k:
                        break
                    true_labels[ii] = 1.0

                ndcg_labels = []

                if user_hit[s] == 0.0:
                    current_ndcg.append(0.0)
                else:
                    prob = user_item_probs[s] * current_masking
                    topk_items = get_topk(prob, k)
                    for j in topk_items:
                        ndcg_label = 0.0
                        if j in user_seleceted_items:
                            ndcg_label = 1.0
                        ndcg_labels.append(ndcg_label)

                    current_ndcg.append(compute_ndcg(ndcg_labels, true_labels))

                    current_selected_item = user_selected_items[s]

                    current_masking[current_selected_item] = 0.0
                    interest_item_num -= 1

            test_user_ndcg.append(np.mean(current_ndcg))

            user_hit = user_hit.reshape(-1)
            samples = samples.reshape(-1)

            current_hit_ratio = np.sum(user_hit) / inference_length
            test_mean_user_hit.append(current_hit_ratio)

        train_hit_mean = float(np.mean(train_mean_user_hit))
        print("file:%s, test_num:%s, k:%s, warm_split:%s" % (file, test_num, str(k), warm_split))
        test_hit_mean = float(np.mean(test_mean_user_hit))
        test_ndcg_mean = float(np.mean(test_user_ndcg))
        diversity =  inter_diversity(test_inter_diversity_list)

        print('epoch:%d, train hr:%.4f, test: HR = %.4f, NDCG@10 = %.4f, diversity = %.4f' % (epoch, train_hit_mean, test_hit_mean, test_ndcg_mean, diversity))




if __name__ == "__main__":

    file = '100k'
    k = 10  

    warm_splits = 0.5

    for i in range(5):

        test_num = str(i) 
        gen_100k_train_test(file, test_num)
        k = 10
        warm_split = 0.0
        warm_train(file, test_num, warm_split, k)



