# -*- coding:  UTF-8 -*-
'''
Created on 2017.12.28
@author: anonymous
'''
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import ops

# Operator
import operator
import sys
import gc
import re
# listdir
from os import listdir
from compiler.ast import flatten
from fileinput import filename
import math
import matplotlib
import matplotlib.pyplot as plt
import os

# Pretraing via autoencoder
#----GOWALLA
#add PAD
# Parameters
it_learning_rate=tf.placeholder("float")
n_hidden = 500
batch_size =64 #batch
n_input=250
iters =100
keep_prob = tf.placeholder("float")
#-----------------
input_x = tf.placeholder(dtype=tf.int32)
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
is_train = True  # TRUE
table_X ={}
new_table_X={}
voc_tra=list()
table_Y={}
table_U={}
total_T = list()
total_U = list()
total_seqlens = list()  #

def extract_character_vocab(total_T):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set(flatten(total_T)))
    set_words = sorted(set_words)
    set_words = [str(item) for item in set_words]
    print len(set_words)
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def extract_words_vocab():
    print 'dictionary length',len(voc_tra)
    int_to_vocab={idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def getPvector(i):  #Embedding tensor
    return table_X[i]

def getXs():  #
    fpointvec = open('data/gowalla_user_vector250d_.dat', 'r')  
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()

        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  
            # if (float(i) > 1.0 or float(i) < -1.0):
            #     print "Error", i
        if lineArr[0] == '</s>':
            table_X['<PAD>']=X  #dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] =X

    print "point number item=", item
    return table_X


def get_index(userT):
    userT = list(set(userT))
    User_List = userT
    # print userT
    return User_List


def get_mask_index(value, User_List):
    #     print User_List #
    return User_List.index(value)


def get_true_index(index, User_List):
    return User_List[index]


def read_y():
    friendlist = open('out_data/over_edges.txt', 'r')
    for line in friendlist.readlines():
        lineArr = line.split()
        line_y = list()
        for i in lineArr[1:]:
            line_y.append(int(i))
        table_Y[int(lineArr[0])] = line_y
    print 'User number is ', len(table_Y)
    return table_Y
def unkown_tra():
    unkonw_userlist = list()
    funknown = open('out_data/maybe_edges.txt', 'r')
    for line in funknown.readlines():
        lineArr = line.split()
        line_u = list()
        for i in lineArr[1:]:
            line_u.append(int(i))
        table_U[int(lineArr[0])] = line_u
    print 'table_U', len(table_U)
    return table_U

def readtraindata():
    test_T = list()
    test_UserT = list()
    test_lens = list()  # gowalla_scopus_1104.dat
    ftraindata = open('data/gowalla_scopus_1104.dat',
                      'r')  
    tempT=list()  
    pointT = list() 
    userT = list()  
    seqlens = list()  
    item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i)) #chanage to string or char type
        tempT.append(X)
        userT.append(int(X[0]))
        pointT.append(X[1:])
        seqlens.append(len(X) - 1)  #data
        item += 1
    # Test 98481
    Train_Size = 20107
    pointT = pointT[:Train_Size]  # all tra
    userT = userT[:Train_Size]  # all user
    seqlens = seqlens[:Train_Size]  # all length
    total_T = pointT
    total_U = userT
    total_seqlens = seqlens

    User_List = get_index(userT)
    # print User_List
    # print "Index numbers",len(User_List)
    # print "point T",pointT[Train_Size-1]

    # choose trajectory via remaining unknown users
    all_T = list()
    TRA_ALL=list()
    all_U_List = sorted(list(table_U.keys()))
    for i in range(len(tempT)):
        if (int(tempT[i][0]) in all_U_List): #INT TYPE
            all_T.append(tempT[i][1:])
            TRA_ALL.append(int(tempT[i][0])) #ID
    print 'UNKNOWN ----->', len(all_T)

    flag = 0
    count = 0;
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  #
    rate = 0.5
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):
            User += 1
            # split 
            if (count > 1):  #
                # print "count",count," ",index
                test_T += (pointT[int((index - math.ceil(count * rate))):index])  # test
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])  # test
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])  # test data
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1;  # reset
            flag = userT[index]  # update
        else:
            count += 1

    pointT = temp_pointT
    userT = temp_userY
    seqlens = temp_seqlens
    print 'Total Numbers=', item - 1
    print 'train trajectories number=', Train_Size
    print 'Train Size=', len(pointT), ' Test Size=', len(test_T), "User numbers=", len(userT)
    #print test_T[-1]
    return TRA_ALL,all_T, pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List, total_T, total_U, total_seqlens  # return parameters

#Encoder layer
def get_encoder_layer(encoder_input, keep_prob):
    with tf.name_scope("encoder"):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        encode_lstm = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0, state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        return outputs, states #[max_time,batch_size,n_hidden]

#Decoder layer
def get_decoder_layer(decoder_input, encode_state, keep_prob, is_train):
    with tf.variable_scope("decoder"):
        decode_lstm = tf.contrib.rnn.LSTMCell(n_hidden)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        output_layer = Dense(TOTAL_SIZE)
        if is_train:
            decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],
                                   1)  # add GO to the end
            decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
            #input_=tf.transpose(decoder_input,[1,0,2])
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                            sequence_length=target_sequence_length)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, training_helper, encode_state, output_layer)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                         impute_finished=True,
                                                         maximum_iterations=max_target_sequence_length)
        else:  #test  different type of decoder output
            copy = tf.tile(tf.constant([vocab_to_int['<GO>']]), [batch_size])
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dic_embeddings, copy, vocab_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, predicting_helper, encode_state)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                         maximum_iterations=max_target_sequence_length)

        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        #target = tf.concat([decoder_embed_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  #mask
        target=encoder_embed_input
        return output, decoder_input_,predicting_logits,training_logits,masks,target

####--------
getXs()
print 'import friend list'
read_y()
print 'unkonwn user lists'
unkown_tra()

print 'Read Training Data--------------------------'
TRA_ALL,all_T, pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List, total_T, total_U, total_seqlens = readtraindata()
#connect label trajectory and unlabel trajectroy to total_T
total_Ts=total_T+all_T
for i_ in range(len(total_Ts)):
    for j_ in range(len(total_Ts[i_])):
        new_table_X[total_Ts[i_][j_]]=table_X[total_Ts[i_][j_]]
#
new_table_X['<GO>']=table_X['<GO>']
new_table_X['<EOS>']=table_X['<EOS>']
new_table_X['<PAD>']=table_X['<PAD>']
for keys in new_table_X:
    voc_tra.append(keys)
print 'train trajectory size',len(pointT)
print 'test trajectory size',len(test_T)
print 'test unlabeled trajectory size',len(total_Ts)

#int_to_vocab, vocab_to_int = extract_character_vocab(total_Ts) #create a dictionary
int_to_vocab, vocab_to_int=extract_words_vocab()
print 'Dictionary Size is ',len(vocab_to_int)
# conver original data to id
new_total_T = list()
for i in range(len(total_Ts)):
    temp = list()
    for j in range(len(total_Ts[i])):
        #print total_Ts[i][j]
        temp.append(vocab_to_int[total_Ts[i][j]])
    new_total_T.append(temp)

# print vocab_to_int['<EOS>']
# print new_total_T[0]
#print 'Point Number', len(vocab_to_int)
TOTAL_SIZE = len(vocab_to_int)
###

## Embedding
# 1. Embedding
embedding_size = 250
target_vocab_size = len(vocab_to_int)
#EmBedding
#dic_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, embedding_size]))
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())

# Encoder

encode_outputs, encode_states = get_encoder_layer(decoder_embed_input, keep_prob)


#Decoder

decoder_output, decoder_input_,predicting_logits,training_logits,masks,target= get_decoder_layer(decoder_embed_input, encode_states, keep_prob, is_train)
cost = tf.contrib.seq2seq.sequence_loss(training_logits, target, masks)
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)

initial = tf.global_variables_initializer()
# if not is_pre:
#     ops.reset_default_graph()
#     sess=tf.InteractiveSession()
def get_batches(sources, batch_size, source_pad_int):
    for batch_i in range(0,len(new_total_T)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
def train():
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print 'name:', v.name
    saver = tf.train.Saver()  #
    with tf.Session() as sess:
        sess.run(initial)
        #saver.restore(sess, './out_data/gw_ae.pkt')
        TOTAL_LOSS = []
        TOTAL_ACC = []
        LEARN_RATE=[]
        initial_learning_rate=0.001
        learning_rate_len=0.000008
        for epoch in range(iters):
            LOSS = 0
            acc_count = 0
            print 'Epoch', epoch
            temp_acc = 0
            num=0
            step = 0  # Record Every Step of training
            initial_learning_rate -= learning_rate_len
            while step < len(new_total_T)//batch_size:  #len(new_total_T)//batch_size
                start_i = step * batch_size

                input_x = new_total_T[start_i:start_i + batch_size]
                #
                sources_batch=pad_sentence_batch(input_x,vocab_to_int['<PAD>'])
                encode_batch=eos_sentence_batch(input_x,vocab_to_int['<EOS>'])
                input_batch=pad_sentence_batch(encode_batch,vocab_to_int['<PAD>'])
                #
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source)+1)
                # print 'output',input_x
                length_x = len(input_x) + 1
                if(initial_learning_rate<=0):
                    initial_learning_rate=0.00001
                enc_,pred, targets, opt, loss = sess.run([encode_states,predicting_logits, target, optimizer, cost],
                                                    feed_dict={target_sequence_length: pad_source_lengths,encoder_embed_input:input_batch,
                                                              decoder_embed_input: sources_batch, it_learning_rate:initial_learning_rate,keep_prob: 0.5})
                LOSS += loss
                # print 'pred', pred
                # print 'input_x', input_x
                if (step % 100 == 0):
                    print 'show acc_count', temp_acc/(100*batch_size)
                    # print 'show pred target',pred,targets
                    temp_acc= 0
                    #print '---->',enc_
                    #pad = vocab_to_int["<PAD>"]
                for i in range(len(input_x)):
                    if((input_x[i]==pred[i][:len(input_x[i])]).all()):
                        #print 'OUT',input_x[i]
                        acc_count+=1
                        temp_acc+=1
                num+=batch_size
                # if ((targets[0] == pred[0]).all()):
                #     acc_count += 1
                #     temp_acc += 1
                step += 1
            print 'learning rate',initial_learning_rate
            print 'ACC_rate', acc_count / num
            print 'Iteration', epoch, 'LOSS', LOSS
            # save the LOSS and ACC
            LEARN_RATE.append(initial_learning_rate)
            TOTAL_LOSS.append(LOSS)
            TOTAL_ACC.append(acc_count / num)
            #             print 'pred',pred
            #             print 'input',targets
            print loss
            saver.save(sess, './out_data/gw_ae.pkt')  # save model
        if (os.path.exists("out_data/gw_loss_ae.txt")): os.remove('out_data/gw_loss_ae.txt')
        save_loss(LEARN_RATE,TOTAL_LOSS,TOTAL_ACC)
        draw_pic_unloss(TOTAL_LOSS)
        draw_pic_unacc(TOTAL_ACC)

def save_loss(LEARN_RATE,TOTAL_LOSS,TOTAL_ACC):
    fopen = open('out_data/gw_loss_ae.txt', 'w')
    for i in range(len(TOTAL_LOSS)):
        fopen.write('epoch\t' + str(i) +'\t' +str(LEARN_RATE[i])+'\t'+str(TOTAL_LOSS[i]) +'\t'+ str(TOTAL_ACC[i])+'\n')
    fopen.close()
#
def draw_pic_unloss(LOSS):
    font = {'family': 'Trajectory',
            'weight': 'bold',
            'size': 18
            }
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(LOSS) + 1, 1))
    plt.plot(train_axis, np.array(LOSS), "b--", label="label loss")
    plt.title("Trajectory AE_FUL Model")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.show()


def draw_pic_unacc(acc):
    font = {'family': 'Trajectory',
            'weight': 'bold',
            'size': 18
            }
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(acc) + 1, 1))
    plt.plot(train_axis, np.array(acc), "b--", label="pretraining acc")
    plt.title("Trajectory AE_FUL Model")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.show()


if __name__ == "__main__":
    print 'start'
    train()