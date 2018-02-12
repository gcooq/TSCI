# -*- coding:  UTF-8 -*-
'''
Created on 2017-12-28
@author: anonymous
'''
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
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

# This files is to pre-train the gowalla dataset bacause the location is much sparse.
#add PAD
# Parameters
n_hidden = 800
batch_size =16 #batch
n_input=250
n_classes=201
keep_prob = tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
#define for clssification
train_iters=20 #num for training
z_size=800
latentscale_iter=tf.placeholder(dtype=tf.float32)
#-----------------
input_x = tf.placeholder(dtype=tf.int32)
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

y_=tf.placeholder(dtype=tf.float32,shape=[batch_size,n_classes,2])
is_train = True  # TRUE
table_X ={}
new_table_X={}
voc_tra=list()
table_Y={}
table_U={}
total_T = list()
total_U = list()
total_seqlens = list()  #
#define the weight and bias dictionary
with tf.name_scope("decoder_inital"):
    weights_de={
        'w_':tf.Variable(tf.random_normal([z_size,n_hidden],mean=0.0, stddev=0.01))
    }
    biases_de = {
    'b_': tf.Variable(tf.random_normal([n_hidden], mean=0.0, stddev=0.01))
    }

with tf.name_scope("w_b"):
   weights={
         'out':tf.Variable(tf.random_normal([z_size+2*n_hidden,2*n_classes]))
    }
   biases={
        'out':tf.Variable(tf.random_normal([2*n_classes]))
    }
#-----------------

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
def getPvector(i):  # check Embedding tensor
    return table_X[i]


def getXs():  # read poi
    fpointvec = open('data/gowalla_user_vector250d_.dat', 'r')  #
    #     table_X={}  #
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()

        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  #
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  #
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
    #     print User_List #weikong
    return User_List.index(value)


def get_true_index(index, User_List):
    return User_List[index]


def read_y():
    friendlist = open('data/over_edges.txt', 'r')
    for line in friendlist.readlines():
        lineArr = line.split()
        line_y = list()
        for i in lineArr[1:]:
            line_y.append(int(i))
        table_Y[int(lineArr[0])] = line_y
    print 'User number is ', len(table_Y)
    return table_Y


def get_code(usertrue_id, User_List):
    y = [0] * n_classes
    y_ = [1, 0] * n_classes
    # print len(y_)
    y_list = table_Y[usertrue_id]

    for i in y_list:
        MASK = get_mask_index(i, User_List)  # mask_id
        # print MASK
        y_[MASK * 2] = 0  # meaning friend
        y_[MASK * 2 + 1] = 1
        y[MASK] = 1
    y_ = np.reshape(y_, [n_classes, 2])
    return y_, y


def get_code_un(list, User_List):
    y = [0] * n_classes
    y_ = [1, 0] * n_classes
    for i in list:
        MASK = get_mask_index(i, User_List)  # mask_id
        # print MASK
        y_[MASK * 2] = 0  # friend
        y_[MASK * 2 + 1] = 1
        y[MASK] = 1
    y_ = np.reshape(y_, [n_classes, 2])
    return y_, y


def unkown_tra():
    unkonw_userlist = list()
    funknown = open('data/maybe_edges.txt', 'r')
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
                      'r')  # gowalla_scopus_1006.dat
    tempT=list()  #all data
    pointT = list()  # 
    userT = list()  # 
    seqlens = list()  #
    item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i)) #chanage to string or char type
        tempT.append(X)
        userT.append(int(X[0]))
        pointT.append(X[1:])
        seqlens.append(len(X) - 1)  #
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
            TRA_ALL.append(int(tempT[i][0])) #
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
            # split data
            if (count > 1):  #
                # print "count",count," ",index
                test_T += (pointT[int((index - math.ceil(count * rate))):index])  # 
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])  #
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])  #
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
    total_T = pointT + test_T
    total_U = userT + test_UserT
    print 'Total Numbers=', item - 1
    print 'train trajectories number=', len(total_T)
    print 'Train Size=', len(pointT), ' Test Size=', len(test_T), "User numbers=", len(User_List)
    return TRA_ALL,all_T, pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List, total_T, total_U, total_seqlens  #

#Encoder layer
def get_encoder_layer(encoder_input, keep_prob,reuse=False):
    with tf.variable_scope("encoder",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  #state_is_tuple=True
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # add dropout
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # state_is_tuple=True
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # add dropout
        # multi-layer
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_state,
          encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32, time_major=True,
                                                            )
        new_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        # encode_lstm = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        # encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        # (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        # # new_states=tf.concat(states,1) #[batch_size,2*n_hidden]
        # new_states = states[0]
        # print 'states-->',new_states
        new_states=encoder_fw_state[0][0] #c
       # print new_states
        #new_states=encoder_state_c
        o_mean = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                   scope="z_mean")  # relu tf.nn.sigmoid
        o_stddev = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                     scope="z_std")  # relu activation_fn=tf.nn.sigmoid
        #print o_mean
        return new_outputs, encoder_fw_state,o_mean,o_stddev,new_states #[max_time,batch_size,n_hidden]

#Decoder layer
def get_decoder_layer(vae_z,decoder_embed_input,encode_state,keep_prob, is_train):
    with tf.variable_scope("decoder"):
        decode_lstm = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0, state_is_tuple=True)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)

        decoder_initial_state=encode_state
        output_layer = Dense(TOTAL_SIZE)
        # initial z
        if is_train:
            decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],
                                   1)  # add GO to the end
            decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
            #input_=tf.transpose(decoder_input,[1,0,2])
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                            sequence_length=target_sequence_length)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, training_helper, decoder_initial_state, output_layer)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                         impute_finished=True,
                                                         maximum_iterations=max_target_sequence_length)
        else:  # test different type of decoder output
            copy = tf.tile(tf.constant([vocab_to_int['<GO>']]), [batch_size])
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dic_embeddings, copy, vocab_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, predicting_helper, decoder_initial_state)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                         maximum_iterations=max_target_sequence_length)

        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        #target = tf.concat([target_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  #mask
        target = decoder_embed_input
        return output,predicting_logits,training_logits,masks,target
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
TOTAL_SIZE = len(vocab_to_int)
###
##
new_pointT = list()
for i in range(len(pointT)):
    temp = list()
    for j in range(len(pointT[i])):
        #print total_Ts[i][j]
        temp.append(vocab_to_int[pointT[i][j]])
    new_pointT.append(temp)

new_testT = list()
for i in range(len(test_T)):
    temp = list()
    for j in range(len(test_T[i])):
        #print total_Ts[i][j]
        temp.append(vocab_to_int[test_T[i][j]])
    new_testT.append(temp)

new_unT = list()
for i in range(len(all_T)):
    temp = list()
    for j in range(len(all_T[i])):
        #print total_Ts[i][j]
        temp.append(vocab_to_int[all_T[i][j]])
    new_unT.append(temp)
##
embedding_size = 250
target_vocab_size = len(vocab_to_int)
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())




# Encoder
encode_outputs, encode_states,z_mean,z_stddev,new_states= get_encoder_layer(encoder_embed_input, keep_prob)
#VAE

samples=tf.random_normal(tf.shape(z_stddev))
z=z_mean+tf.exp(z_stddev*0.5)*samples
#Decoder
# inital state vae_z
h_state =tf.nn.softplus(tf.matmul(z, weights_de['w_']) + biases_de['b_'])  # tf.nn.relu

decoder_initial_state = LSTMStateTuple(h_state, encode_states[0][1])

decoder_output, predicting_logits, training_logits, masks, target = get_decoder_layer(z,
                                                                                                          decoder_embed_input,
                                                                                                          decoder_initial_state,
                                                                                                          keep_prob,
                                                                                                          is_train)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1) #KL_
#variable

latent_cost=tf.reduce_mean(latent_loss)
laten_=latentscale_iter* tf.reduce_mean(latent_loss)
latent_space=tf.reduce_mean(encode_outputs,axis=0) #
#classifier
y_lantent=tf.concat([latent_space,z],1)
pred_y= tf.matmul(y_lantent, weights['out']) + biases['out']
pred_ = tf.reshape(pred_y, [batch_size,n_classes, 2])
cost_classifier= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred_))
encropy_loss=tf.contrib.seq2seq.sequence_loss(training_logits, target, masks)
cost_vae = tf.reduce_mean(
        tf.contrib.seq2seq.sequence_loss(training_logits, target, masks) + latentscale_iter * (latent_loss))
cost=cost_classifier+0.1*cost_vae
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)
pred_top = tf.arg_max(pred_, 2)  # a list of result

def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
initial = tf.global_variables_initializer()
def train_tuf():
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print 'name:', v.name
    variables1=all_vars[:2]
    variables2=all_vars[4:]
    variables=variables1+variables2
    saver=tf.train.Saver(variables) #variables
    savers=tf.train.Saver() #variables
    with tf.Session() as sess:
        sess.run(initial)
        saver.restore(sess, './out_data/GW_VAE.pkt')
        #savers.restore(sess, './temp/GW_VAE_S.pkt')
        print'Read train & test data'

        TOTAL_LOSS=[]
        CR_LOSS=[]
        EN_LOSS=[]
        KL_LOSS=[]
        VAEs_LOSS=[]
        train_ACC=[]
        pred_ACC=[]
        unpred_ACC=[]
        F1_AVG=[]
        Test_1_F1=[]
        Test_2_F1=[]
        TRAIN_RECORED=[]
        RECOVERY_RECORED=[]
        RECOMMEND_RECORED=[]

        LEARN_RATE=[]
        lens = 0.000008
        initial_learning_rate=0.0008
        learning_rate_len=0.000008
        min_kl=1.0         #0.158
        min_kl_epoch=min_kl
        #sort
        index_T={}
        new_trainT=[]
        new_trainU=[]
        for i in range(len(new_pointT)):
            index_T[i]=len(new_pointT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id=temp_size[i][0]
            new_trainT.append(new_pointT[id])
            new_trainU.append(userT[id])

        for epoch in range(train_iters): #train_iters
            #define inital vaue
            step = 0  # Record Every Step of training
            num=0
            acc_count = 0
            temp_acc=0
            LOSS = 0
            gen_LOSS = 0
            en_Loss = 0
            C_LOSS=0
            VAE_LOSS=0
            VALUE=[]
            accuracy = []
            FULL=[]
            #initial_learning_rate -= learning_rate_len
            while step < len(new_trainT)//batch_size:
                start_i = step * batch_size
                input_x = new_trainT[start_i:start_i + batch_size]
                #
                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                encode_batch=eos_sentence_batch(input_x,vocab_to_int['<EOS>'])
                input_batch=pad_sentence_batch(encode_batch,vocab_to_int['<PAD>'])
                #
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source) + 1)
                # print 'output',input_x
                batch_t_y = []
                batch_mask_y = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step, y_mask = get_code(new_trainU[y_i], User_List)  # ,class_optimizer  , train_optimizer
                    batch_t_y.append(xsy_step)
                    batch_mask_y.append(y_mask)
                length_x = len(input_x) + 1
                if min_kl_epoch<1.0:
                    min_kl_epoch = min_kl + step* lens
                if (initial_learning_rate <= 0):
                    initial_learning_rate = 0.000008
                pred_out,encropy_cost,loss_classifier,kl_loss,pred, loss_vae,opt,loss = sess.run([pred_top,encropy_loss,cost_classifier,latent_loss,predicting_logits, cost_vae, optimizer,cost],
                                                    feed_dict={target_sequence_length: pad_source_lengths,
                                                             decoder_embed_input: input_batch, y_:batch_t_y,encoder_embed_input:sources_batch,keep_prob: 0.5,latentscale_iter:min_kl_epoch,it_learning_rate:initial_learning_rate})

                value, full, acc = acc_compute(pred_out, batch_mask_y)
                VALUE.append(value)

                FULL.append(full)
                accuracy.append(acc)
                LOSS += loss
                C_LOSS+=loss_classifier
                en_Loss+=encropy_cost
                gen_LOSS+=np.mean(kl_loss)
                VAE_LOSS+=loss_vae

                # if (step % 1000 == 0):
                #     print 'show acc_count', temp_acc/(1000*batch_size)
                #     print 'min_kl_epoch', min_kl_epoch
                #     temp_acc= 0
                #     print'total_cost',loss,'encropy_cost:',encropy_cost,'laten_cost:',np.mean(vae_loss)
                #     print('test-----------------')
                #     #print 'pred', pred[0][:len(input_x[0])]
                #     # print 'states_x', new_states_[0][:15]
                #     print 'end'
                for i in range(len(input_x)):
                    if((input_x[i]==pred[i][:len(input_x[i])]).all()):
                        #print 'OUT',input_x[i]
                        acc_count+=1
                        temp_acc+=1
                step+= 1
                num+=batch_size
            #last batch<batch_size dealing
            sid=(step)*batch_size
            last_x = new_trainT[sid:]
            #print'--',last_x[-1]
            lens=len(last_x)
            lost_len=batch_size-lens

            input_x=last_x+new_trainT[:lost_len]
            #print len(input_x)
            #
            batch_t_y = []
            batch_mask_y = []
            for y_i in range((step) * batch_size, (step) * batch_size + lens):
                xsy_step, y_mask = get_code(new_trainU[y_i], User_List)  # ,class_optimizer  , train_optimizer
                batch_t_y.append(xsy_step)
                batch_mask_y.append(y_mask)
            for y_i in range(0, lost_len):
                xsy_step, y_mask = get_code(new_trainU[y_i], User_List)  # ,class_optimizer  , train_optimizer
                batch_t_y.append(xsy_step)
                batch_mask_y.append(y_mask)
            sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])

            encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
            input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
            #
            pad_source_lengths = []

            for source in input_x:
                pad_source_lengths.append(len(source) + 1)

            pred_out,encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
                [pred_top,encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
                feed_dict={target_sequence_length: pad_source_lengths,
                           decoder_embed_input: input_batch, encoder_embed_input: sources_batch, y_: batch_t_y,keep_prob: 0.5,
                           latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})
            value, full, acc = acc_compute(pred_out, batch_mask_y)
            VALUE.append(value)

            FULL.append(full)
            accuracy.append(acc)
            value_rate = np.mean(VALUE)
            full_rate = np.mean(FULL)
            avg_acc = np.mean(accuracy)
            macro_f1 = 2 * value_rate * full_rate / (value_rate + full_rate)
            print 'learning_rate', initial_learning_rate
            print 'Precision in train epoch', value_rate, 'Recall of train', full_rate, 'macro_f1 of train', macro_f1, 'Accuarcy of train', avg_acc

            print'TEST--------------\n'
            test_value, test_full, tuf_f1, tuf_acc = prediction_tuf(sess, new_testT, test_UserT, User_List)
            print 'Discovery Precision', test_value, 'Discovery Recall', test_full, 'tuf_accuracy', tuf_acc, 'tuf_f1', tuf_f1
            print '\n'
            # predict the Unkown Users
            print '\nTEST Unknown User Friendship'
            utest_value, utest_full, un_f1, un_acc = prediction_tuf_unkown(sess, new_unT, User_List, TRA_ALL)
            print 'Un Precision', utest_value, 'Un Recall', utest_full, 'macro-f1', un_f1, 'Un Accuracy', un_acc
            print 'ToTal Loss',LOSS,'Classifier Loss',C_LOSS,'VAE-Loss',VAE_LOSS,'KL Loss',gen_LOSS,'Entropy loss',en_Loss
            print '\n'
            TRAIN_RECORED.append([value_rate, full_rate, macro_f1, avg_acc])
            RECOVERY_RECORED.append([test_value, test_full, tuf_f1, tuf_acc])
            RECOMMEND_RECORED.append([utest_value, utest_full, un_f1, un_acc])
            LEARN_RATE.append(initial_learning_rate)
            F1_AVG.append(macro_f1)
            Test_1_F1.append(tuf_f1)
            Test_2_F1.append(un_f1)
            train_ACC.append(full_rate)  # BUG
            TOTAL_LOSS.append(LOSS)
            CR_LOSS.append(C_LOSS)
            VAEs_LOSS.append(VAE_LOSS)
            EN_LOSS.append(en_Loss)
            KL_LOSS.append(gen_LOSS)
            pred_ACC.append(test_full)  # BUG
            unpred_ACC.append(utest_full)  # BUG
            print 'epoch', epoch, 'ACC Train', acc_count / num
            savers.save(sess, './temp/GW_VAE_S.pkt')
            # draw pic
            # print TOTAL_LOSS
        draw_pic(TOTAL_LOSS)
        draw_pic_acc(train_ACC, pred_ACC, unpred_ACC)
        save_allloss(TOTAL_LOSS,CR_LOSS,VAEs_LOSS,EN_LOSS,KL_LOSS)
        save_metrics(LEARN_RATE, TRAIN_RECORED, RECOVERY_RECORED, RECOMMEND_RECORED)
        print 'END'
def save_metrics(LEARN_RATE,TRAIN_RECORED,RECOVERY_RECORED,RECOMMEND_RECORED):
    files=open('Result/GW_metric_VAE_S.txt','a+')
    files.write('epoch \t learning_rate \t train_P \t train_R \t train_F \t train_a \t recovery_P \t recovery_R \t recovery_F \t recovery_A \t recommend_P \t recommend_R \t recommend_F \t recommend_A\n')
    for i in range(len(TRAIN_RECORED)):
        files.write(str(i)+'\t')
        files.write(str(LEARN_RATE[i])+'\t')
        files.write(str(TRAIN_RECORED[i][0]) + '\t'+str(TRAIN_RECORED[i][1])+'\t'+str(TRAIN_RECORED[i][2])+'\t'+str(TRAIN_RECORED[i][3])+'\t')
        files.write(str(RECOVERY_RECORED[i][0]) + '\t' + str(RECOVERY_RECORED[i][1]) + '\t' + str(RECOVERY_RECORED[i][2]) + '\t'+str(RECOVERY_RECORED[i][3])+'\t')
        files.write(str(RECOMMEND_RECORED[i][0]) + '\t' + str(RECOMMEND_RECORED[i][1]) + '\t' + str(RECOMMEND_RECORED[i][2])+'\t'+ str(RECOMMEND_RECORED[i][3])+'\n')
    files.close()
def save_allloss(TOTAL_LOSS,CR_LOSS,VAEs_LOSS,EN_LOSS,KL_LOSS):
    fopen = open('Result/gw_loss_vae_s.txt', 'a+')
    fopen.write('epoch \t TOTAL_LOSS\t CR_LOSS \t VAEs_LOSS \t EN_LOSS\t KL_LOSS \n')
    for i in range(len(TOTAL_LOSS)):
        fopen.write('epoch\t' + str(i) +'\t' +str(TOTAL_LOSS[i])+'\t' +str(CR_LOSS[i])+'\t'+str(VAEs_LOSS[i]) +'\t'+ str(EN_LOSS[i])+'\t'+ str(KL_LOSS[i])+'\n')
    fopen.close()
#pic part
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

def acc_compute(pred,label): #
    batch_P=[]
    batch_R=[]
    batch_ACC=[]
    for step in range(len(pred)):
        Vs=0
        count=0
        length=0
        value_true=0
        step_pred=pred[step]
        step_label=label[step]
        for i in range(len(step_pred)):
            if(step_label[i]==1): #
                count+=1
            if (step_pred[i]==1): #
                length+=1
            if(step_pred[i]==1 and step_label[i]==1):  #TP 
                value_true+=1
            if (step_pred[i] == 1 or step_label[i] == 1):
                Vs += 1
        if(length==0):
            length=1 #
        batch_P.append(value_true/length)
        batch_R.append(value_true/count)
        batch_ACC.append(value_true/Vs)
    p=np.mean(batch_P)
    r=np.mean(batch_R)
    acc=np.mean(batch_ACC)
    return p,r,acc
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
def get_batches(sources, batch_size, source_pad_int):
    for batch_i in range(0,len(new_total_T)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def prediction_tuf(sess,testT,testU,User_List):
    step=0
    temp_acc=0
    acc_count=0
    num=0
    # sort
    index_T = {}
    new_testT = []
    new_testU = []
    Test_Value=[]
    Test_FULL=[]
    ACC=[]
    for i in range(len(testT)):
        index_T[i] = len(testT[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        new_testT.append(testT[id])
        new_testU.append(testU[id])

    while step<len(new_testT)//batch_size: #
        start_i = step * batch_size
        input_x = new_testT[start_i:start_i + batch_size]

        #
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        # record length
        pad_source_lengths = []
        for source in input_x:
            pad_source_lengths.append(len(source) + 1)
        batch_t_y = []
        batch_mask_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step, y_mask = get_code(new_testU[y_i], User_List)  # ,class_optimizer  , train_optimizer
            batch_t_y.append(xsy_step)
            batch_mask_y.append(y_mask)

        pred_out = sess.run(
            pred_top,
            feed_dict={target_sequence_length: pad_source_lengths,
                       decoder_embed_input: input_batch, y_:batch_t_y,encoder_embed_input: sources_batch, keep_prob: 1.0,
                       })
        value,full,acc=acc_compute(pred_out,batch_mask_y)
        Test_Value.append(value)
        Test_FULL.append(full)
        ACC.append(acc)
        step += 1
        num += batch_size


    # last batch<batch_size dealing
    sid = (step) * batch_size
    last_x = new_testT[sid:]
    lens = len(last_x)
    lost_len = batch_size - lens
    input_x = last_x + new_testT[:lost_len]
    batch_t_y = []
    batch_mask_y = []
    for y_i in range((step ) * batch_size, (step) * batch_size + lens):
        xsy_step, y_mask = get_code(new_testU[y_i], User_List)  # ,class_optimizer  , train_optimizer
        batch_t_y.append(xsy_step)
        batch_mask_y.append(y_mask)
    for y_i in range(0, lost_len):
        xsy_step, y_mask = get_code(new_testU[y_i], User_List)  # ,class_optimizer  , train_optimizer
        batch_t_y.append(xsy_step)
        batch_mask_y.append(y_mask)
    #
    sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
    encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
    input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
    #
    pad_source_lengths = []
    for source in input_x:
        pad_source_lengths.append(len(source) + 1)

    pred_out= sess.run(
        pred_top,
        feed_dict={target_sequence_length: pad_source_lengths,
                   decoder_embed_input: input_batch, y_:batch_t_y,encoder_embed_input: sources_batch, keep_prob: 1.0,
                   })
    value, full, acc = acc_compute(pred_out, batch_mask_y)
    Test_Value.append(value)
    Test_FULL.append(full)
    ACC.append(acc)
    #last batch
    accuracy=np.mean(ACC)
    value=np.mean(Test_Value)
    full=np.mean(Test_FULL)
    f1 = 2*value*full/(value+full)
    return value,full,f1,accuracy
def prediction_tuf_unkown(sess,all_T,User_List,TRA_ALL):
    step=0
    num=0
    # sort
    index_T = {}
    new_testT = []
    new_testU = []
    Test_Value=[]
    Test_FULL=[]
    ACC=[]
    for i in range(len(all_T)):
        index_T[i] = len(all_T[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        new_testT.append(all_T[id])
        new_testU.append(TRA_ALL[id])
    while step<len(new_testT)//batch_size:
        start_i = step * batch_size
        input_x = new_testT[start_i:start_i + batch_size]


        #
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        #
        pad_source_lengths = []
        batch_t_y = []
        batch_mask_y = []
        for y_i in range(start_i, start_i + batch_size):
            User = table_U[new_testU[y_i]]
            xsy_step, y_mask=get_code_un(User,User_List)  # ,class_optimizer  , train_optimizer
            batch_t_y.append(xsy_step)
            batch_mask_y.append(y_mask)

        for source in input_x:
            pad_source_lengths.append(len(source) + 1)

        pred_out = sess.run(pred_top,
            feed_dict={target_sequence_length: pad_source_lengths,
                       decoder_embed_input: input_batch,y_:batch_t_y, encoder_embed_input: sources_batch, keep_prob: 1.0,
                       })

        #print pred_out

        value,full,acc=acc_compute(pred_out,batch_mask_y)

        Test_Value.append(value)
        Test_FULL.append(full)
        ACC.append(acc)

        step += 1
        num+=batch_size

    # last batch<batch_size dealing
    sid = (step) * batch_size
    last_x = new_testT[sid:]
    lens = len(last_x)
    lost_len = batch_size - lens
    input_x = last_x + new_testT[:lost_len]
    batch_t_y = []
    batch_mask_y = []
    for y_i in range((step ) * batch_size, (step) * batch_size + lens):
        User = table_U[new_testU[y_i]]
        xsy_step, y_mask = get_code_un(User, User_List)  # ,class_optimizer  , train_optimizer
        batch_t_y.append(xsy_step)
        batch_mask_y.append(y_mask)
    for y_i in range(0, lost_len):
        User = table_U[new_testU[y_i]]
        xsy_step, y_mask = get_code_un(User, User_List)  # ,class_optimizer  , train_optimizer
        batch_t_y.append(xsy_step)
        batch_mask_y.append(y_mask)
    #
    sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
    encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
    input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
    #
    pad_source_lengths = []
    for source in input_x:
        pad_source_lengths.append(len(source) + 1)

    pred_out = sess.run(pred_top,
        feed_dict={target_sequence_length: pad_source_lengths,
                   decoder_embed_input: input_batch, y_:batch_t_y,encoder_embed_input: sources_batch, keep_prob: 1.0,
                   })
    value, full, acc = acc_compute(pred_out[:lens], batch_mask_y[:lens])
    Test_Value.append(value)
    Test_FULL.append(full)
    ACC.append(acc)
    #last batch
    value = np.mean(Test_Value)
    full = np.mean(Test_FULL)
    f1 = 2*value*full/(value+full)
    accuracy=np.mean(ACC)
    return value, full, f1,accuracy
#pic part
def draw_pic(LOSS):
    font={'family':'Trajectory',
          'weight':'bold',
          'size':18
    }
    width=12
    height=12
    plt.figure(figsize=(width,height))
    train_axis=np.array(range(1,len(LOSS)+1,1))
    plt.plot(train_axis,np.array(LOSS),"b--",label="label loss")
    plt.title("Trajectory rnn_FUL Model")
    plt.legend(loc='upper right',shadow=True)
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.show()
def draw_pic_acc(train_ACC,pred_ACC,unpred_ACC):
    font={'family':'Trajectory',
          'weight':'bold',
          'size':18
    }
    width=12
    height=12
    plt.figure(figsize=(width,height))
    train_axis=np.array(range(1,len(train_ACC)+1,1))
    plt.plot(train_axis,np.array(train_ACC),"b--",label="train acc")
    train_axis=np.array(range(1,len(pred_ACC)+1,1))
    plt.plot(train_axis,np.array(pred_ACC),"g--",label="pred acc")
    train_axis=np.array(range(1,len(unpred_ACC)+1,1))
    plt.plot(train_axis,np.array(unpred_ACC),"r--",label="unpred acc")
    plt.title("Trajectory rnn_FUL Model")
    plt.legend(loc='upper right',shadow=True)
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.show()
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
if __name__ == "__main__":
    print 'start'
    train_tuf()
