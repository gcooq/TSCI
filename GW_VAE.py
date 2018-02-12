# -*- coding:  UTF-8 -*-
'''
Created on 2017-7-28
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

# This files is to pre-train the brightkite dataset bacause the location is much sparse.
#add PAD
# Parameters
n_hidden = 800
batch_size =16 #batch
n_input=250
n_classes=201
keep_prob = tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
#define for clssification
train_iters=50 #num for training
z_size=800
latentscale_iter=tf.placeholder(dtype=tf.float32)
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
total_seqlens = list()  # 原始估计长度
#define the weight and bias dictionary
with tf.name_scope("decoder_inital"):
    weights_de={
        'w_':tf.Variable(tf.random_normal([z_size,n_hidden],mean=0.0, stddev=0.01))
    }
    biases_de = {
    'b_': tf.Variable(tf.random_normal([n_hidden], mean=0.0, stddev=0.01))
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
def getPvector(i):  # 传递的是轨迹ID 查询其Embedding tensor
    return table_X[i]


def getXs():  # 读取轨迹向量
    fpointvec = open('data/gowalla_user_vector250d_.dat', 'r')  # 获取check-in向量 已经用word2vec训练得到
    #     table_X={}  #建立字典索引
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()

        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  # 统计条目数
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  # 读取向量数据
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
        y_[MASK * 2] = 0  # 说明其是朋友
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
        y_[MASK * 2] = 0  # 说明其是朋友
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
    tempT=list()  #临时数据 所有数据
    pointT = list()  # 轨迹ID集合
    userT = list()  # 用户ID
    seqlens = list()  # 句子长度或者说是轨迹点的个数
    item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i)) #chanage to string or char type
        tempT.append(X)
        userT.append(int(X[0]))
        pointT.append(X[1:])
        seqlens.append(len(X) - 1)  # 包含了一个用户data
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
            TRA_ALL.append(int(tempT[i][0])) #存储用户ID
    print 'UNKNOWN ----->', len(all_T)

    flag = 0
    count = 0;
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  # 记录用户数量
    rate = 0.5
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):
            User += 1
            # 分割数据
            if (count > 1):  # 分割/home/gaoqiang/workspace_demo
                # print "count",count," ",index
                test_T += (pointT[int((index - math.ceil(count * rate))):index])  # 测试数据轨迹点
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])  # 测试数据用户
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])  # 测试数据轨迹长
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1;  # 复位
            flag = userT[index]  # 更新
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
    return TRA_ALL,all_T, pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List, total_T, total_U, total_seqlens  # 返回相关参数

#Encoder layer
def get_encoder_layer(encoder_input, keep_prob,reuse=False):
    with tf.variable_scope("encoder",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # 前向 , state_is_tuple=True
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # 后向 , state_is_tuple=True
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        # 预留多层正反向LSTM功能
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
        else:  # 测试 different type of decoder output
            copy = tf.tile(tf.constant([vocab_to_int['<GO>']]), [batch_size])
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dic_embeddings, copy, vocab_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, predicting_helper, decoder_initial_state)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                         maximum_iterations=max_target_sequence_length)

        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        #target = tf.concat([target_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  # 掩码轨迹点
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
latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
#variable
# a=tf.reduce_sum(tf.exp(z_stddev),1)
# b=tf.reduce_sum(z_stddev,1)
# c=tf.reduce_sum(tf.square(z_mean),1)
latent_cost=tf.reduce_mean(latent_loss)
laten_=latentscale_iter* tf.reduce_mean(latent_loss)

encropy_loss=tf.contrib.seq2seq.sequence_loss(training_logits, target, masks)
cost = tf.reduce_mean(
        tf.contrib.seq2seq.sequence_loss(training_logits, target, masks) + latentscale_iter * (latent_loss))
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)


def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
initial = tf.global_variables_initializer()
def train_tuf():
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print 'name:', v.name
    saver=tf.train.Saver() #variables
    with tf.Session() as sess:
        sess.run(initial)
        #saver.restore(sess, './out_data/bk_ae.pkt')
        #saver.restore(sess,'./out_data/bri_train_ae.pkt')
        print'Read train & test data'
        TOTAL_LOSS=[]
        TOTAL_ACC = []

        LEARN_RATE=[]
        lens = 0.000008
        initial_learning_rate=0.001
        learning_rate_len=0.000008
        min_kl=0.0         #0.158
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
            initial_learning_rate -= learning_rate_len
            while step < len(new_trainT)//batch_size:
                start_i = step * batch_size
                input_x = new_trainT[start_i:start_i + batch_size]
                # 补全序列
                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                encode_batch=eos_sentence_batch(input_x,vocab_to_int['<EOS>'])
                input_batch=pad_sentence_batch(encode_batch,vocab_to_int['<PAD>'])
                # 记录长度
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source) + 1)
                # print 'output',input_x
                length_x = len(input_x) + 1
                if min_kl_epoch<1.0:
                    min_kl_epoch = min_kl + step* lens
                if (initial_learning_rate <= 0):
                    initial_learning_rate = 0.000008
                encropy_cost,new_states_,vae_loss,pred, targets,opt,loss = sess.run([encropy_loss,decoder_initial_state,latent_loss,predicting_logits, target, optimizer,cost],
                                                    feed_dict={target_sequence_length: pad_source_lengths,
                                                             decoder_embed_input: input_batch, encoder_embed_input:sources_batch,keep_prob: 0.5,latentscale_iter:min_kl_epoch,it_learning_rate:initial_learning_rate})


                LOSS += loss
                en_Loss+=encropy_cost
                gen_LOSS+=np.mean(vae_loss)

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
            #print lost_len
            #print len(new_trainT[:lost_len])
            input_x=last_x+new_trainT[:lost_len]
            #print len(input_x)
            # 补全序列
            sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])

            encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
            input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
            # 记录长度
            pad_source_lengths = []

            for source in input_x:
                pad_source_lengths.append(len(source) + 1)
            encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
                [encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
                feed_dict={target_sequence_length: pad_source_lengths,
                           decoder_embed_input: input_batch, encoder_embed_input: sources_batch, keep_prob: 0.5,
                           latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})

            tuf_acc = prediction_tuf(sess, new_testT, test_UserT,User_List,initial_learning_rate,min_kl_epoch)
            un_acc = prediction_tuf_unkown(sess, new_unT, User_List, TRA_ALL,initial_learning_rate,min_kl_epoch)

            print 'epoch',epoch,'ACC Train',acc_count/num,'tuf acc',tuf_acc,'un_acc',un_acc
            saver.save(sess, './out_data/GW_VAE.pkt')
        # 画图
        # print TOTAL_LOSS
        if (os.path.exists("out_data/GW_loss_vae_new.txt")): os.remove('out_data/GW_loss_vae_new.txt')
        save_loss(LEARN_RATE,TOTAL_LOSS,TOTAL_ACC)
        #draw_pic_unloss(TOTAL_LOSS)
        #draw_pic_unacc(TOTAL_ACC)

def save_loss(LEARN_RATE,TOTAL_LOSS,TOTAL_ACC):
    fopen = open('out_data/GW_loss_vae.txt', 'w')
    for i in range(len(TOTAL_LOSS)):
        fopen.write('epoch\t' + str(i) +'\t' +str(LEARN_RATE[i])+'\t'+str(TOTAL_LOSS[i]) +'\t'+ str(TOTAL_ACC[i])+'\n')
    fopen.close()
#画图部分
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
def get_batches(sources, batch_size, source_pad_int):
    for batch_i in range(0,len(new_total_T)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
def prediction_tuf_unkown(sess,all_T,User_List,TRA_ALL,initial_learning_rate,min_kl_epoch):
    step=0
    num=0
    acc_count=0
    temp_acc=0
    # sort
    index_T = {}
    new_testT = []
    new_testU = []
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


        #补全序列
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        # 记录长度
        pad_source_lengths = []

        for source in input_x:
            pad_source_lengths.append(len(source) + 1)
        encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
            [encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
            feed_dict={target_sequence_length: pad_source_lengths,
                       decoder_embed_input: input_batch, encoder_embed_input: sources_batch, keep_prob: 0.5,
                       latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})

        #print pred_out


        # if (step % 100 == 0):
        #     print 'show acc_count', temp_acc / (100 * batch_size)
        #     print 'min_kl_epoch', min_kl_epoch
        #     temp_acc = 0
        #     print'total_cost', loss, 'encropy_cost:', encropy_cost, 'laten_cost:', np.mean(vae_loss)
        #     print('test-----------------')
        #     # print 'pred', pred[0][:len(input_x[0])]
        #     # print 'states_x', new_states_[0][:15]
        #     print 'end'
        for i in range(len(input_x)):
            if ((input_x[i] == pred[i][:len(input_x[i])]).all()):
                # print 'OUT',input_x[i]
                acc_count += 1
                temp_acc += 1
        step += 1
        num+=batch_size

    # last batch<batch_size dealing
    sid = (step) * batch_size
    last_x = new_testT[sid:]
    lens = len(last_x)
    lost_len = batch_size - lens
    input_x = last_x + new_testT[:lost_len]

    # 补全序列
    sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
    encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
    input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
    # 记录长度
    pad_source_lengths = []
    for source in input_x:
        pad_source_lengths.append(len(source) + 1)
    encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
        [encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
        feed_dict={target_sequence_length: pad_source_lengths,
                   decoder_embed_input: input_batch, encoder_embed_input: sources_batch, keep_prob: 0.5,
                   latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})

    #last batch
    ACC=acc_count/num
    return ACC
def prediction_tuf(sess,testT,testU,User_List,initial_learning_rate,min_kl_epoch):
    step=0
    temp_acc=0
    acc_count=0
    num=0
    # sort
    index_T = {}
    new_testT = []
    new_testU = []
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

        #补全序列
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        # 记录长度
        pad_source_lengths = []
        for source in input_x:
            pad_source_lengths.append(len(source) + 1)
        batch_t_y = []
        batch_mask_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step, y_mask = get_code(new_testU[y_i], User_List)  # ,class_optimizer  , train_optimizer
            batch_t_y.append(xsy_step)
            batch_mask_y.append(y_mask)
        encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
            [encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
            feed_dict={target_sequence_length: pad_source_lengths,
                       decoder_embed_input: input_batch, encoder_embed_input: sources_batch, keep_prob: 0.5,
                       latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})

        # if (step % 1000 == 0):
        #     print 'show acc_count', temp_acc / (100 * batch_size)
        #     print 'min_kl_epoch', min_kl_epoch
        #     temp_acc = 0
        #     print'total_cost', loss, 'encropy_cost:', encropy_cost, 'laten_cost:', np.mean(vae_loss)
        #     print('test-----------------')
        #     # print 'pred', pred[0][:len(input_x[0])]
        #     # print 'states_x', new_states_[0][:15]
        #     print 'end'
        for i in range(len(input_x)):
            if ((input_x[i] == pred[i][:len(input_x[i])]).all()):
                # print 'OUT',input_x[i]
                acc_count += 1
                temp_acc += 1
        step += 1
        num += batch_size


    # last batch<batch_size dealing
    sid = (step) * batch_size
    last_x = new_testT[sid:]
    lens = len(last_x)
    lost_len = batch_size - lens
    input_x = last_x + new_testT[:lost_len]

    # 补全序列
    sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
    encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
    input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
    # 记录长度
    pad_source_lengths = []
    for source in input_x:
        pad_source_lengths.append(len(source) + 1)
    encropy_cost, new_states_, vae_loss, pred, targets, opt, loss = sess.run(
        [encropy_loss, decoder_initial_state, latent_loss, predicting_logits, target, optimizer, cost],
        feed_dict={target_sequence_length: pad_source_lengths,
                   decoder_embed_input: input_batch, encoder_embed_input: sources_batch, keep_prob: 0.5,
                   latentscale_iter: min_kl_epoch, it_learning_rate: initial_learning_rate})

    #last batch
    ACC=acc_count/num
    return ACC

if __name__ == "__main__":
    print 'start'
    train_tuf()
