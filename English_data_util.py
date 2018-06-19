#-*- utf-8 -*-
"""
input: label data, wordvector data
output:
    revs: list.     Store the all informaiton of sentences: label, text, num_words
    vocab: dict: word, nums.    the dict to store all appeared word in train_test data
    w2v: dict: word, word_vector. 
    word_idx_map: dict: word, word_id.
    W: dict: word_id, word_vector. part of vector from Google 300 vector
    W2: dict: word_id, word_vector. All random vector, in range(-0.25, 0.25)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))

"""
import numpy as np
np.random.seed(1337) # for reproducibility

import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import gc
import time

haha=""
#data_folder: the path to load data
#clean_string : True = clean the sentence
def build_data_cv(data_folder, clean_string=True):
    """
    Loads sentence data ;
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    ne_pos = open("lower_positive","w")
    ne_neg = open("lower_negative","w")
    vocab = defaultdict(float)
    max_word_len = 0
    alphabet = []
    with open(pos_file) as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            words = set(orig_rev.split())
            if(len(orig_rev.split())> 55):
                continue
            flag=0
            for word in words:
                if(len(word)>20):
                    flag=1
                    break
                vocab[word] += 1
                global haha
                if(len(word)>max_word_len):
                    haha=word
                max_word_len = max(max_word_len, len(word))
                for alp in word:
                    if alp not in alphabet:
                        alphabet.append(alp)
            
            datum  = {"label":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split())
                      }
            if(flag == 0):
                ne_pos.write(orig_rev+"\n")
                revs.append(datum)
    with open(neg_file) as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            if(len(orig_rev.split())> 55):
                continue
            flag = 0
            for word in words:
                if(len(word)>20):
                    flag=1
                    break
                vocab[word] += 1
                max_word_len = max(max_word_len, len(word))
                for alp in word:
                    if alp not in alphabet:
                        alphabet.append(alp)

            datum  = {"label":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split())
                      }
            if(flag ==0):
                ne_neg.write(orig_rev+"\n")
                revs.append(datum)
    ne_pos.close()
    ne_neg.close()
    return revs, vocab, max_word_len, set(alphabet)
    
def generate_char2vec(alphabet, char2v_dim):
    char2vec = {}
    for alp in alphabet:
        char2vec[alp] = np.random.uniform(-0.25, 0.25, char2v_dim)
    char2vec[" "] = np.random.uniform(-0.25, 0.25, char2v_dim)
    return char2vec

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def char_load_data(path, vector_type="-word2vec", filter_h=5, model_type="CNN"):

    print "Loading data..."
    X = cPickle.load(open(path, "rb"))
    max_len, max_word_len, char2v_dim, revs, char2vec = X[0], X[1], X[2],X[3], X[4]
    #cPickle.dump([max_l, max_word_len, char2v_dim, revs, char2vec], open("charmr.p", "wb"))

    revs_size = len(revs)
    pad = filter_h -1
    #gc.collect()
    print "data loaded"
    print "number of sentences: " + str(revs_size)
    print "max sentence length: " + str(max_len)
    print "max word length: " + str(max_word_len)

    #forward back zero
    

    #input_shape
    input_shape = []
    input_shape.append(1)
    input_shape.append(max_len)
    input_shape.append(max_word_len)
    input_shape.append(char2v_dim)

    zeros_vector = np.zeros(char2v_dim)

    label = np.empty((revs_size,), dtype="uint8")
    if(model_type == "CNN"):
        data = np.empty((revs_size, input_shape[0], input_shape[1]+2*pad, input_shape[2]+2,input_shape[3]), dtype = "float32")
        for i in range(revs_size):
            rev = revs[i]
            s2v = get_sen_char2v(rev["text"], char2vec,zeros_vector, max_len, max_word_len, input_shape, filter_h)
            data[i,0,:,:,:] = s2v
            label[i] = int(rev["label"])
        label_class = len(set(label))

    elif (model_type == "RNN"):
        data = np.empty((revs_size, (max_len + 2*pad) ,  w2v_dim), dtype = "float32")
        #data = np.empty((revs_size,1, (max_len + 2*pad) *  w2v_dim), dtype = "float32")
        for i in range(revs_size):
            rev = revs[i]
            s2v = get_sen_char2v(rev["text"], char2vec,zeros_vector, max_len, max_word_len, input_shape, filter_h)
            data[i,:,:] = s2v
            #data[i,0,:] = [x for X in s2v for x in X]
            label[i] = int(rev["label"])
        label_class = len(set(label))


    return char2v_dim, label_class, data, label

#sen: sentence
#w2v: word to vector
#s2v: sententce to vector
def get_sen_char2v(sent, char2vec, zeros_vector, max_len, max_word_len, input_shape, filter_h=5):
    words = sent.split()
    pad = filter_h - 1
    s2charvec = []
    pad1 = (2*pad+max_len-len(words))/2
    pad2 = 2*pad+max_len-len(words)-pad1

    for i in range(pad1):
        temp=[]
        temp.append(char2vec[" "])
        for j in range(max_word_len):
            temp.append( zeros_vector )
        temp.append(char2vec[" "])
        s2charvec.append(temp)

    for word in words:
        temp=[]
        temp.append(char2vec[" "])
        gap = int( (max_word_len - len(word)) /2)
        for i in range(gap):
            temp.append(zeros_vector)
        for alp in word:
            temp.append(char2vec[alp])
        for i in range(max_word_len-len(word)-gap):
            temp.append(zeros_vector)
        temp.append(char2vec[" "])
        s2charvec.append(temp)

    for i in range(pad2):
        temp=[]
        temp.append(char2vec[" "])
        for j in range(max_word_len):
            temp.append( zeros_vector )
        temp.append(char2vec[" "])
        s2charvec.append(temp)

       
    return s2charvec








if __name__=="__main__":    

    #config
    data_folder = ["./positive","./negative"]    
    char2v_dim = 15

    #Begin
    print "loading data...",        
    revs, vocab, max_word_len, alphabet = build_data_cv(data_folder, clean_string=True)
    #find the longest sentences as the input feature, shorter than it
    #will  be pad by zero
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "generate char2vec vectors...",
    char2vec = generate_char2vec(alphabet, char2v_dim)
    #w2v = load_bin_vec(w2v_file, vocab)
    print "char2vec generate!"

    """
    revs: list.     Store the all informaiton of sentences: label, text, num_words
    vocab: dict: word, nums.    the dict to store all appeared word in train_test data
    word_idx_map: dict: word, word_id.
    W: dict: word_id, word_vector. part of vector from Google 300 vector
    W2: dict: word_id, word_vector. All random vector, in range(-0.25, 0.25)

    """
    print (max_l, max_word_len, char2v_dim,haha)
    cPickle.dump([max_l, max_word_len, char2v_dim, revs, char2vec],open("charmr_big", "wb"))
    print "dataset created!"
    
