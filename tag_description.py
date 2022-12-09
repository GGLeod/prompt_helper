import numpy as np
import csv
import pandas as pd
import nltk
from nltk import pos_tag
from nltk import RegexpParser
import extract_data_HMM as edH
import copy
import re

tags_list = ["NNS"]
wordVoc = edH.getWordVocab_Bayes(file = 'NN_label.txt')

def getTag(tag):
    tag = tag.replace("[", "")
    tag = tag.replace("]", "")
    tag = tag.replace("'", "")
    tag = tag.replace(", ", ",")
    tokens = tag.split(",")
    return tokens

def getNNS(class_token):
    NNS_list = []
    for words in class_token:
        if words[1] in tags_list:
            NNS_list.append(words[0])
    return NNS_list

def getNN(class_token):
    NN_list = []
    for words in class_token:
        if words[1] == 'NN':
            NN_list.append(words[0])
    return NN_list

def removeNNS(tag_token, NNS_list):
    for word in class_token:
        if word in NNS_list:
            tag_token.remove(word)

def getClass_NaiveBayes(tag_token):
    tag_list = copy.deepcopy(tag_token)
    des_list = []
    for word in tag_token:
        tag_p = 1
        des_p = 1
        word_new = word.split(" ")
        for word_single in word_new:
            if word_single in wordVoc:
                tag_p = tag_p * wordVoc[word_single][0]
                des_p = des_p * wordVoc[word_single][1]
        print(word, tag_p, des_p)
        if (tag_p >= des_p):
            tag_list.remove(word)
        else:
            des_list.append(word)
    return tag_list, des_list


#nltk.download()
text = "learn php from guru99"
tokens = nltk.word_tokenize(text)
print(type(tokens))
tag = nltk.pos_tag(tokens)
print(tag)



#read data
data = pd.read_csv(r'test.csv')
df = pd.DataFrame(data, columns=['original_prompt', 'description', 'tag'])
length = int(df.size / 3)
NN_list = []
f = open("NN.txt", "w")
origin_prompt_list = []
description_list = []
tag_list = []
for i in range(0, length):
    NN_list = []
    print(i / length * 100, '%')
    tag = df.iloc[i]['tag']
    description = df.iloc[i]['description']
    origin = df.iloc[i]['original_prompt']
    description_token = getTag(description)
    origin_token = getTag(origin)
    tag_token = getTag(tag)
    tag_classification = nltk.pos_tag(tag_token)
    NNS_list = getNNS(tag_classification)
    NN_list    = getNN(tag_classification)
    description_token.extend(NNS_list)
    tag_token = [word for word in tag_token if word not in NNS_list]
    #print(tag_token)
    tag_final, des_extend = getClass_NaiveBayes(tag_token)
    description_token.extend(des_extend)
    origin_prompt_list.append(origin_token)
    description_list.append(description_token)
    tag_list.append(tag_final)
    #if (i < 400):
    #    print(NN_list)
    #    for word in NN_list:
    #        f.write(word + '\n')
    #if (i == 400):
    #    f.close()
dict = {'original_prompt': origin_prompt_list, 'description': description_list, 'tag': tag_list}  
df_wr = pd.DataFrame(dict) 
df_wr.to_csv('refine.csv') 
    #print('\n', tag_classification, '\n')
    #print(classification)
    #print(classification[0][1])

