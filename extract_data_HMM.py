import numpy as np
import os
import copy


def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    #for x in unique_list:
    #    print x,
    return unique_list

def getWordVocab_Bayes(file = 'NN_label.txt'):
    word_line = []
    unique_line = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            line_new = line.replace("\n", "")
            #print(line_new.split(" "))
            line_new = line_new.split(" ")

            if (line_new[-1] != "1" and line_new[-1] != "2" ):
                line_new.pop()
                word_line.append(copy.deepcopy(line_new))
                if (line_new[-1] == "1"):
                    line_new.remove("1")     
                else:
                    line_new.remove("2")   
                unique_word_line = unique(line_new)
                unique_line.extend(unique_word_line)
                #print(line_new[-1])
            else:
                word_line.append(copy.deepcopy(line_new))
                if (line_new[-1] == "1"):
                    line_new.remove("1")     
                else:
                    line_new.remove("2")  
                unique_word_line = unique(line_new)
                unique_line.extend(unique_word_line)
                #print(line_new[-1])
        unique_line = unique(unique_line)

    word_vocablary = {}
    for word in unique_line:
        num_total   = 0
        tag         = 1
        description = 1
        for i in range(0, len(word_line)):
            line = word_line[i]
            num_total = num_total + 1
            if word in line:
                if word_line[-1] == "1":
                    tag = tag + 1
                else:
                    description = description + 1
        tag_p = tag / num_total
        des_p = description / num_total
        word_vocablary[word] = [tag_p, des_p]
    return word_vocablary
#print(word_vocablary)