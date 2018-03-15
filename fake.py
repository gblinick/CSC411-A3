import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import re
from collections import Counter

##Additions for Part 7
from sklearn import tree
#import graphviz

#os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')

#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project3/CSC411-A3')

def get_data(filename):
    ''' return a list whose elements are each a headline from filename'''
    rd.seed(0)
    with open(filename, 'r') as file:
        lines = file.read().split('\n')
    rd.shuffle(lines) #randomize the order of headlines
    return lines

def get_stats(lines):
    ''' return a dictionary, where every word that appears in a headline in lines
    is a key, with value equal to count(word)/number of headlines'''
    dict = {}
    for line in lines:
        words = list(set( line.split(' ') )) #converting to set and back to a list removes duplicates
        for word in words:
            if word in dict.keys():
                dict[word] += 1
            else: 
                dict[word] = 1
    k = len( lines ) #number of headlines
    dict = { word:dict[word]/k for word in dict.keys() }
    return dict

def top_keywords(dict, num):
    ''' returns a list of tuples with the keys in dict having the highest value;
    num tuples will in the list'''
    keys = dict.keys()
    vals = dict.values() #the order will match, given that dict is not touched in between calls
    kv = sorted( zip(vals, keys), reverse=True )[:num] #sorts both lists based on order of vals, and selects the top num results
    return kv


def part_1(dict1,dict2):   
    ''' return a list of the top 10 words by percentage that appear in dict1 over dict2'''
    new_dict1 = {x: 0 for x in dict2 if x not in dict1}
    dict1.update(new_dict1)
    dict1_minus_dict2_perc = {x: dict1[x] - dict2[x] for x in dict1 if x in dict2}
    dict1_minus_dict2_keywords = top_keywords(dict1_minus_dict2_perc, 10)
    
    return dict1_minus_dict2_keywords

def sets(fake_lines, real_lines):
    rd.seed(0)
    rd.shuffle(fake_lines)
    training_set = fake_lines[:int(round(0.7*len(fake_lines)))]
    validation_set = fake_lines[int(round(0.7*len(fake_lines))):int(round(0.85*len(fake_lines)))]
    testing_set = fake_lines[int(round(0.85*len(fake_lines))):]
    
    rd.seed(0)
    rd.shuffle(real_lines)
    training_set.extend(real_lines[:int(round(0.7*len(real_lines)))])
    validation_set.extend(real_lines[int(round(0.7*len(real_lines))):int(round(0.85*len(real_lines)))])
    testing_set.extend(real_lines[int(round(0.85*len(real_lines))):])
    
    rd.seed(0)
    rd.shuffle(training_set)
    rd.shuffle(validation_set)
    rd.shuffle(testing_set)
    
    return training_set, validation_set, testing_set

def mutual_info(Y, x):
    #I(Y, x) = H(x) - H(x, Y) = H(Y) - H(Y,x)
    #H(Y) = Prob(Y) - sum[ ]
    pass

if __name__ == "__main__":
    
    # Part 1
    ## Get data
    fake_lines = get_data('resources/clean_fake.txt') #Get list containing headlines
    real_lines = get_data('resources/clean_real.txt')
    
    ## Sort 
    '''Note that fake_stats and real_stats get changed (appended to) in the part_1 function
    when we add in missing words'''
    fake_stats = get_stats(fake_lines) #compute probabilities for each word
    real_stats = get_stats(real_lines)
   
    
    ## Top differences by percentage
    fake_minus_real_top = part_1(fake_stats, real_stats)
    real_minus_fake_top = part_1(real_stats, fake_stats)

    ## Top 10 keywords by percentage
    fake_keywords = top_keywords(fake_stats, 10)
    real_keywords = top_keywords(real_stats, 10)
    
    ## Divide datasets
    training_set, validation_set, testing_set = sets(fake_lines, real_lines)


    # Part 2
    

    #Part 7
    rd.seed(0) #numpy randomness used internally of sklearn.tree
    max_depths = [5, 10, 15, 20, None]
    for dep in max_depths:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=dep)
        X = [[0, 0], [1, 1]] #replace these with actual training data
        Y = [0, 1]
        clf.fit(X,Y)
        dot_data = tree.export_graphviz(clf, out_file='resources/part7/max_dep='+str(dep)+'.dot' ) 
        #use http://webgraphviz.com/ to generate graphic from this file

    
    if False: #graphviz doesn't work
        from sklearn.datasets import load_iris
        iris = load_iris()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris.data, iris.target)
        dot_data = tree.export_graphviz(clf, out_file='test.dot') 
        graph = graphviz.Source(dot_data) 
        graph.render() 