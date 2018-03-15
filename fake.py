import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import re
from collections import Counter
import os

#os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project3/CSC411-A3')

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



##Additions for Part 7
from sklearn import tree
#import graphviz

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
    ### By percentages
    fake_stats = get_stats(fake_lines) #compute probabilities for each word
    real_stats = get_stats(real_lines)
   

    new_real = {x: 0 for x in fake_stats if x not in real_stats}
    new_fake = {x: 0 for x in real_stats if x not in fake_stats}
    
    fake_stats.update(new_fake)
    real_stats.update(new_real)
    

    ## Percentages
    ### Get top 10 
    fake_keywords = top_keywords(fake_stats, 10)
    real_keywords = top_keywords(real_stats, 10)
    
    ### Get differences in for common words
    fake_minus_real_perc = {x: fake_stats[x] - real_stats[x] for x in fake_stats if x in real_stats}
    real_minus_fake_perc = {x: real_stats[x] - fake_stats[x] for x in real_stats if x in fake_stats}
    
    ### Get top common words that differ 
    fake_minus_real_keywords = top_keywords(fake_minus_real_perc, 10)
    real_minus_fake_keywords = top_keywords(real_minus_fake_perc, 10)
    
    
    
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