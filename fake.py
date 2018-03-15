import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import re
from collections import Counter
import os



os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project3/CSC411-A3')

#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')
#os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')





##Additions for Part 7
from sklearn import tree
#import graphviz

def mutual_info(Y, x):
    #I(Y, x) = H(x) - H(x, Y) = H(Y) - H(Y,x)
    #H(Y) = Prob(Y) - sum[ ]
    pass

if __name__ == "__main__":
    
    # Part 1
    with open('resources/clean_fake.txt') as f:
        passage_fake = f.read()
    
    words_fake = re.findall(r'\w+', passage_fake)
    cap_words_fake = [word.upper() for word in words_fake]
    word_counts_fake = Counter(cap_words_fake)
    most_common_fake = word_counts_fake.most_common(5)
    
    with open('resources/clean_real.txt') as f:
        passage_real = f.read()
    
    words_real = re.findall(r'\w+', passage_real)
    cap_words_real = [word.upper() for word in words_real]
    word_counts_real = Counter(cap_words_real)
    most_common_real = word_counts_real.most_common(5)
    

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