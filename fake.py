import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#import graphviz

#os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')
os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project3/CSC411-A3')


def get_data(filename):
    ''' return a list whose elements are each a headline from filename'''
    rd.seed(0)
    with open(filename, 'r') as file:
        lines = file.read().split('\n')
    rd.shuffle(lines) #randomize the order of headlines
    return lines

def get_stats(lines, output='dict'):
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

def all_words(dict1, dict2):
    missing = { x:0 for x in dict1.keys() if x not in dict2.keys() }
    dict2.update( missing )
    
    missing = { x:0 for x in dict2.keys() if x not in dict1.keys() }
    dict1.update( missing )
    return dict1, dict2

def part_1(dict1, dict2):   
    ''' return a list of the top 10 words by percentage that appear in dict1 over dict2'''
    #new_dict1 = {x: 0 for x in dict2 if x not in dict1}
    #dict1.update(new_dict1)
    dict1_minus_dict2_perc = {x: dict1[x] - dict2[x] for x in dict1 if x in dict2}
    dict1_minus_dict2_keywords = top_keywords(dict1_minus_dict2_perc, 10)
    return dict1_minus_dict2_keywords

def sets(fake_lines, real_lines):
    '''divide the data into training, validation, and testing sets'''
    rd.seed(1)
    rd.shuffle(fake_lines)
    rd.shuffle(real_lines)

    training_set   = fake_lines[:int(round(0.7*len(fake_lines)))]
    validation_set = fake_lines[int(round(0.7*len(fake_lines))):int(round(0.85*len(fake_lines)))]
    testing_set    = fake_lines[int(round(0.85*len(fake_lines))):]
    y_tr = [1]*len(training_set)
    y_va = [1]*len(validation_set)
    y_te = [1]*len(testing_set)

    training_set.extend(   real_lines[:int(round(0.7*len(real_lines)))] )
    validation_set.extend( real_lines[int(round(0.7*len(real_lines))):int(round(0.85*len(real_lines)))] )
    testing_set.extend(    real_lines[int(round(0.85*len(real_lines))):] )
    y_tr += [0]*( len(training_set) - len(y_tr) )
    y_va += [0]*( len(validation_set) - len(y_va) )
    y_te += [0]*( len(testing_set) - len(y_te) )

    return training_set, validation_set, testing_set, y_tr, y_va, y_te

def dict_to_vec(all_words, lines):
    '''given a list of headlines in lines, return a list of lists, where each inner list 
    is a sparse vector with a 1 in the place of each word in the headline'''
    X = []
    for line in lines:
        x = []
        words = list(set( line.split(' ') )) #converting to set and back to a list removes duplicates
        for word in all_words:
            val = 1 if word in words else 0
            x += [val]
        X += [x]
    return X

def mutual_info(Y, x):
    #I(Y, x) = H(x) - H(x, Y) = H(Y) - H(Y,x)
    #H(Y) = Prob(Y) - sum[ ]
    pass


if __name__ == "__main__":
    
    ## Part 1
    #   Get data
    fake_lines = get_data('resources/clean_fake.txt') #Get list containing headlines
    real_lines = get_data('resources/clean_real.txt')
    #   Get probabilities 
    fake_stats = get_stats(fake_lines) #compute probabilities for each word
    real_stats = get_stats(real_lines)
    fake_stats, real_stats = all_words(fake_stats, real_stats) #add missing words to each dict
   
    #   Top 10 keywords by percentage
    fake_keywords = top_keywords(fake_stats, 10)
    real_keywords = top_keywords(real_stats, 10)
    #   Top differences by percentage
    fake_minus_real_top = part_1(fake_stats, real_stats)
    real_minus_fake_top = part_1(real_stats, fake_stats)

    #   Divide datasets
    training_set, validation_set, testing_set, y_tr, y_va, y_te = sets(fake_lines, real_lines)
    tr = get_stats(training_set)
    va = get_stats(validation_set)
    te = get_stats(testing_set)



    ## Part 2
    p_fake = len(fake_lines)/(len(fake_lines) + len(real_lines))
    p_real = len(real_lines)/(len(fake_lines) + len(real_lines))
    
    # p(word|real) = the percentage for that word in real_stats
    
    #wasn't sure what this was for... commenting cause I kept thinking I needed to edit this
    '''
    list1 = []
    for headline in training_set[0:2]:
        #print(headline)
        words = list(set( headline.split(' ') )) #converting to set and back to a list removes duplicates
        for word in words:
            #print(word)
            real_stats.get(word)
    '''

    ## Part 7
        #note that this entire section was run with the rd.seed() in the sets() function as rd.seed(1)
    rd.seed(0)  #numpy randomness used internally of sklearn.tree
    max_depths = [2, 3, 5, 10, 15, 20, 35, 50, 75, 100, None]
    max_feats = [3, 10, 15, None] #max_features
    all_words = list( fake_stats.keys() )
    stp_wrds=False #include stop words
    if not stp_wrds:
        all_words = [x for x in all_words if x not in ENGLISH_STOP_WORDS]
    
    X = dict_to_vec(all_words, training_set)
    Y = y_tr.copy()
    
    x_va = dict_to_vec(all_words, validation_set)
    y_va = np.array(y_va)
    
    x_te = dict_to_vec(all_words, testing_set)
    y_te = np.array(y_te)
    
    for max_feat in max_feats:
        tr_res = [] #initialize variables to store results
        va_res = []
        te_res = []
        for dep in max_depths:
            clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=dep, max_features=max_feat)
                #X = [[0, 0], [1, 1], [3,2] ] #replace these with actual training data
                #Y = [0, 1,1]
            clf.fit(X,Y)
            
            info = 'max_dep='+str(dep) + '_max_features='+str(max_feat) + 'stop_words='+str(stp_wrds) #label for filename with info on parameters used
            dot_data = tree.export_graphviz(clf, out_file='resources/part7/'+info+'.dot' )
                #use http://webgraphviz.com/ to generate graphic from this file
            
            tr_result = clf.predict(X) #get accuracy on training set
            correct = len(y_tr) - np.count_nonzero(tr_result - y_tr) 
            tr_res += [ correct/len(y_tr) ] 
            va_result = clf.predict(x_va) #on validation set
            correct = len(y_va) - np.count_nonzero(va_result - y_va) 
            va_res += [ correct/len(y_va) ]
            te_result = clf.predict(x_te) #on testing set
            correct = len(y_te) - np.count_nonzero(te_result - y_te) 
            te_res += [ correct/len(y_te) ] 
        filename = 'part7a_max_features='+str(max_feat)+'stop_words='+str(stp_wrds)+'.jpg'
        e = len(max_depths) - 1
        plt.scatter(max_depths[:e], tr_res[:e], label='Training Data')
        plt.scatter(max_depths[:e], va_res[:e], label='Validation Data')
        plt.scatter(max_depths[:e], te_res[:e], label='Testing Data')
        plt.title('Learning Curve')
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        plt.legend()
        #plt.show()
        plt.savefig('resources/' + filename)
        plt.close()
    
    
    
    '''
    if False: #graphviz doesn't work
        from sklearn.datasets import load_iris
        iris = load_iris()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris.data, iris.target)
        dot_data = tree.export_graphviz(clf, out_file='test.dot') 
        graph = graphviz.Source(dot_data) 
        graph.render() 
    '''