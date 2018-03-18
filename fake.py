import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import math
import time
from sklearn import tree
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#import graphviz

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

def get_count(lines, output='dict'):
    ''' return a dictionary, where every word that appears in a headline in lines
    is a key, with value equal to count(word)'''
    dict = {}
    for line in lines:
        words = list(set( line.split(' ') )) #converting to set and back to a list removes duplicates
        for word in words:
            if word in dict.keys():
                dict[word] += 1
            else: 
                dict[word] = 1
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

def mutual_info(word, y):
    count_word = 0 #number of headlines with word in training set
    lines_with_word = []
    lines_without_word = []
    for k in range(len(training_set)):
        if word in training_set[k]:
            count_word += 1
            lines_with_word += [k]
        else:
            lines_without_word += [k]
    prob_word = count_word/len(training_set)
    
    y_with = np.array( [y[i] for i in lines_with_word] )
    y_without = np.array( [y[i] for i in lines_without_word] )
    
    prob_fake = np.count_nonzero(y)/len(y)
    H = prob_fake*np.log(prob_fake) + (1 - prob_fake)*np.log(1 - prob_fake) #entropy before split
    H = - H/np.log(2) #convert to base 2, and apply negative
    
    prob_fake_with = np.count_nonzero(y_with)/len(y_with)
    Hy = prob_fake_with*np.log(prob_fake_with) + (1 - prob_fake_with)*np.log(1 - prob_fake_with) 
    Hy = - Hy/np.log(2) #entropy of headlines with word
    
    prob_fake_without = np.count_nonzero(y_without)/len(y_without)
    Hn = prob_fake_without*np.log(prob_fake_without) + (1 - prob_fake_without)*np.log(1 - prob_fake_without) 
    Hn = - Hn/np.log(2)  #entropy of headlines without word
    
    I = H - (Hy*len(y_with) + Hn*len(y_without) )/len(y)
    return I



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

    


    ## Part 2
    p_fake = len(fake_lines)/(len(fake_lines) + len(real_lines))
    p_real = len(real_lines)/(len(fake_lines) + len(real_lines))
    
        # p(word|real) = the percentage for that word in real_stats
        # what we want is p(real|words in headline).
        # this is equal to p(words in headline|real)*p(real), divided by
        #   p(words in headline|real)*p(real) + p(words in headline|fake)*p(fake)
        # We already have p(real) and p(fake) from above, so we just need to find
        # p(words in headline|real) and p(words in headline|fake)
        # p(words in headline|real) is equal to count(word & real) + mp
        # divded by count(real) + m.
        # So we need to choose values for m and p and then come up with a
        # dictionaries of the adjusted probabilities.
    
    all_words = list(real_stats.keys())
    
    p = 1/(2*5833) #2 classes and 5833 words leads to 1 example per word per class
    m = 2*5833
    
    fake_counts = get_count(fake_lines) 
    real_counts = get_count(real_lines)

    missing = { x:0 for x in fake_counts.keys() if x not in real_counts.keys() }
    real_counts.update( missing )
    missing = { x:0 for x in real_counts.keys() if x not in fake_counts.keys() }
    fake_counts.update( missing )
    
    
    adjusted_fake_counts = {} 
    adjusted_real_counts = {} 
    
    for word in all_words:
        adjusted_fake_counts[word] = fake_counts[word] + m*p
        adjusted_real_counts[word] = real_counts[word] + m*p
        
    naive_divisor = len(all_words) + m

    adjusted_fake_stats = {} #P(w | fake)
    adjusted_real_stats = {} #P(w | real)
    for word in all_words:
        adjusted_fake_stats[word] = adjusted_fake_counts[word]/naive_divisor
        adjusted_real_stats[word] = adjusted_real_counts[word]/naive_divisor
 
    
    # p(words in headline|real) and p(words in headline|fake) 
    # put into the dictionaries total_real_probabilities and 
    # total_fake_probabilities, respectively
    total_real_probabilities = {}
    total_fake_probabilities = {}
    
    for headline in training_set:
        probabilities_real = []
        probabilities_fake = []
        
        headline_words = list(set( headline.split(' ') )) #converting to set and back to a list removes duplicates
        for word in headline_words:
            probabilities_real.append(adjusted_real_stats.get(word))
            probabilities_fake.append(adjusted_fake_stats.get(word))
        non_headline_words = [x for x in all_words if x not in headline_words]
        for word in non_headline_words:
            probabilities_real.append(1 - adjusted_real_stats.get(word))
            probabilities_fake.append(1 - adjusted_fake_stats.get(word))
        
        total_real_probability = 0
        total_fake_probability = 0
        for k in range(len(probabilities_real)):
            total_real_probability = total_real_probability + math.log(probabilities_real[k])
            total_fake_probability = total_fake_probability + math.log(probabilities_fake[k])
        total_real_probability = math.exp(total_real_probability)
        total_real_probabilities[headline] = total_real_probability
        total_fake_probability = math.exp(total_fake_probability)
        total_fake_probabilities[headline] = total_fake_probability

    #Compute final probabilites
    numerator = [p_fake*x for x in total_fake_probabilities.values()]
    denom_term2 = [p_real*x for x in total_real_probabilities.values()]
    
    final = [numerator[i]/(numerator[i] + denom_term2[i]) for i in range(len(numerator))]
    pred = [round(x) for x in final]
    
    pred = np.array(pred)
    y_tr2 = np.array(y_tr)
    
    correct = len(y_tr) - np.count_nonzero(tr_result - y_tr)
    acc = correct/len(y_tr)



    ## Part 7
        #note that this entire section was run with the rd.seed() in the sets() function as rd.seed(1)
    rd.seed(0)  #numpy randomness used internally of sklearn.tree
    max_depths = [2, 3, 5, 10, 15, 20, 35, 50, 75, 100, None]
    max_feats = [3, 10, 15, None] #max_features
    all_words = list( fake_stats.keys() )
    stp_wrds=True #True -> includes stop words
    if not stp_wrds:
        all_words = [x for x in all_words if x not in ENGLISH_STOP_WORDS]
    split_cond = 'entropy' # or 'gini'
    
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
            clf = tree.DecisionTreeClassifier(criterion=split_cond, max_depth=dep, max_features=max_feat)
                #X = [[0, 0], [1, 1], [3,2] ] #replace these with actual training data
                #Y = [0, 1,1]
            clf.fit(X,Y)
            
            info = '_splitCondition='+str(split_cond) + '_maxFeatures='+str(max_feat) + '_stopWords='+str(stp_wrds) #label for filename with info on parameters used
            dot_data = tree.export_graphviz(clf, out_file='resources/part7/tree_data/max_dep='+str(dep)+info+'.dot' )
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
        
        filename = 'resources/part7/part7a'+info+'.jpg'
        e = len(max_depths) - 1
        plt.scatter(max_depths[:e], tr_res[:e], label='Training Data')
        plt.scatter(max_depths[:e], va_res[:e], label='Validation Data')
        plt.scatter(max_depths[:e], te_res[:e], label='Testing Data')
        plt.title('Learning Curve')
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        plt.legend()
        #plt.show()
        plt.savefig(filename)
        plt.close()


    ## Part 8    
    word = 'donald'
    I = mutual_info(word, y_tr)

    index = rd.randint(0, len(all_words) )
    word = all_words[index]
    I = mutual_info(word, y_tr)
    

    