import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import math
import time

from sklearn import tree
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project3/CSC411-A3')


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

def add_words(dict1, dict2):
    missing = { x:0 for x in dict1.keys() if x not in dict2.keys() }
    dict2.update( missing )
    
    missing = { x:0 for x in dict2.keys() if x not in dict1.keys() }
    dict1.update( missing )
    return dict1, dict2

def part_1(dict1, dict2):   
    ''' return a list of the top 10 words by percentage that appear in dict1 over dict2'''
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
    y_tr = [1]*len(training_set) #1 represents fake
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


    


def training_part(fake_lines_training_set, real_lines_training_set, m, p):
    '''Takes in a list of training set lines divided into real and fake groups.
    Returns a list of probabilities that represent p(word|real) and p(word|fake).
    this list will be 5833 long to account for every word. '''
    
    #Part 1: Training: calculate p(real), p(fake), and p(word|real), p(word|fake)
    fake_stats_training_set = get_stats(fake_lines_training_set) #compute probabilities for each word
    real_stats_training_set = get_stats(real_lines_training_set)
    
    fake_counts_training_set = get_count(fake_lines_training_set) #compute counts for each word
    real_counts_training_set = get_count(real_lines_training_set)
    
    p_fake = len(fake_lines_training_set)/(len(fake_lines_training_set) + len(real_lines_training_set))
    p_real = len(real_lines_training_set)/(len(fake_lines_training_set) + len(real_lines_training_set))
    
        # what we want is p(real|words in headline).
        # this is equal to p(words in headline|real)*p(real), divided by
        #   p(words in headline|real)*p(real) + p(words in headline|fake)*p(fake)
        # We already have p(real) and p(fake) from above, so we just need to find
        # p(words in headline|real) and p(words in headline|fake)
        # p(word in headline|real) is equal to count(word & real) + mp
        # divded by count(real) + m.
        # So we need to choose values for m and p and then come up with a
        # dictionaries of the adjusted probabilities.
    
    all_words = list(real_counts_total.keys())
    
    missing = { x:0 for x in fake_counts_total.keys() if x not in real_counts_training_set.keys() }
    real_counts_training_set.update( missing )
    missing = { x:0 for x in real_counts_total.keys() if x not in fake_counts_training_set.keys() }
    fake_counts_training_set.update( missing )
    
    adjusted_fake_counts_training_set = {} 
    adjusted_real_counts_training_set = {} 
    
    
    for word in all_words:
        adjusted_fake_counts_training_set[word] = fake_counts_training_set[word] + mp
        adjusted_real_counts_training_set[word] = real_counts_training_set[word] + mp
        
    naive_divisor = len(all_words) + m

    adjusted_fake_stats_training_set = {} #P(w | fake)
    adjusted_real_stats_training_set = {} #P(w | real)
    
    for word in all_words:
        adjusted_fake_stats_training_set[word] = adjusted_fake_counts_training_set[word]/naive_divisor
        adjusted_real_stats_training_set[word] = adjusted_real_counts_training_set[word]/naive_divisor
    

    return p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set 

def evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, SET):
    ''' Calculate p(fake|headline) for all headlines in a given set (TRAINING, TEST, VALIDATION).
    Takes p(real), p(fake), p(word|real), p(word|fake) as parameters.'''

    total_real_probabilities = {}
    total_fake_probabilities = {}
    
    all_words = list(real_counts_total.keys())
    
    for headline in SET:
        if len(total_real_probabilities)%50 == 0:
            print(len(total_real_probabilities))
            print(m,mp)
        probabilities_real = []
        probabilities_fake = []
        
        headline_words = list(set( headline.split(' ') )) #converting to set and back to a list removes duplicates
        for word in headline_words:
            probabilities_real.append(adjusted_real_stats_training_set.get(word))
            probabilities_fake.append(adjusted_fake_stats_training_set.get(word))
        non_headline_words = [x for x in all_words if x not in headline_words]
        for word in non_headline_words:
            probabilities_real.append(1 - adjusted_real_stats_training_set.get(word))
            probabilities_fake.append(1 - adjusted_fake_stats_training_set.get(word))
        
        total_real_probability = 0
        total_fake_probability = 0
        for k in range(len(probabilities_real)):
            total_real_probability = total_real_probability + math.log(probabilities_real[k])
            total_fake_probability = total_fake_probability + math.log(probabilities_fake[k])
        total_real_probability = math.exp(total_real_probability)
        total_real_probabilities[headline] = total_real_probability
        total_fake_probability = math.exp(total_fake_probability)
        total_fake_probabilities[headline] = total_fake_probability

    # At this point, we have calculated p(headline|real) and p(headline|fake) for all
    # headlines in whatever set we are testing
    # Now we need to find p(fake|headline) which means multiplying p(headline|fake) by p(fake) and 
    # dividing by p(headline|fake)*p(fake) + p(headline|real)*p(real) 

    #Compute final probabilites
    numerator_fake = [p_fake*x for x in total_fake_probabilities.values()]
    numerator_real = [p_real*x for x in total_real_probabilities.values()]
    
    final_fake = [numerator_fake[i]/(numerator_fake[i] + numerator_real[i]) for i in range(len(numerator_fake))]
    final_real = [numerator_real[i]/(numerator_fake[i] + numerator_real[i]) for i in range(len(numerator_real))]
    return final_fake, final_real

def check_accuracy(final_fake, y):
    '''Given a list of fake probabilities, compare with the actual results by rounding.
    If the fake  probability is greater than 0.5, consider it fake and if less, consider it real.
    Output the accuracy rate'''
    pred_fake = np.array([round(x) for x in final_fake])
    y_2 = np.array(y)
    
    incorrect = np.count_nonzero(pred_fake - y_2)
    total = len(y)
    correct = total - incorrect
    accuracy = correct/total   
    return accuracy



def optimize_mp(fake_lines_training_set, real_lines_training_set, m_s, mp):
    val_acc = {}
    for m in m_s:
        print('Optimizing Naive Bayes with m = ' + str(m) )
        p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set = training_part(fake_lines_training_set, real_lines_training_set, m, mp)
        final_fake, final_real = evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, validation_set)
        val_acc[m] = check_accuracy(final_fake, y_va)
    return val_acc


def forward(theta, x_train):
    '''Logistic Regression forward pass'''
    o = np.dot(x_train, theta)
    return softmax(o)

def softmax(y):
    '''apply softmax pointwise'''
    return 1/(1 + np.exp(-y) )

def NLL(y_, y): 
    #y is output of network, y_ is correct results
    return -np.sum(y_*np.log(y))

def grad(y_, y, x, gamma, theta): 
    '''compute gradient'''
    diff = (y - y_) #y is output of network
    grad_theta = np.sum( (x.T)*diff, 1)  - gamma*(theta)/np.linalg.norm(theta)
    return  grad_theta

def train(data, rate, gamma): #train logistic regression for part 4
    rd.seed(0)
    theta = rd.rand( len(all_words) )
    theta = theta - 0.5
    train_x, train_y, val_x, val_y, test_x, test_y = data
    
    max_iter = 1000
    iterations = [] #for storing x-axis data for plotting
    train_acc = []
    val_acc = []
    test_acc = []
    iter = 0
    
    while iter < max_iter: #Train
        iter += 1
        y = forward(theta, train_x)
        theta = theta - rate*grad(train_y, y, train_x, gamma, theta)
        if iter%50 == 0:
            iterations += [iter]
            train_acc += [1 - np.count_nonzero(np.round(y) - train_y)/len(train_y) ]
            val_pred = forward(theta, val_x)
            val_acc += [1 - np.count_nonzero(np.round(val_pred) - val_y)/len(val_y) ]
            test_pred = forward(theta, test_x)
            test_acc += [1 - np.count_nonzero(np.round(test_pred) - test_y)/len(test_y) ]
    
    # Add final accuracies
    y = forward(theta, train_x)
    train_acc += [1 - np.count_nonzero(np.round(y) - train_y)/len(train_y) ]
    val_pred = forward(theta, val_x)
    val_acc += [1 - np.count_nonzero(np.round(val_pred) - val_y)/len(val_y) ]
    test_pred = forward(theta, test_x)
    test_acc += [1 - np.count_nonzero(np.round(test_pred) - test_y)/len(test_y) ]
    iterations += [iter]
    return iterations, train_acc, val_acc, test_acc, theta


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
    fake_lines_total = get_data('resources/clean_fake.txt') #Get list containing headlines
    real_lines_total = get_data('resources/clean_real.txt')
    
    fake_stats_total = get_stats(fake_lines_total) #compute probabilities for each word
    real_stats_total = get_stats(real_lines_total)
    fake_stats_total, real_stats_total = add_words(fake_stats_total, real_stats_total) #add missing words to each dict

    #   Get counts 

    fake_counts_total = get_count(fake_lines_total) #compute counts for each word
    real_counts_total = get_count(real_lines_total)
    fake_counts_total, real_counts_total = add_words(fake_counts_total, real_counts_total) #add missing words to each dict


    #   Top 10 keywords by percentage
    fake_keywords = top_keywords(fake_stats_total, 10)
    real_keywords = top_keywords(real_stats_total, 10)
    #   Top differences by percentage
    fake_minus_real_top = part_1(fake_stats_total, real_stats_total)
    real_minus_fake_top = part_1(real_stats_total, fake_stats_total)

    #   Divide datasets
    training_set, validation_set, testing_set, y_tr, y_va, y_te = sets(fake_lines_total, real_lines_total)
    tr = get_stats(training_set)
    va = get_stats(validation_set)
    te = get_stats(testing_set)
    
    
#%%
    ## Part 2
        # First, use the training set to determine p(word|real) for all words in the TOTAL set
        # as well as p(real) and p(false) based on just the TRAINING set.
    rd.seed(1)
    rd.shuffle(fake_lines_total)
    rd.shuffle(real_lines_total)


    fake_lines_training_set = fake_lines_total[ :int(round(0.7*len(fake_lines_total))) ]
    real_lines_training_set = real_lines_total[ :int(round(0.7*len(fake_lines_total))) ]
    
    m = 2*5833
    mp=1
    
    
    # find the best m,p

    mp = 1
    m_s = [(1*5833), (2*5833),(3*5833),(4*5833)]
    #val_acc = optimize_mp(fake_lines_training_set, real_lines_training_set, m_s, mp)
    
    # Part A: Use TRAINING set to get p(real), p(fake), p(word|real) and p(word|fake) for ALL words
    p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set = training_part(fake_lines_training_set, real_lines_training_set, m, mp)
 

    # Part B: Calculate p(fake|headline) given a set of headlines and the parameters from the previous step.
    final_fake_train, final_real_train = evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, training_set)
    final_fake_test, final_real_test = evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, testing_set)


    
    # Part C: Check the accuracy of our model
    training_accuracy = check_accuracy(final_fake_train, y_tr)
    testing_accuracy = check_accuracy(final_fake_test, y_te)
    
    print("training accuracy:", training_accuracy*100,"%")
    print("testing accuracy:", testing_accuracy*100,"%")
    
    #testing
    #p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set = training_part(fake_lines_training_set, real_lines_training_set, 10, 5)
    #p_fake2, p_real2, adjusted_fake_stats_training_set2, adjusted_real_stats_training_set2 = training_part(fake_lines_training_set, real_lines_training_set, 10000, 0.5)

    #final_fake_val, final_real_val = evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, validation_set)
    #final_fake_va2, final_real_val2 = evaluate(p_fake, p_real, adjusted_fake_stats_training_set, adjusted_real_stats_training_set, validation_set)

    #val_accuracy = check_accuracy(final_fake_val, y_va)
    #val_accuracy2 = check_accuracy(final_fake_va2, y_va)
    
    #%%
  
    ## Part 3
    # For this part, we want to find p(real|word) - the top 10 percentages
    # p(real|word) = p(word|real)*p(real)/p(word)
    # p(word|real) = adjusted_real_stats_training_set
    
    rd.seed(1)
    rd.shuffle(fake_lines_total)
    rd.shuffle(real_lines_total)

    fake_lines_training_set = fake_lines_total[:int(round(0.7*len(fake_lines_total)))]
    real_lines_training_set = real_lines_total[:int(round(0.7*len(real_lines_total)))]
    training_set2 = fake_lines_training_set + real_lines_training_set
    
    #rd.seed(1)
    #rd.shuffle(fake_lines_total)
    #rd.shuffle(real_lines_total)

    #training_set   = fake_lines_total[:int(round(0.7*len(fake_lines_total)))]
    #training_set.extend(   real_lines_total[:int(round(0.7*len(real_lines_total)))] )
    
    # missing = { x:0 for x in counts_training.keys() if x not in real_stats_training_set.keys() }
    #counts_training.get('zieht')
    #real_stats_training_set.get('zieht')
    
    # p(real) and p(fake)
    p_real = len(real_lines_training_set)/(len(fake_lines_training_set) + len(real_lines_training_set))
    p_fake = len(fake_lines_training_set)/(len(fake_lines_training_set) + len(real_lines_training_set))

    
    # p(word)
    # p(word) = count(number of headlines with word)/count(number of headlines)
    
    ## count(number of headlines)
    divisor = len(training_set2)
    ## count(number of headlines with word) for all words in training set
    #counts_training = get_count(training_set)
    counts_training2 = get_count(training_set2)
    
    p_words = {}
    for word in counts_training2.keys():
        p_words[word] = counts_training2.get(word)/divisor
    
    #p(word|real) without adjustment   
    real_stats_training_set = get_stats(real_lines_training_set)
    #p(word|fake) without adjustment
    fake_stats_training_set = get_stats(fake_lines_training_set)

    real_stats_training_set, fake_stats_training_set = add_words(real_stats_training_set, fake_stats_training_set)
        
    p_realIword = {}
    p_fakeIword = {}
    for word in p_words.keys():
        p_realIword[word] = (real_stats_training_set.get(word)*p_real)/p_words.get(word)
        p_fakeIword[word] = (fake_stats_training_set.get(word)*p_fake)/p_words.get(word)
    
     #   Top 10 keywords by percentage
    p_realIword_top = top_keywords(p_realIword, 2000)
    p_fakeIword_top = top_keywords(p_fakeIword, 2000)
    
        

#%%

    ## Part 4
    all_words = list( fake_stats_total.keys() )
    all_words.sort()
    
    train_x = np.array( dict_to_vec(all_words, training_set) )
    train_y = np.array( y_tr.copy() )
    val_x = np.array( dict_to_vec(all_words, validation_set) )
    val_y = np.array( y_va.copy() )
    test_x = np.array( dict_to_vec(all_words, testing_set) )
    test_y = np.array( y_te.copy() )
    data = (train_x, train_y, val_x, val_y, test_x, test_y)
    
    rates = [1e-3, 1e-4]
    gammas = [0, 0.1, 0.2, 0.5, 1, 5]
    for rate in rates:
        for gamma in gammas:
            print('Starting Training with: rate = ' + str(rate) + ' and gamma = ' + str(gamma) )
            iterations, train_acc, val_acc, test_acc = train(data, rate, gamma)
            
            #Plot learning curve
            info = '_lr='+str(rate) + '_gamma='+str(gamma)
            filename = 'resources/part4/'+info+'.jpg'
            plt.scatter(iterations, train_acc, label='Training Data')
            plt.scatter(iterations, val_acc, label='Validation Data')
            plt.scatter(iterations, test_acc, label='Testing Data')
            plt.title('Learning Curve')
            plt.xlabel('iterations')
            plt.ylabel('accuracy')
            plt.legend()
            #plt.show()
            plt.savefig(filename)
            plt.close()

    ## Part 6
    rate, gamma = 1e-3, 1
    iterations, train_acc, val_acc, test_acc, theta = train(data, rate, gamma)
    num = 10
    
    ind = np.argpartition(theta, -num)[-num:] #get highest num values of theta
    thetas = [ theta[i] for i in ind]
    max_thetas = sorted( zip(thetas, ind), reverse=True) #list of tuples ( theta[i], i)
    print('Highest - ')
    for tuple in max_thetas:
        print( all_words[tuple[1]] + ': ' + str(tuple[0]) )
    
    ind = np.argpartition(-theta, -num)[-num:] #get lowest num values of theta
    thetas = [ theta[i] for i in ind]
    max_thetas = sorted( zip(thetas, ind), reverse=True)
    print('\nLowest - ')
    for tuple in max_thetas:
        print( all_words[tuple[1]] + ': ' + str(tuple[0]) )



    ## Part 7
        #note that this entire section was run with the rd.seed() in the sets() function as rd.seed(1)
    rd.seed(0)  #numpy randomness used internally of sklearn.tree
    max_depths = [2, 3, 5, 10, 15, 20, 35, 50, 75, 100, None]
    max_feats = [3, 10, 15, None] #max_features
    all_words = list( fake_stats_total.keys() )
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
    

    