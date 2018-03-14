import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import re
from collections import Counter
import os


#script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#rel_path = "resources/clean_fake.txt"
#abs_file_path = os.path.join(script_dir, rel_path)

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
    
    # part below almost right - have to get it so doesn't change the Counters themselves
    #diff1 = word_counts_fake.subtract(word_counts_real)
    #diff2 = word_counts_real.subtract(word_counts_fake)
    
    
    
    ##
    '''This is what Ariel stupidly did (in addition to the functions above)'''
    fake_lines = get_data('resources/clean_fake.txt') #Get list containing headlines
    real_lines = get_data('resources/clean_real.txt')
    fake_stats = get_stats(fake_lines) #compute probabilities for each word
    real_stats = get_stats(real_lines)
    
    #Part 1
    fake_keywords = top_keywords(fake_stats, 10)
    real_keywords = top_keywords(real_stats, 10)
    