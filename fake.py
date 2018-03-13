import os

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import re
from collections import Counter

#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')
os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')




if __name__ == "__main__":
    with open('\resources\clean_fake.txt') as f:
        passage_fake = f.read()
    
    words_fake = re.findall(r'\w+', passage_fake)
    cap_words_fake = [word.upper() for word in words_fake]
    word_counts_fake = Counter(cap_words_fake)
    word_counts_fake.most_common(3)
    
    '''with open('clean_fake.txt') as f:
        passage_fake = f.read()
    
    words_fake = re.findall(r'\w+', passage_fake)
    cap_words_fake = [word.upper() for word in words_fake]
    word_counts_fake = Counter(cap_words_fake)
    word_counts_fake.most_common(3)
    '''
    