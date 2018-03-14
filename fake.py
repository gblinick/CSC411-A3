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


#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')
#os.chdir('\Users\Gideon\Desktop\U of T\Year 4\Term2\CSC411\A3\CSC411-A3')




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
    
    