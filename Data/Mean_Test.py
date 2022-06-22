# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:05:47 2022

Testing Scientific and Harmonic Mean of Oscillating Datasets
@author: oelkassabany
"""

# Context: We want to calculate the harmonic and scientific mean of each interval of K
    
# Between K [0,1], x(t) steady state is 0 

# Between K [1,3], x(t) steady staete is between 0 and 0.65 

# Between K [3, 3.499], x(t) oscillates between 2 values (periodic cycle of 2) 

# Between K [3.5, 3.544], x(t) oscillates between 4 values (periodic cycle of 4) 

# Between K [3.55, 3.599], x(t) oscillates between continually doubling values (periodic cycle of 8, 16, 32, etc.) 

# Between K [3.6, 4], x(t) is random between 0 and 1 

# We will 1) generate datasets of each range of K we know values of, then 2) calculate Pythagorean
# means of each dataframe's x(t) values.
# Reference article: https://towardsdatascience.com/on-average-youre-using-the-wrong-average-geometric-harmonic-means-in-data-analysis-2a703e21ea0

import DigitalTwinDataGenerator as gen
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gmean
from scipy.stats import hmean
gen = gen.DataGen()

# Generate datasets
x01 = gen.generate(n=1000, kLower = 0, kUpper = 0.999)
x13 = gen.generate(n=1000, kLower = 1, kUpper = 2.999)
x335 = gen.generate(n=1000, kLower = 3, kUpper = 3.499)
x35355 = gen.generate(n=1000, kLower = 3.5, kUpper = 3.544)
x3536 = gen.generate(n=1000, kLower = 3.55, kUpper = 3.599)
x364 = gen.generate(n=1000, kLower = 3.6, kUpper = 4)

# Pull out x_t values
data = np.array([x01['x_t'],x13['x_t'],x335['x_t'],x35355['x_t'],x3536['x_t'],x364['x_t']])
data = data.transpose()

# Create aggregated dataframe
column_names = ["K Range", "Min", "Max", "Median", "Arithmetic Mean", "Geometric Mean", "Harmonic Mean"]
agg = pd.DataFrame(columns = column_names)
k_ranges = ["K[0,1)", "K[1,3)", "K[3,3.5)", "K[3.5,3.55)","K[3.55,3.6)","K[3.6,4)"]
agg["K Range"] = k_ranges

# Populate aggregated table with means
for i in range(0,5):
    agg["Min"][i] = min(data[i])
    agg["Max"][i] = max(data[i])
    agg["Median"][i] = np.median(data[i])
    agg["Arithmetic Mean"][i] = np.mean(data[i])
    agg["Geometric Mean"][i] = gmean(data[i])
    agg["Harmonic Mean"][i] = hmean(data[i])
    
print(agg)
