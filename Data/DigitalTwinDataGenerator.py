# -*- coding: utf-8 -*-
"""
@author: johnn
"""
import random
import pandas as pd
import csv

class DataGen:
    def __init__(self):
        self.fields = {"k":[], 
                              "x_t":[], 
                              "Component A":[], 
                              "Component B":[],
                              "Fatigue":[]}
        self.sigFig = 10
        
        #n = number of data points
        #kLower = lower bound of k
        #Kupper = upper bound of k
    def generate(self, n = 1000, kLower = 0, kUpper = 4,):
        output = pd.DataFrame(self.fields)
        i = 0
        while i < n:
            j = 0
            k = kLower + random.random() * (kUpper - kLower)
            x_t = random.random() # x_t is x(t)
            fat = 0     #fat is the fatigue
            compA = 1 - x_t  #compA is xA(t)
            compB = x_t * k  #compB is xB(t)
            while j < 100:
                compA = 1 - x_t
                compB = x_t * k
                fat = fat + abs(compA*compB - x_t)
                x_t = compA * compB
                j += 1  
            output.loc[len(output.index)] = [round(k, self.sigFig), round(x_t, self.sigFig), round(compA, self.sigFig), round(compB, self.sigFig), round(fat, self.sigFig)]
            i+=1
        return output
        
    def generateCSV(self, fileName, n = 1000, kLower = 0,  kUpper = 4):
        f = open(fileName, 'w+')
        writer = csv.writer(f)
        writer.writerow(self.fields)
        i = 0
        while i < n:
            j = 0
            k = kLower + random.random() * (kUpper - kLower)
            x_t = random.random() 
            fat = 0
            compA = 1 - x_t
            compB = x_t * k
            while j < 100:
                compA = 1 - x_t
                compB = x_t * k
                fat = fat + abs(compA*compB - x_t)
                x_t = compA * compB
                j += 1  
            row = [round(k, self.sigFig), round(x_t, self.sigFig), round(compA, self.sigFig), round(compB, self.sigFig), round(fat, self.sigFig)]
            writer.writerow(row) 
            i+=1
        f.close()
        