"""
@author: johnn
"""
import DigitalTwinDataGenerator as gen
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

gen = gen.DataGen()
test = gen.generate(1000)
plt.scatter(test['k'], test['x_t'], label = "x(t)")
plt.scatter(test['k'], test['Fatigue']/max(test['Fatigue']), label = "Fatigue")
plt.xlabel("k")
plt.legend()

