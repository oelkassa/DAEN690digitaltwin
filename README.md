# George Mason University DAEN 690 Capstone Project - Digital Twins
Modern systems are composed of tightly coupled interacting components. Faulty initial conditions can have emergent interactions that create significant issues without a registered component malfunction. System validation entails identifying scenarios that can generate hazardous emergent interactions, and system engineers use a digital twin to simulate possible initial conditions to recreate these hazardous interactions in a safe environment to predict their onset. One approach to creating the digital twin is building a mid-fidelity model of the system components. In this experiment, two components interact to produce an output that is cycled back to the system, yielding a system that models steady-state and chaotic behavior based on an input growth factor. This system behavior was modelled using deep learning neural network (DLNN) techniques to predict chaotic behavior given initial conditions supplied to the digital twin. This analysis was performed by building a simulation of the system, generating a complete set of system behaviors, developing a DLNN using the full dataset and further validating the model with a partial “holdout” set of data and reporting DLNN performance. performance decreased as the growth factor increased and possible outputs increased in variance.

# Install Requirements
Python:
The Python portion of this project utilizes the following libraries TensorFlow, Numpy, MatplotLib, Pandas, and Skearn. In order to run the python code all of the libriaries as well as Python must be installed.

# Code
Python:
Data for this project is produced using the functions found in DigitalTwinDataGenerator.py. This class is used throughout most of the python code. The example results from this data generator can be found in the data tab. The results from this data generator were tested and the results of these tests can be found in GeneratorTester.py. The DLNNTesting.py class includes some preliminary testing for creating models. Our final results can be found in HyperPerameterTests.py and ModelGraphs.py. Both of these rely on the ModelGenerator.py for functions to create nural networks.

# Executing Code
Python: 
In order to run this code we used Spyder, which is an open source IDE (integrated developement enviornment), however, any IDE should work. The DigitalTwinDataGenerator.py and ModelGenerator.py class should be run before running any other code.

# Data
Data for this project was generated using the DigitalTwinDataGenerator.py class. This class simulates 100 interactions of two hypothetical components. The number of datapoints for a dataset can be specified, as well as the output format (csv or a Pandas dataframe) and starting growth constant. The starting condition for this system is randomized. For each datapoint the initial starting condition(x(0)), growth constant(k), final output(x(t)), is recorded. Additionally the ending conditions for both components and a fatigue value are given, however these values were not used in our project.
