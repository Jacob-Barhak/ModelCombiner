Model Combiner 
==============
This project tests ways to solve an optimization problem of linearly combining base functions to fit a target function. The intent is to imitate combination of models through a simple example with some analogies.
This file accompanies the paper "Optimizing Model Combinations" published in MODSIM 2016. 

Problem definition:

Given a set of base functions: y1(xi), y2(x), .. yn(x)
And a target function Y(x)
Compute a set of coefficients a1,a2,...an to minimize the fitness function Sum(Diff(x_i))^2:
where:
Diff(x) = Y(x) - a1*y1(x) + a2*y2(x) + ... + an*yn(x) + Err
and x_i are sample points from the interval (x_Low,x_High).

We also add random error to the functions to simulate Monte-Carlo error.

We will use several solutions:
* Linear regression - assuming that individual results y_i are available
* Gradient descent
* Evolutionary Computation or specifically Simulated Annealing - we will use Inspyred library for that


EXAMPLE:
--------
A simple linear problem demonstrating this is:
y1(x) = 1
y2(x) = x
y3(x) = x**2

Y(x) = 0.1 + 0.2*x + 0.3*x**2 

The ideal solution for this would be
a1 = 0.1
a2 = 0.2
a3 = 0.3

However, considering that we aggregate sample point results, the problem has two degrees of freedom and we can get close to the solution by finding a point on the hyperplane that includes the solution. 

The regression solution shows how to reach the exact solution. While Gradient Descent and Simulated Annealing are used to show convergence to the optimum hyperplane.


USAGE:
------
python ModelCombine.py [SolutionType] [Functions] [Support] [NoiseLevel] [Bounds] [RandomSeed] [...]

* SolutionType - is either R for regression, G for Gradient Descent, E - for Evolutionary Computation. The default is R.
* Functions -  A list enclosed in brackets of comma separated function strings with the variable x. The last function is the target function. The others are base functions. The default is ['1','x','x**2','0.1+0.2*x+0.3*x**2']
* Support - either: 1) a list of comma separated x values for support such as [2, 2.01, 2.02, ... 2.99] which is the default or 2) a list containing a tuple and a and number of the format [((x_Low,x_High), n] instructing the program to randomly generate n points in the interval [x_Low,x_High].
* NoiseLevel - a number indicating the level of Gaussian random noise to be added to Err for each point calculated
* Bounds - Constraints for each solution coefficient for defined as a list or tuples indicating (Low,How) bounds. The default is : [(0,1),(0,1),(0,1)] . Note that this does not affect regression results.
* RandomSeed - A random seed for all computations. The default is 0. Uses 'None' for time based random generation.


The rest of the parameters are by solution type:

### Gradient Descent parameters:
* DerivativeStepSize: Step size to calculate gradient components using first order forward approximation. The default is: 0.0001
* IterationStepSize: Step size to move each iteration in the gradient direction. The default is: 0.01
* MaxIterations: Maximum number of iterations. The default is: 1000
* StopThreshold: Stop if improvement difference is lower than : 0.01
* InitialGuess: Starting point at iteration 0. The default is: [0,0,0]

### Evolutionary Computation parameters:
* MaxEvaluations: Stop criterion to reduce number of evaluations. The default is: 1000
* MutationRate: Mutation rate of the solution. The default is: 1
* GaussianSTD: The standard deviation used in the Gaussian function. The default is: 0.05


INSTALLATION & DEPENDENCIES:
----------------------------
To install:
1. Copy ModelCombine.py to a directory of choice
2. Install Anaconda from https://www.continuum.io/downloads
3. install Inspyred: pip install inspyred

Dependant libraries are: Inspyred, numpy, statsmodels, matplotlib

It is recommended you use Anaconda, yet other python environments should work as well
This code was tested on Windows 7 Python 2.7.11 and Anaconda 2.4.1 (64-bit) with Inspyred 1.0.




EXAMPLES:
---------

To reproduce the results in the paper results type the following:
python ModelCombine.py 

To solve the 3 function problem above using Gradient Descent type:
python ModelCombine.py G

To solve using Evolutionary Computation type:
python ModelCombine.py E

To solve the non aggregate linear problem using regression type:
python ModelCombine.py R


For Gradient Descent with another support vector and more convergence parameters type:
python ModelCombine.py G ['1','x','x**2','0.1+0.2*x+0.3*x**2'] [(2,3),10] 0 [(0,1),(0,1),(0,1)] 0 0.001 0.01 500 0.001




FILES:
------
ModelCombine.py : Main file with calculations
Readme.md : The file that you are reading now


VERSION HISTORY:
----------------
Development started on 3-Aug-2015.
Uploaded to Github on 5-Feb-2016 - no version number assigned


DEVELOPER CONTACT INFO:
-----------------------

Please pass questions to:

Jacob Barhak Ph.D.
Email: jacob.barhak@gmail.com
http://sites.google.com/site/jacobbarhak/



ACKNOWLEDGEMENTS:
-----------------
Thanks to W. Andrew Pruett for the discussions that helped write the paper.


LICENSE
-------

Copyright (C) 2015 Jacob Barhak, Aaron Garrett
 
This file is part of the Model Combiner . The Model Combiner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The Model Combiner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.
