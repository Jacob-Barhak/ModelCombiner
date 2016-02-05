##############################################################################
#Copyright (C) 2015 Jacob Barhak, Aaron Garrett
# 
#This file is part of the Model Combiner . The Model Combiner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#The Model Combiner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#See the GNU General Public License for more details.
#############################################################################

from __future__ import division
import sys
import random
import inspyred
import numpy
import types
import matplotlib.pyplot as plt
import statsmodels.api as sm


        
# Define infinite
Inf = float('inf')

def GetSampleVectorFromSupport(Support,random):
    "Convert Support expression to sample vector is necessary"
    if type(Support[0]) == types.TupleType:
        # if a tuple starts the support this means generate random numbers
        # to match the support
        [(Low,High),NumberOfSamples] = Support
        SampleVector = [random.uniform(Low,High) for Entry in range(NumberOfSamples)]
    else:
        # otherwise just follow the points given by the user
        SampleVector = Support
    return SampleVector



def FullEvaluation(Candidate,Functions,Support,NoiseLevel,random):
    "Calculate difference between candidate solution and base function"
    # first check what kind of support is given:
    SampleVector = GetSampleVectorFromSupport(Support,random)
    # now calculate errors in these locations
    ErrVec = []
    FunctionEvaluationMatrix = []
    for x in SampleVector:
        FunctionEvaluations = [Func(x) for Func in Functions]
        Solutions = [FunctionEvaluations[CoeffEnum]*Coeff for (CoeffEnum,Coeff) in enumerate(Candidate + [-1])]
        SolutionsWithNoise = [Entry + NoiseLevel*random.gauss(0,1) for Entry in Solutions]
        Err = sum(SolutionsWithNoise)
        ErrVec.append(Err)
        FunctionEvaluationMatrix.append(FunctionEvaluations)
    ErrMean = numpy.mean(ErrVec)
    ErrStd = numpy.std(ErrVec)
    ErrMin = min(ErrVec)
    ErrMax = max(ErrVec)
    ErrNorm = numpy.linalg.norm(ErrVec)
    Fitness = ErrMean**2
    return  (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector)
    




def GradientDescentVariate(Solution, StepSize, Bounds):
    """ Variate Solution in Dimsnsion Dim considering Bounds """
    NewSolution = Solution[:]
    for Dim in range(len(Solution)):
        if Bounds == None:
            (Low,High) = (-Inf,Inf)
        else:
            (Low,High) = Bounds[Dim]
        NewSolution[Dim] = min(High, max(Low, NewSolution[Dim] + StepSize[Dim]))
        
    return NewSolution


def SolveProblemUsingGradientDescent(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, DerivativeStepSize, IterationStepSize, MaxIterations, StopThreshold, InitialGuess):
    """ Solve the problem using Gradient Descent within bounds """
    # Initial condition
    Solution = InitialGuess
    PathToSolution = []
    
    # Create Step Sizes for each 
    DerivativeStepVectors = []
    for Dim in range(len(Bounds)):
        DerivativeStepVector = [(Enum == Dim)*DerivativeStepSize for (Enum,Entry) in enumerate(Bounds)]
        DerivativeStepVectors.append(DerivativeStepVector)
    
    PreviousResult = Inf
    for Iteration in range(MaxIterations):
        (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = FullEvaluation(Solution,Functions,Support,NoiseLevel,RandomGeneratorToUse)
        StatsToOutput = (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector)
        CurrentResult = Fitness
        PathToSolution.append((Solution,CurrentResult))
        # if solution become worst than a given threshold then stop
        if abs(PreviousResult - CurrentResult) < StopThreshold:
            # Note that iterations stop when results stopm improving 
            break
        # uncomment for debug prints per iteration
        #else:
        #    print Iteration, CurrentResult
        #    print Solution

        # Compute Gradient
        Gradient = []
        for Dim in range(len(Bounds)):
            PerturbedSolution = GradientDescentVariate(Solution, DerivativeStepVectors[Dim], None)
            (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = FullEvaluation(PerturbedSolution,Functions,Support,NoiseLevel,RandomGeneratorToUse)
            PerturbedResult = Fitness
            Gradient.append((PerturbedResult-CurrentResult)/DerivativeStepSize)
        # Normalize gradient
        GradientSize = (sum([Entry*Entry for Entry in Gradient]))**0.5
        NormalizedGradient = [Entry/GradientSize for Entry in Gradient]
        StepSize = [-Entry*IterationStepSize for Entry in NormalizedGradient]
        # Now update the solution:
        NewSolution = GradientDescentVariate(Solution, StepSize, Bounds)
        Solution = NewSolution
        PreviousResult = CurrentResult
    return (Solution, Iteration, CurrentResult, StatsToOutput, PathToSolution)

def PrepareDataForRegressionLinearModel( Functions, Support, NoiseLevel, RandomGeneratorToUse):
    "Prepare data for regression by evaluating functions"
    # y = b0*x0 + b1*x1 + ... + bn*xn
    # m is number of data points collected
    # n+1 is dimentionality of the problem
    SampleVector = GetSampleVectorFromSupport(Support,RandomGeneratorToUse)
    b = numpy.empty([len(SampleVector), 1])
    A = numpy.empty([len(SampleVector), len(Functions)-1])
    for i, x in enumerate(SampleVector):
        for j, f in enumerate(Functions[:-1]):
            A[i, j] = f(x)
        # Noise is added to the solution
        b[i, 0] = Functions[-1](x) + NoiseLevel* random.gauss(0,1)
    return (SampleVector, A, b)

    
def SolveUsingRegressionLinearModel(A, b):
    "Solve simple combination problem using linear regression"
    sm.add_constant(A)
    ProblemSolutionObject = sm.OLS(b, A)
    ProblemSolution = ProblemSolutionObject.fit()
    return ProblemSolution
    
def PlotResults(SampleVector, ProblemSolution, A, b):
    "Output results to screen and plot"
    print(ProblemSolution.summary())
    b_hat = ProblemSolution.predict(A)
    plt.scatter(SampleVector, b, alpha=0.3)  
    plt.plot(SampleVector, b_hat, 'r', alpha=0.9)    
    plt.show()

def SolveProblemUsingRegressionLinearModel(Functions, Support, NoiseLevel, RandomGeneratorToUse):
    """ Solve the problem using Regression Linear model """
    (SampleVector, A, b) = PrepareDataForRegressionLinearModel(Functions, Support, NoiseLevel, RandomGeneratorToUse)
    ProblemSolution = SolveUsingRegressionLinearModel(A, b)
    return (SampleVector, ProblemSolution, A, b)
    


### Functions needed for EC
def Generator(random, args):
    "Generate solutions"
    Bounds = args['Bounds']
    # coefficients are generated in between bounds given for each function
    GeneratedCoefficients = [random.uniform(Low,High) for (Low,High) in Bounds]
    return GeneratedCoefficients

@inspyred.ec.evaluators.evaluator
def Evaluator(Candidate, args):
    "evaluate candidtes by calling the full evaluation function"
    # First recreate the team from the swaps
    Functions = args['Functions']
    Support = args['Support']
    random = args['_ec']._random
    # use full evaluation function
    (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = FullEvaluation(Candidate,Functions,Support,NoiseLevel,random)
    return Fitness



def SolveProblemUsingSimulatedAnealing(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, MaxEvaluations, MutationRate, GaussianSTD):
    "Solve aggregate problem using Simulated annealing"
    # Mutation infroamtion is:
    # RoughMutationRate,FineMutationRate,LowMultationMultiplier,HighMutationMultiplier    
    #MutationInformation = (0.1, 0.95, 1.05)

    TransposedBounds = map(None,*Bounds)

    # Uncomment below to solve with Simulated annealing
    ea = inspyred.ec.SA(RandomGeneratorToUse)
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    FinalPopulation = ea.evolve(evaluator=Evaluator, 
                          generator=Generator, 
                          maximize=False,
                          bounder=inspyred.ec.Bounder(*TransposedBounds),
                          max_evaluations=MaxEvaluations,    
                          mutation_rate=MutationRate,
                          gaussian_stdev=GaussianSTD,
                          Functions=Functions,
                          Support = Support,
                          Bounds = Bounds,
                          NoiseLevel = NoiseLevel)
    
    FinalPopulation.sort(reverse=True)
    BestCandidate = FinalPopulation[0]
                                                                                                                                         
    (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = FullEvaluation(BestCandidate.candidate,Functions,Support,NoiseLevel,RandomGeneratorToUse)
    return ea, Fitness, FinalPopulation, BestCandidate, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec


def PrintResults(FunctionTexts,Support,NoiseLevel,Bounds,RandomSeed,ComuptationResults):
    "Output results nicely"    
    if len(ComuptationResults) == 5 :
        (Solution, Iteration, CurrentResult, StatsToOutput, PathToSolution) = ComuptationResults
        (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = StatsToOutput
        FinalSolution = Solution
    else:
        (ea, Fitness, FinalPopulation, BestCandidate, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec) = ComuptationResults
        FinalSolution = BestCandidate.candidate
    print ('#'*70)
    print ('#'*70)
    print ('Functions:')
    for FunctionText in FunctionTexts[:-1]:
        print ('Base Function: ' + FunctionText)
    print ('Target Function: ' + FunctionTexts[-1])
    print ('Bound Conditions were: ' + str(Bounds)) 
    print ('Noise Level was: ' + str(NoiseLevel)) 
    print ('Function Support interval was: ' + str(Support)) 
    print ('Err Mean = ' + str(ErrMean))
    print ('Err STD = ' + str(ErrStd))
    print ('Err Min = ' + str(ErrMin))
    print ('Err Max = ' + str(ErrMax))
    print ('Err Norm = ' + str(ErrNorm))
    print ('Fitness = ' + str(Fitness))
    print ('Final Coefficients are: ' + str(FinalSolution))




if __name__ == '__main__':
    # By default reproduce paper results
    Method = "PAPER"
    # Gradient Descent
    # Method = "GRAD"
    # Other Methods are 
    # Regression
    # Method = "REG"
    # Evolutionary Computation
    # Method = "EC"

    #FunctionTexts = ['1','x','x**2','0.1+0.2*x+0.3*x**2']
    #FunctionTexts = ['1','x','x**8','0.1+0.2*x+0.3*x**8']
    # To solve a problem with different scales uncomment the next line
    FunctionTexts = ['1','x','x**2','0.1+0.2*x+0.3*x**2']
    # To solve a 2D problem uncomment the next line
    #FunctionTexts = ['x**5','x**3','0.1*x**5+0.2*x**3']
    # To solve a 1D problem uncomment the next line
    #FunctionTexts = ['x**4','0.1*x**4']

    # by default 100 points between 2 and 3
    #Support = [(2,3),100] # Just did this myself below.
    # one can use this nice set instead
    Support = [2+0.01*Entry for Entry in range(100)]
    # for debug purposes use this:
    # Support = [2]

    NoiseLevel = 0
    Bounds = [(0,1)]*(len(FunctionTexts)-1)
    RandomSeed = 0
    DerivativeStepSize = 0.001
    IterationStepSize = 0.01
    MaxIterations = 500
    StopThreshold = 0.001
    InitialGuess = [0 for (Low,High) in Bounds]
    MaxEvaluations = 1000
    MutationRate = 1
    GaussianSTD = 0.05

    Args = sys.argv
    if len(Args) >= 2:
        Method = (Args[1])

    if not Method.startswith('P'):
        if len(Args) >= 3:
            FunctionTexts = eval(Args[2])
        if len(Args) >= 4:
            Support = eval(Args[3])
        if len(Args) >= 5:
            NoiseLevel = eval(Args[4])
        if len(Args) >= 6:
            Bounds = eval(Args[5])
        if len(Args) >= 7:
            RandomSeed = eval(Args[6])

        
        if Method.startswith('G'):
            # Control variables for Gradient Descent
            if len(Args) >= 8:
                DerivativeStepSize = eval(Args[7])
            
            if len(Args) >= 9:
                IterationStepSize = eval(Args[8])
    
            if len(Args) >= 10:
                MaxIterations = eval(Args[9])
    
            if len(Args) >= 11:
                StopThreshold = eval(Args[10])
                
            if len(Args) >= 12:
                InitialGuess = eval(Args[11])
            else:
                InitialGuess = [0 for (Low,High) in Bounds]
    
    
        if Method.startswith('E'):
            # Control variables for Gradient Descent
            if len(Args) >= 8:
                MaxEvaluations = eval(Args[7])
    
            if len(Args) >= 9:
                MutationRate = eval(Args[8])
    
            if len(Args) >= 10:
                GaussianSTD = eval(Args[9])


    # Generate the functions    
    Functions = []
    for FunctionText in FunctionTexts:
        Func = eval ("lambda x : " + FunctionText)
        Functions.append(Func)

    # Set the random seed
    random.seed(RandomSeed)
    RandomGeneratorToUse = random

    # Gradient Descent
    if Method.startswith('G'):
        AllGradientDescentResults = SolveProblemUsingGradientDescent(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, DerivativeStepSize, IterationStepSize, MaxIterations, StopThreshold, InitialGuess)
        PrintResults(FunctionTexts,Support,NoiseLevel,Bounds,RandomSeed,AllGradientDescentResults)
    if Method.startswith('R'):
        (SampleVector, ProblemSolution, A, b) = SolveProblemUsingRegressionLinearModel(Functions, Support, NoiseLevel, RandomGeneratorToUse)
        PlotResults(SampleVector, ProblemSolution, A, b)
    if Method.startswith('E'):
        AllEvolutionaryComuptationResults = SolveProblemUsingSimulatedAnealing(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, MaxEvaluations, MutationRate, GaussianSTD)
        PrintResults(FunctionTexts,Support,NoiseLevel,Bounds,RandomSeed,AllEvolutionaryComuptationResults)
    if Method.startswith('P'):
        # Reproduce paper results
        PopulationSizesPerPlot = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        SupportTypes = ['Fixed','Random']
        SupportTypeColors = ['b','r']
        NoiseLevels = [0, 0.01, 0.1, 1, 10]
        NoiseMarkers = ['.','+','^','*','o']
        NoiseLines = ['-','--','-.',':',' ']
        for Method in ['Regression','Gradient Descent','Evolutionary Computation']:
            # setup the plot
            Fig = plt.figure()
            Axis1 = Fig.add_subplot(111)
            Axis1.set_title(Method)
            Axis1.set_xlabel('Population Size')
            Axis1.set_ylabel('Fitness')
            Axis1.set_yscale('log')
            # Try all three methods
            for (SupportTypeEnum,SupportType) in enumerate(SupportTypes):
                # Rotate support types
                for (NoiseLevelEnum,NoiseLevel) in enumerate(NoiseLevels):
                    FitnessPlotValues = []
                    for PopulationSize in PopulationSizesPerPlot:
                        if SupportType == 'Fixed':
                            Support = [2+1/PopulationSize*Entry for Entry in range(PopulationSize)]
                        elif SupportType == 'Random':
                            Support = [(2,3),PopulationSize]
                        if Method.startswith('G'):
                            AllGradientDescentResults = SolveProblemUsingGradientDescent(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, DerivativeStepSize, IterationStepSize, MaxIterations, StopThreshold, InitialGuess)
                            (Solution, Iteration, CurrentResult, StatsToOutput, PathToSolution) = AllGradientDescentResults
                            (Fitness, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec, SolutionsWithNoise, FunctionEvaluationMatrix, SampleVector) = StatsToOutput
                        if Method.startswith('R'):
                            (SampleVector, ProblemSolution, A, b) = SolveProblemUsingRegressionLinearModel(Functions, Support, NoiseLevel, RandomGeneratorToUse)
                            Solution = list(ProblemSolution.params)
                            Fitness = ((Solution[0]-0.1)**2 + (Solution[1]-0.2)**2 + (Solution[2]-0.3)**2)
                            Iteration = 0
                        if Method.startswith('E'):
                            AllEvolutionaryComuptationResults = SolveProblemUsingSimulatedAnealing(Functions, Support, NoiseLevel, Bounds, RandomGeneratorToUse, MaxEvaluations, MutationRate, GaussianSTD)
                            (ea, Fitness, FinalPopulation, BestCandidate, ErrMean, ErrStd, ErrMin, ErrMax, ErrNorm, ErrVec) = AllEvolutionaryComuptationResults
                            FinalSolution = BestCandidate.candidate
                            Iteration = 0
                        print Method, SupportType, NoiseLevel, PopulationSize, Fitness, Iteration, Solution
                        FitnessPlotValues.append(Fitness)
                        # Now plot the results
                    plt.plot(PopulationSizesPerPlot, FitnessPlotValues, c = SupportTypeColors[SupportTypeEnum], marker = NoiseMarkers[NoiseLevelEnum], linestyle = NoiseLines[NoiseLevelEnum], label = SupportType + ' Noise =' + str(NoiseLevel), markersize = (NoiseLevelEnum+1)*2)
            Box = Axis1.get_position()
            Axis1.set_position([Box.x0, Box.y0 +  Box.height*0.2, Box.width , Box.height*0.8])
            # Put a legend to the right of the current axis
            Axis1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=2, fontsize=9)
            plt.savefig(Method.replace(' ','')+'.png')
            # uncomment to show plot on screen
            #plt.show()
