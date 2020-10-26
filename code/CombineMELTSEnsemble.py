# GenMELTSEnsemble
# Written 2019 by Zack Gainsforth 
# The purpose of this file is to take inputs and generate a bunch of MELTS calculations
# to explore some parameter space (we'll call it an ensemble here).
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
import deepdish as dd
import itertools
import pandas as pd
from collections import Mapping, Container 
from sys import getsizeof
import h5py

def deep_getsizeof(o, ids): 
    """Find the memory footprint of a Python object
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    
    From: https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, str):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r      

ParameterizedInputs = None
def ReadInAllOutputs(ComputeScratchSpace, DataGrid):
    """ Goes through all the completed MELTS calculations and loads in the results to make a
    monster DataGrid.
        Inputs: 
            ComputeScratchSpace (str): Path root for where all the MELTS calculations reside.
            DataGrid (pd.dataframe): Holds the parameterized space on input.
        Outputs:
            DataGrid (pd.dataframe): Now populated with all the data.
    """

    # Store the input parameter space for later use.
    global ParameterizedInputs
    ParameterizedInputs =  h5py.File(os.path.join(ComputeScratchSpace, 'ParameterizedInputs.hdf5'), 'r')

    # Add a column to contain the MELTS calculations.
    DataGrid['MELTS'] = None

    for i in range(DataGrid.shape[0]):
        print(f'Loading MELTS data from {DataGrid.iloc[i]["DirName"]}.')
        # MELTSData will be a list of dataframes -- one for each phase that exists in that MELTS computation.
        MELTSData = dict()
        # Read in the phases that exist.
        for PhaseName  in ['PhaseMassPercents', 'CombinedFitIndex', 'Olivine', 'Clinopyroxene', 'Orthopyroxene', 'Plagioclase', 'Orthoclase', 'Liquid', 'Spinel', 'AlloyLiquid', 'AlloySolid']:
            try:
                DataItem = pd.read_csv(os.path.join(ComputeScratchSpace, DataGrid.iloc[i]['DirName'], 'Output_' + PhaseName + '.csv'))
            except FileNotFoundError as e:
                # File not found just means that phase doesn't exist under these conditions of simulation.
                continue
            MELTSData[PhaseName] = DataItem
        DataGrid.at[i, 'MELTS'] = MELTSData

    return DataGrid

def ExtractIndependentAxis(DataGrid, AxisPath):
    if '/' in AxisPath:
        Axis = ExtractMELTSIndependentAxis(DataGrid, AxisPath)
    else:
        global ParameterizedInputs
        Axis = ParameterizedInputs['data'][AxisPath][:]
    return Axis

def ExtractMELTSIndependentAxis(DataGrid, AxisPath):
    Path = AxisPath.split('/')
    # Find out the range of temperatures across all the values of fO2.
    Axis = None
    for i in range(DataGrid.shape[0]):
        try:
            T = np.array(DataGrid.iloc[i][Path[0]][Path[1]][Path[2]])
            # T = np.array(DataGrid.iloc[i]['MELTS']['Olivine']['Temperature'])
        except KeyError as e:
            print(f'{AxisPath} not found.')
        if 'T' in locals():
            if Axis is None:
                Axis = T
            else:
                Axis = np.union1d(Axis,T)
    return Axis

def IndexByPath(Prefix, Path):
    PathStr = ''.join(["['"+s+"']" for s in Path.split("/")])
    return f'{Prefix}{PathStr}'

def Make2DCrossSection(DataGrid, ParameterizedAxisPath, IndependentAxisPath, DependentPath, Plot=True, SavePath=None, Constraints={}):
    ParameterizedAxis = ExtractIndependentAxis(DataGrid, ParameterizedAxisPath)
    IndependentAxis = ExtractIndependentAxis(DataGrid, IndependentAxisPath)
    if ParameterizedAxis is None:
        print(f'Make2DCrossSection: cannot find parameterized axis: {ParameterizedAxisPath}. It is likely that you are trying to draw a plot for a phase that didn\'t occur in the output simulations.')
        return None, None, None
    if IndependentAxis is None:
        print(f'Make2DCrossSection: cannot find independent axis: {IndependentAxisPath}.  It is likely that you are trying to draw a plot for a phase that didn\'t occur in the output simulations.')
        return None, None, None
    CrossSec = np.zeros((len(ParameterizedAxis), len(IndependentAxis)))
    # We should apply constraints here.  No constraints will be required if there is only one independent variable.
    AllParameterizedAxes = list(DataGrid.columns)
    AllParameterizedAxes.remove('MELTS')
    AllParameterizedAxes.remove('DirName')
    for c in AllParameterizedAxes:
        # If this is an unconstrained parameter then choose a default value.
        if c not in Constraints.keys() and c != ParameterizedAxisPath:
            print(f'{c} not in Constraints list.  Assuming value {DataGrid[c][0]}.')
            Constraints[c] = DataGrid[c][0]
    # Now apply the constraints.
    for c, v in Constraints.items():
        DataGrid = DataGrid.loc[DataGrid[c] == v]
    for i in range(DataGrid.shape[0]):
        Parameterizedval = DataGrid.iloc[i][ParameterizedAxisPath]
        Parameterizedidx = np.where(ParameterizedAxis == Parameterizedval)[0][0]
        IndependentAxisPathParts = IndependentAxisPath.split('/')
        try:
            Independentvals = np.array(DataGrid.iloc[i][IndependentAxisPathParts[0]][IndependentAxisPathParts[1]][IndependentAxisPathParts[2]])
        except KeyError as e:
            print(f'{IndependentAxisPathParts} not found.')
        DependentPathParts = DependentPath.split('/')
        try:
            DependentVals = np.array(DataGrid.iloc[i][DependentPathParts[0]][DependentPathParts[1]][DependentPathParts[2]])
        except KeyError as e:
            print(f'{DependentPathParts} not found.')
            if 'Independentvals' in locals():
                DependentVals = np.zeros(len(Independentvals))
        if 'Independentvals' in locals():
            for j, y in enumerate(Independentvals):
                Independentidx = np.where(IndependentAxis == y)[0][0]
                CrossSec[Parameterizedidx, Independentidx] = DependentVals[j]
    # If the user wants this plotted then we do that now.
    if Plot==True:
        plt.figure()
        # Zeros will mess up the scale of the plot.  We want it to autoscale only on the numbered data.
        CrossSec[CrossSec==0] = np.NaN
        # Figure out the label names.
        ParameterizedLabel =  DetermineLabelName(ParameterizedAxisPath)
        IndependentLabel =  DetermineLabelName(IndependentAxisPath)
        Title = DependentPath.split('/')[-2] + ' ' + DependentPath.split('/')[-1] 
        for c, v in Constraints.items():
            Title += f', {c} = {v}'
        Plot2DCrossSection(CrossSec, IndependentAxis, ParameterizedAxis, IndependentLabel, ParameterizedLabel, Title)
        if SavePath is not None:
            plt.savefig(SavePath)
    return ParameterizedAxis, IndependentAxis, CrossSec

def DetermineLabelName(AxisPath):
    # The label is the last items on the AxisPath.  For example, if the path is MELTS/Olivine/Temperature then the axis will be labeled as temp.
    Label = AxisPath.split('/')[-1]
    # Some labels have prettier versions with LaTeX to make it nice in matplotlib.
    if Label == 'fO2':
        Label = 'f$_{O_2}$'
    if Label == 'Temperature':
        Label = 'Temperature $^{\circ}$C'
    if Label == 'Pressure':
        Label = 'Pressure (Bar)'
    if Label == 'SiO2':
        Label = 'SiO$_2$ (wt.%)'
    if Label == 'MgO':
        Label = 'MgO (wt.%)'
    if Label == 'Al2O3':
        Label = 'Al$_2$O$_3$ (wt.%)'
    if Label == 'Fe2O3':
        Label = 'Fe$_2$O$_3$ (wt.%)'
    if Label == 'K2O':
        Label = 'K$_2$O (wt.%)'
    if Label == 'Na2O':
        Label = 'Na$_2$O (wt.%)'
    if Label == 'CaO':
        Label = 'CaO (wt.%)'
    return Label

def Plot2DCrossSection(CrossSec, XAxis, YAxis, XAxisLabel=None, YAxisLabel=None, Title=None, SavePath=None):
    plt.imshow(CrossSec, origin='lower', extent=[XAxis[0], XAxis[-1], YAxis[0], YAxis[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel(XAxisLabel)
    plt.gca().invert_xaxis()
    plt.ylabel(YAxisLabel)
    plt.title(Title)
    if SavePath is not None:
        plt.savefig(SavePath)

    
if __name__ == "__main__":
    print('------------------------------ START ------------------------------')

    # Location to the alphamelts executable.
    alphaMELTSLocation = os.path.join(os.getcwd(), 'alphamelts')
    # Location to where the compute files are.
    ComputeScratchSpace = os.path.join(os.getcwd(), 'ComputeScratchSpace')

    # Load the inputs, both constant and parameterized
    ConstantInputs = dd.io.load(os.path.join(ComputeScratchSpace, 'ConstantInputs.hdf5'))
    print(ConstantInputs)
    ParameterizedInputs = dd.io.load(os.path.join(ComputeScratchSpace, 'ParameterizedInputs.hdf5'))
    print(ParameterizedInputs)

    # Load the parameterized space to disk so we can reassemble the ensemble.
    DataGrid = pd.read_csv(os.path.join(ComputeScratchSpace, 'ParameterGrid.csv'), index_col=0)

    DataGrid = ReadInAllOutputs(ComputeScratchSpace, DataGrid)

    # Constraints = {'Na2O': 0.0}
    # IndependentVariable = {'fO2': 'fO2'}
    # DependentVariable = {'FitIndex': 'MELTS/Olivine/FitIndex'}
    # PlotTandfO2vsDependent(DataGrid, Constraints, DependentVariable)

    # print(DataGrid.columns)
    # print(len(DataGrid.columns))
    # print(DataGrid.iloc[0]['MELTS'].keys())
    # print(DataGrid.iloc[0]['MELTS']['Olivine'].keys())

    # TempAxis = ExtractIndependentAxis(DataGrid, 'MELTS/Olivine/Temperature')
    # fO2Axis = ExtractIndependentAxis(DataGrid, 'fO2')

    # fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Olivine/Temperature', 'MELTS/Olivine/FitIndex')
    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Orthopyroxene/Temperature', 'MELTS/Orthopyroxene/FitIndex')
    # fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Liquid/Temperature', 'MELTS/Liquid/FitIndex')
    # fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/CombinedFitIndex/Temperature', 'MELTS/CombinedFitIndex/FitIndex')
    CrossSec[CrossSec==0] = np.NaN

    plt.imshow(np.fliplr(CrossSec), origin='lower', extent=[TempAxis[0], TempAxis[-1], fO2Axis[0], fO2Axis[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel('Temperature $^{\circ}$C')
    plt.gca().invert_xaxis()
    plt.ylabel('f$_{O_2}$')
    plt.title('FitIndex Orthopyroxene')
    plt.show()

    print('------------------------------ DONE ------------------------------')

