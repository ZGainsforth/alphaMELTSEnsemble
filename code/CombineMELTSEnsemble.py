# GenMELTSEnsemble
# Written 2019 by Zack Gainsforth 
# The purpose of this file is to take inputs and generate a bunch of MELTS calculations
# to explore some parameter space (we'll call it an ensemble here).
import numpy as np
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

def ReadInAllOutputs(ComputeScratchSpace, DataGrid):
    """ Goes through all the completed MELTS calculations and loads in the results to make a
    monster DataGrid.
        Inputs: 
            ComputeScratchSpace (str): Path root for where all the MELTS calculations reside.
            DataGrid (pd.dataframe): Holds the parameterized space on input.
        Outputs:
            DataGrid (pd.dataframe): Now populated with all the data.
    """

    # Add a column to contain the MELTS calculations.
    DataGrid['MELTS'] = None

    for i in range(DataGrid.shape[0]):
        print(f'Loading MELTS data from {DataGrid.iloc[i]["DirName"]}.')
        # MELTSData will be a list of dataframes -- one for each phase that exists in that MELTS computation.
        MELTSData = dict()
        # Read in the phases that exist.
        for PhaseName  in ['PhaseMassPercents', 'CombinedFitIndex', 'Olivine', 'Clinopyroxene', 'Orthopyroxene', 'Liquid', 'Spinel', 'AlloyLiquid', 'AlloySolid']:
            try:
                DataItem = pd.read_csv(os.path.join(ComputeScratchSpace, DataGrid.iloc[i]['DirName'], 'Output_' + PhaseName + '.csv'))
            except FileNotFoundError as e:
                # File not found just means that phase doesn't exist under these conditions of simulation.
                continue
            MELTSData[PhaseName] = DataItem
        DataGrid.at[i, 'MELTS'] = MELTSData
        print(MELTSData.keys())
        # if i > 5:
        #     break

    return DataGrid


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
    DataGrid = pd.read_csv(os.path.join(ComputeScratchSpace, 'ParameterGrid.csv'))

    DataGrid = ReadInAllOutputs(ComputeScratchSpace, DataGrid)
    # print(DataGrid.iloc[0]['MELTS'])

    print(deep_getsizeof(DataGrid, set()))

    print('------------------------------ DONE ------------------------------')

