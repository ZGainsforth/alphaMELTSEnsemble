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

def AddMELTSLine(MELTSStr, key, val):
    if key == 'fO2':
        MELTSStr += f'Log fO2 Delta: {val}\n'
    elif key == 'Buffer':
        MELTSStr += f'Log fO2 Path: {val}\n'
    elif key == 'P':
        MELTSStr += f'Initial Pressure: {val}\n'
    elif key == 'T':
        MELTSStr += f'Initial Temperature: {val}\n'
    elif key == 'Title':
        pass
    elif 'O' in key:
        MELTSStr += f'Initial Composition: {key} {val}\n'
    else:
        MELTSStr += f'Initial Trace: {key} {val}\n'
    return MELTSStr

def CreateNameOfCalculation(DataRow):#, Title):
    NameStr = ''
    for i in DataRow.index:
        if i != 'DirName':
            NameStr += f'_{i}={DataRow[i]}'
    return NameStr

def GenerateMeltsEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                            ConstantInputs, ParameterizedInputs):
    # Make a string which will be used to create a bash script that will process all the MELTS calculations.
    RunAllStr = ''

    # Make a product space of all the parameterized values.  We will have a pandas frame with
    # one row for each calculation.
    # Make a list of all the numpy arrays with just the numbers.  It's an ordered dict, so we can still
    # Match the keys up later.
    assert (type(ParameterizedInputs) is OrderedDict), 'ParamaterizedInput must be of type collections.OrderedDict.'
    ParameterValues = []
    for key, val in ParameterizedInputs.items():
        ParameterValues.append(val)
    ProductSpace = list(itertools.product(*ParameterValues))
    # Make a dataframe with the values..
    ParameterGrid = pd.DataFrame(np.array(ProductSpace), columns=ParameterizedInputs.keys())

    # Now Compute a directory name for each calculation and add that to the dataframe.
    ParameterGrid['DirName'] = ConstantInputs['Title']
    for i in range(ParameterGrid.shape[0]):
        ParameterGrid.at[i,'DirName'] = ParameterGrid.iloc[i]['DirName'] + CreateNameOfCalculation(ParameterGrid.iloc[i])

    # Report to the user how beefy this calculation will be.
    print(f'Total number of alphaMELTS calculations will be: {ParameterGrid.shape[0]}')

    # This will be a shell script to run all the simulations.
    RunAll = ''

    # Now we build a file for each parameterized combination.
    for index, row in ParameterGrid.iterrows():

        # Make a name for this MELTS computation.
        NameStr = ConstantInputs['Title']

        # Make a string for the contents of this MELTS file.
        MELTSStr = ''
        # A title always goes first.
        MELTSStr += f'Title: {ConstantInputs["Title"]}\n'
    
        # For the parameterized inputs, we go through the list and pull out the current index that we're working on.
        for param in row.index:
            if param != 'DirName':
                MELTSStr = AddMELTSLine(MELTSStr, param, row[param])

        # For the constant inputs, we just dump them into the file.
        for key,val in ConstantInputs.items():
            MELTSStr = AddMELTSLine(MELTSStr, key, val)

        # Now create a directory for this computation.
        ComputeDir = os.path.join(ComputeScratchSpace, row['DirName'])
        if not os.path.exists(ComputeDir):
            os.makedirs(ComputeDir)

        # And write the melts file into this directory.
        with open(os.path.join(ComputeDir, 'input.melts'), 'w') as f:
            f.write(MELTSStr)

        # Copy the alphamelts settings file, and the batch command file.
        for FileName in ['alphamelts_settings.txt', 'mybatch']:
            shutil.copy(os.path.join(Path(__file__).parent.absolute(), FileName), os.path.join(ComputeDir, FileName))

        # Add this computation to the RunAll file.
        RunAll += 'cd "' + ComputeDir + '" && '
        RunAll += os.path.join(alphaMELTSLocation, 'run_alphamelts.command') + ' -f alphamelts_settings.txt -b mybatch\n'

    # Store the parameterized space to disk so we can reassemble the ensemble later.
    dd.io.save(os.path.join(ComputeScratchSpace, 'ConstantInputs.hdf5'), ConstantInputs)

    # Store the parameterized space to disk so we can reassemble the ensemble later.
    dd.io.save(os.path.join(ComputeScratchSpace, 'ParameterizedInputs.hdf5'), ParameterizedInputs)

    # Store the parameterized space to disk so we can reassemble the ensemble later.
    ParameterGrid.to_csv(os.path.join(ComputeScratchSpace, 'ParameterGrid.csv'))

    # Write out the Runall File.
    with open(os.path.join(ComputeScratchSpace, 'runall.sh'), 'w') as f:
        f.write(RunAll)
    
if __name__ == "__main__":
    print('------------------------------ START ------------------------------')

    # Set up directory structure
    # Location to the alphamelts executable.
    alphaMELTSLocation = os.path.join(os.getcwd(), 'alphamelts')
    # Location to where to put the compute files.
    ComputeScratchSpace = os.path.join(os.getcwd(), 'ComputeScratchSpace')

    # # Set up constant inputs.
    # T1 = 1500 # Start at 1500C.
    # Buffer = 'IW' # We will deal with fO2 relative to the IW buffer.
    # Pressure = 1 # bar.

    # Constant inputs
    ConstantInputs = dict()
    ConstantInputs['Title'] = 'VaryfO2'
    ConstantInputs['Buffer'] = 'IW'
    # Set the temperature.  Because this is a single value, there will only be an initial temperature entry made in the inputs melt file.
    ConstantInputs['T'] = 1500 # Temperature
    ConstantInputs['P'] = 1 # Bar
    # And so for all the elements.
    ConstantInputs['SiO2'] = 44.10
    ConstantInputs['TiO2'] = 0.07
    ConstantInputs['Al2O3'] = 0.66
    ConstantInputs['Cr2O3'] = 0.52
    ConstantInputs['FeO'] = 11.53
    ConstantInputs['MnO'] = 1.34
    ConstantInputs['MgO'] = 38.95
    ConstantInputs['CaO'] = 0.62
    ConstantInputs['NiO'] = 0.31
    
    # Set up the inputs to the simulation that vary.
    ParameterizedInputs = OrderedDict()
    # Set the fugacity.  Because this is a set of values, a new MELTS simulation will be created for each value.
    ParameterizedInputs['fO2'] = np.arange(-6, 0, 0.25)
    ParameterizedInputs['Na20'] = np.arange(0.00, 5.00, 1)
    
    GenerateMeltsEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                            ConstantInputs, ParameterizedInputs)

    print('------------------------------ DONE ------------------------------')

