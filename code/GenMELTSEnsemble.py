# GenMELTSEnsemble
# Written 2019 by Zack Gainsforth 
# The purpose of this file is to take inputs and generate a bunch of MELTS calculations
# to explore some parameter space (we'll call it an ensemble here).
import numpy as np
import sys, os
import shutil
import tempfile
from collections import OrderedDict
import json
from pathlib import Path
import yaml

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

def GenerateMeltsEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                            ConstantInputs, ParameterizedInputs):
    # Make a string which will be used to create a bash script that will process all the MELTS calculations.
    RunAllStr = ''

    # We make a mesh of the parameterized values -- this is an n-dimensional grid of all the variable calculation values.
    X = np.meshgrid([x for x in ParameterizedInputs.values()])

    # Report to the user how beefy this calculation will be.
    TotalCalculations = 0
    for x in X:
        TotalCalculations += np.prod(x.shape)
    print(f'Total number of alphaMELTS calculations will be: {TotalCalculations}')

    # Flatten out the fancy n-dimensional space so we can do this in a single loop.
    Y = [x.ravel() for x in X]

    # Then we make a dictionary like ParameterizedInputs so we can look up values based on elements.
    ParameterizedInputsMeshed = OrderedDict()
    for i, (key, val) in enumerate(ParameterizedInputs.items()):
        ParameterizedInputsMeshed[key] = Y[i]

    # This will be a shell script to run all the simulations.
    RunAll = ''

    # Now we build a file for each parameterized combination.
    for i in range(TotalCalculations):
        # Make a name for this MELTS computation.
        NameStr = ConstantInputs['Title']

        # Make a string for the contents of this MELTS file.
        MELTSStr = ''
        # A title always goes first.
        MELTSStr += f'Title: {ConstantInputs["Title"]}\n'
    
        # For the parameterized inputs, we go through the list and pull out the current index that we're working on.
        for key,arr in ParameterizedInputsMeshed.items():
            val = arr[i]
            MELTSStr = AddMELTSLine(MELTSStr, key, val)
            NameStr += f'_{key}={val}'

        # For the constant inputs, we just dump them into the file.
        for key,val in ConstantInputs.items():
            MELTSStr = AddMELTSLine(MELTSStr, key, val)

        # Now create a directory for this computation.
        ComputeDir = os.path.join(ComputeScratchSpace, NameStr)
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

    # Put a json file down which records our parameter space.
    with open(os.path.join(ComputeScratchSpace, 'ParameterizedInputs.json'), 'w') as f:
        # json.dumps(ParameterizedInputs, f)
        yaml.dump(ParameterizedInputs, f)

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
    ParameterizedInputs['fO2'] = np.arange(-7, 0, 0.25)
    
    GenerateMeltsEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                            ConstantInputs, ParameterizedInputs)

    print('------------------------------ DONE ------------------------------')

