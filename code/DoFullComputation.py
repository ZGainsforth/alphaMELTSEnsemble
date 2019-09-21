import numpy as np
import matplotlib.pyplot as plt
import sys, os
from collections import OrderedDict
import deepdish as dd
import pandas as pd
from GenMELTSEnsemble import GenerateMELTSEnsemble
from ProcessMELTS import ProcessAlphaMELTS
from CombineMELTSEnsemble import ReadInAllOutputs, Make2DCrossSection, Plot2DCrossSection

if __name__ == "__main__":
    print('------------------------------ START ------------------------------')
    # Only needs to be done if changing bulk composition or initial conditions/parameterization.
    DoMELTSSims = False
    # Only needs to be run if we changed the fitindex properties or something like that.
    DoAnalysis = True
    # We should replot.
    DoPlotting = True

    # Set up directory structure
    # Location to the alphamelts executable.
    alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alphamelts')
    # Location to where to put the compute files.
    ComputeScratchSpace = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ComputeScratchSpace2')

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
    # ParameterizedInputs['Na2O'] = np.arange(0.00, 5.00, 1)
    
    print('------------------------------ MELTS SIMULATIONS --------------------------')

    if DoMELTSSims:
        # Create MELTS simulations
        GenerateMELTSEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                                ConstantInputs, ParameterizedInputs)

        # And run them
        os.system('cd ' + ComputeScratchSpace + '; parallel < runall.sh; cd -')

    print('------------------------------ ANALYZE MELTS SIMULATIONS ---------------------------')

    if DoAnalysis:
        # Create a dictionary for each phase that we want to include in a fit index.
        # Each phase has a dictionary for all the oxides to include.
        TargetCompositions = dict()
        TargetCompositions['Olivine'] = {'SiO2':41.626, 'MgO':48.536, 'FeO':7.849}#, 'MnO':1.494, 'CaO':0.101, 'Cr2O3':0.394}
        TargetCompositions['Orthopyroxene'] = {'SiO2':54.437, 'MgO':31.335, 'FeO':4.724}
        # TargetCompositions['Alloy-Liquid'] = {'Fe':91.428, 'Ni':8.572}
        # TargetCompositions['Liquid'] = {'SiO2':48.736, 'MgO':25.867}
        
        ProcessOneDirectory = False
        if ProcessOneDirectory:
            # Do a computation on a single directory
            DirName = 'ComputeScratchSpace/VaryfO2_fO2=-3.5'
            ProcessAlphaMELTS(DirName=DirName, TargetCompositions=TargetCompositions)
        else:
            # Or in parallel on an entire ensemble.
            from glob2 import glob
            # from dask import delayed, compute
            import dask
            dask.config.set(scheduler='processes')
            # dask.config.set(scheduler='synchronous')

            ThisDir = os.path.dirname(os.path.abspath(__file__))
            # Dirs = glob(os.path.join(ThisDir, '../ComputeScratchSpace/*/'))
            Dirs = glob(ComputeScratchSpace + '/*/')

            @dask.delayed
            def DoOneDir(DirName):
                print('Dask for {}'.format(DirName))
                ProcessAlphaMELTS(DirName=DirName, TargetCompositions=TargetCompositions)
            Computes = [DoOneDir(Dir) for Dir in Dirs]
            dask.compute(Computes)

    print('------------------------------ PLOT MELTS SIMULATIONS ------------------------------')

    if DoPlotting:
        import matplotlib
        matplotlib.use('Qt5Agg',warn=False, force=True)
        from matplotlib import pyplot as plt
        print("Switched to:",matplotlib.get_backend())

        # Load the inputs, both constant and parameterized
        ConstantInputs = dd.io.load(os.path.join(ComputeScratchSpace, 'ConstantInputs.hdf5'))
        print(ConstantInputs)
        ParameterizedInputs = dd.io.load(os.path.join(ComputeScratchSpace, 'ParameterizedInputs.hdf5'))
        print(ParameterizedInputs)

        # Load the parameterized space to disk so we can reassemble the ensemble.
        DataGrid = pd.read_csv(os.path.join(ComputeScratchSpace, 'ParameterGrid.csv'), index_col=0)

        DataGrid = ReadInAllOutputs(ComputeScratchSpace, DataGrid)

        plt.figure()
        fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Olivine/Temperature', 'MELTS/Olivine/FitIndex')
        CrossSec[CrossSec==0] = np.NaN
        Plot2DCrossSection(CrossSec, TempAxis, fO2Axis, 'Temperature $^{\circ}$C', 'f$_{O_2}$', 'FitIndex Olivine')

        plt.figure()
        fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Orthopyroxene/Temperature', 'MELTS/Orthopyroxene/FitIndex')
        CrossSec[CrossSec==0] = np.NaN
        Plot2DCrossSection(CrossSec, TempAxis, fO2Axis, 'Temperature $^{\circ}$C', 'f$_{O_2}$', 'FitIndex Orthopyroxene')

        # plt.figure()
        # fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Liquid/Temperature', 'MELTS/Liquid/FitIndex')
        # CrossSec[CrossSec==0] = np.NaN
        # Plot2DCrossSection(CrossSec, TempAxis, fO2Axis, 'Temperature $^{\circ}$C', 'f$_{O_2}$', 'FitIndex Glass')

        plt.figure()
        fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/CombinedFitIndex/Temperature', 'MELTS/CombinedFitIndex/FitIndex')
        CrossSec[CrossSec==0] = np.NaN
        Plot2DCrossSection(CrossSec, TempAxis, fO2Axis, 'Temperature $^{\circ}$C', 'f$_{O_2}$', 'FitIndex Combined')

        plt.show()

    print('------------------------------ DONE ------------------------------')

