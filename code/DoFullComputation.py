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

    # NOTA BENE!  There is no longer a need to edit the file.  All the input parameters and editable portions have been moved to SimulationParameters.py.  The user should edit the file SimulationParameters.py.  It has the input composition as well as desired output plots, etc.
    from SimulationParameters import *

    print('------------------------------ MELTS SIMULATIONS --------------------------')

    if DoMELTSSims:
        # Create MELTS simulations
        GenerateMELTSEnsemble(  alphaMELTSLocation, ComputeScratchSpace,
                                ConstantInputs, ParameterizedInputs)

        # And run them
        os.system('cd "' + ComputeScratchSpace + '"; parallel < runall.sh; cd -')

    print('------------------------------ ANALYZE MELTS SIMULATIONS ---------------------------')

    if DoAnalysis:
        ProcessOneDirectory = False
        if ProcessOneDirectory:
            # Do a computation on a single directory
            DirName = '../ComputeScratchSpace/VaryfO2_fO2=-4.5'
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

        # This funtion is in SimulationParameters.py -- configured by the user.
        DrawEnsemblePlots(ComputeScratchSpace, DataGrid)

        plt.show()

    print('------------------------------ DONE ------------------------------')
