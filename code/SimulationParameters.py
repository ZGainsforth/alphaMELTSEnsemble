import os
from collections import OrderedDict
import numpy as np
from CombineMELTSEnsemble import Make2DCrossSection, Plot2DCrossSection

# Only needs to be done if changing bulk composition or initial conditions/parameterization.
DoMELTSSims = True
# Only needs to be run if we changed the fitindex properties or something like that.
DoAnalysis = True
# We should replot.
DoPlotting = True

# Set up directory structure
# Location to the alphamelts executable.
alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alphamelts')
# Location to where to put the compute files.
ComputeScratchSpace = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ComputeScratchSpace')

# Constant inputs
ConstantInputs = dict()
ConstantInputs['Title'] = 'VaryfO2'
ConstantInputs['Buffer'] = 'IW'
# Set the temperature.  Because this is a single value, there will only be an initial temperature entry made in the inputs melt file.
ConstantInputs['T'] = 2000 # Temperature
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

# Create a dictionary for each phase that we want to include in a fit index.
# Each phase has a dictionary for all the oxides to include.
TargetCompositions = dict()
TargetCompositions['Olivine'] = {'SiO2':41.626, 'MgO':48.536, 'FeO':7.849}#, 'MnO':1.494, 'CaO':0.101, 'Cr2O3':0.394}
TargetCompositions['Orthopyroxene'] = {'SiO2':54.437, 'MgO':31.335, 'FeO':4.724}
# TargetCompositions['Alloy-Liquid'] = {'Fe':91.428, 'Ni':8.572}
TargetCompositions['Liquid'] = {'SiO2':48.736, 'MgO':25.867}

# At the end of the plotting stage, after all data is gathered and processed, which final plots do you want to draw?
def DrawEnsemblePlots(ComputeScratchSpace, DataGrid):
    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Olivine/Temperature', 'MELTS/Olivine/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexOlivine.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Orthopyroxene/Temperature', 'MELTS/Orthopyroxene/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexOrthopyroxene.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Clinopyroxene/Temperature', 'MELTS/Clinopyroxene/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexClinopyroxene.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/AlloySolid/Temperature', 'MELTS/AlloySolid/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexAlloySolid.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/AlloyLiquid/Temperature', 'MELTS/AlloyLiquid/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexAlloyLiquid.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/Liquid/Temperature', 'MELTS/Liquid/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexLiquid.pdf'))

    fO2Axis, TempAxis, CrossSec = Make2DCrossSection(DataGrid, 'fO2', 'MELTS/CombinedFitIndex/Temperature', 'MELTS/CombinedFitIndex/FitIndex', SavePath=os.path.join(ComputeScratchSpace, 'FitIndexCombined.pdf'))

