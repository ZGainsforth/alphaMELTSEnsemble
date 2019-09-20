import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use('AGG')
import pandas as pd
import re
from pprint import pprint
import platform
import os, sys, shutil
from io import StringIO
import pickle

# Plotting is great, but slows things down by about 3x.  So if you just want the final plots from PlotParameterSpace.py you can disable the plotting of each phase composition and fit index here.
enable_plotting = True

def ProcessAlphaMELTS(DirName=os.getcwd(), TargetCompositions=dict()):
    """ ProcessAlphaMELTS
        Input: 
            DirName (str): Name of directory containing a completed alphaMELTS calculation.
            TargetCompositions (dict of dict): Dictionary structure containing target compositions for phases when computing
                Fit index.  Like so: TargetCompositions['Olivine'] = {'SiO2':41.626, 'MgO':48.536, ...}
        Output:
            The calculation in the directory is summarized into CSV's and plots.
    """

    print('Processing ' + DirName)    

    # Read the alphaMELTS file that has all the output.
    try:
        with open(os.path.join(DirName, 'alphaMELTS_tbl.txt'), 'r') as myfile:
            data=myfile.read()
    except:
        print('Failed to open ' + FileName + '.  Skipping.')
        return

    ### Phase masses
    # Get the phase masses chunk of text.
    header = 'Phase Masses:'
    PhaseData = GetalphaMELTSSection(data, header)
    # Check if this MELTS computation includes the phase table.
    if PhaseData is not None:
         PlotPhaseMasses(PhaseData, DirName)
    else:
        print ('Error!  No phase masses section in %s. Skipping.' % FileName)

    # Put all the phases into a dictionary for easy access
    PhasesData = dict()

    ### Olivine
    PhasesData['Olivine'] = ExtractAndPlotOnePhase(data, 'Olivine', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Spinel
    PhasesData['Spinel'] = ExtractAndPlotOnePhase(data, 'Spinel', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Liquid (Glass)
    PhasesData['Liquid'] = ExtractAndPlotOnePhase(data, 'Liquid', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Orthopyroxene
    PhasesData['Orthopyroxene'] = ExtractAndPlotOnePhase(data, 'Orthopyroxene', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Clinopyroxene
    PhasesData['Clinopyroxene'] = ExtractAndPlotOnePhase(data, 'Clinopyroxene', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Alloy-Solid (metal)
    PhasesData['Alloy-Solid'] = ExtractAndPlotOnePhase(data, 'Alloy-Solid', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    ### Alloy-Liquid (metal)
    PhasesData['Alloy-Liquid'] = ExtractAndPlotOnePhase(data, 'Alloy-Liquid', DirName, PlotAxis='Temperature', TargetCompositions=TargetCompositions)

    # Make a combined Fit index.
    # CombinedFitAxis and index will be the interesection of all phases.
    CombinedFitAxis = None # The x-axis for plotting the fit index.  Usually temperature.
    CombinedFitIndex = None # The fit index for all phases.
    TotalDOF = 0
    for key, val in TargetCompositions.items():
        print(f'Adding {len(val)} DOFs for {key} to aggregate fit index.')
        TempFitAxis = PhasesData[key]['Temperature']
        TempFitIndex = PhasesData[key]['FitIndex']
        if CombinedFitIndex is None:
            # If this is the first phase, then there is nothing to combine yet.
            CombinedFitAxis = TempFitAxis 
            CombinedFitIndex = TempFitIndex * len(val)
            TotalDOF += len(val)
        else:
            # Be sure to re-multiply by the number of DOF.
            TempFitIndex *= len(val)
            TotalDOF += len(val)
            # Check if any temps in our combined axis are not included in this Temp axs.
            Mask = np.isin(CombinedFitAxis, TempFitAxis)
            # If so, trim those values.
            CombinedFitAxis = CombinedFitAxis[Mask]
            CombinedFitIndex = CombinedFitIndex[Mask]
            # And do it the other way around
            Mask = np.isin(TempFitAxis, CombinedFitAxis)
            TempFitAxis = TempFitAxis[Mask]
            TempFitIndex = TempFitIndex[Mask]
            # Finally, we get to add the FitIndexes together.
            CombinedFitIndex += TempFitIndex

    # After combining the fit indices, we need to divide by the total DOF.
    CombinedFitIndex /= TotalDOF

    # print(CombinedFitAxis[:,np.newaxis].T)
    # print(CombinedFitIndex[:,np.newaxis].T)
    # print(TotalDOF)

    # Save the ensemble fit index as a csv.
    FitIndex = pd.DataFrame({'Temperature':CombinedFitAxis, 'FitIndex':CombinedFitIndex})
    FitIndex.to_csv(os.path.join(DirName, 'Output_CombinedFitIndex.csv'))

    # Save the Combined Fit index to a plot.
    plt.figure()
    plt.plot(FitIndex['Temperature'], FitIndex['FitIndex'])
    plt.xlabel('Temperature')
    plt.ylabel('Fit index')
    plt.title('Combined Fit index with {} DOFs'.format(TotalDOF))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_CombinedFitIndex.png'))
    
def mysavefig(filename):
    if enable_plotting:
        filename = filename.replace('png', 'pdf')
        plt.savefig(filename)

def GetalphaMELTSSectionAsTxt(data, start):
    """ GetAlphaMELTSSection(): Given the output file from a alphaMELTS calculation, extract just one section.
        Input:
            data (str): The output of the melts calculation (from file alphaMELTS_tbl.txt).
            start (str): The name of the section.
        Output:
            Returns the string containing the entire section except for the line containing the start string.
    """

    # The end of the section is a double <CR>
    stop = '\n\n'
    # We are looking for all the text (any characters) between the start string and \n\n
    reout = re.compile(r'%s.*?%s' % (start, stop), re.S)
    try:
        SectionStr = reout.search(data).group(0)
    except:
        # It is possible that this MELTS computation didn't produce this mineral.  If so, just bail.
        return None

    # This is handling a bug in alphaMELTS where alloy-solid doesn't include a label for the structure column.
    if ('alloy-solid' in start) or ('alloy-liquid' in start):
        SectionStr = SectionStr.replace('formula', 'structure formula')

    return StringIO(SectionStr)

def GetalphaMELTSSection(data, start):
    """ GetAlphaMELTSSection(): Given the output file from a alphaMELTS calculation, extract just one section.
        Input:
            data (str): The output of the melts calculation (from file alphaMELTS_tbl.txt).
            start (str): The name of the section.
        Output:
            Returns the string containing the entire section except for the line containing the start string.
    """
    # The end of the section is a double <CR>
    stop = '\n\n'
    # We are looking for all the text (any characters) between the start string and \n\n
    reout = re.compile(r'%s.*?%s' % (start, stop), re.S)
    try:
        SectionStr = reout.search(data).group(0)
    except:
        # It is possible that this MELTS computation didn't produce this mineral.  If so, just bail.
        return None

    # Convert it into a numpy array.
    SectionData = np.genfromtxt(StringIO(SectionStr), skip_header=1, skip_footer=0, dtype=None)
    return SectionData

def PlotMultiWtPct(T, ColNames, indices, MineralData, DirName, PlotTitle):
    """ PlotMultiWtPct
        Input:
            T (np.array): Temperature axis.
            ColNames (list): The names of all the elements (or phases) to plot.
            indices (np.array): Which columns in MineralData to plot (not always every column should be plotted!)
            MineralData (np.array): The values to plot with dimensions T x len(ColNames) 
            DirName (str): Location to put the output files.
            PlotTitle (str): A nice string for the header of the plot.
        Output:
            A plot with a name derived from PlotTitle is written to DirName.
    """

    # For output files, we can't use all the latex characters...
    valid_chars = "+-_.()/= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Make a figure.
    plt.figure()
    plt.clf()
    ax = plt.subplot(111)
    
    # Figure out which elements to plot and plot them.
    PlotStr = ', '.join(['T, MineralData[:, ' + str(i) + ']' for i in indices])
    LegendList = ['Wt % ' + ColNames[i] for i in indices]
    eval('plt.plot(' + PlotStr + ')')

    # Pretty up the labels and axes.
    plt.xlabel('Temperature ($^{\circ}$C)')
    plt.ylabel('Wt %')
    plt.gca().invert_xaxis()
    plt.title(PlotTitle)

    # Make a fancy looking legend.
    box = ax.get_position()
    LegendHeightFraction = 0.15
    plt.legend(LegendList, loc='lower center', ncol=5, bbox_to_anchor=(0.5,-LegendHeightFraction-0.1), prop={'size':8})
    ax.set_position([box.x0, box.y0 + box.height * LegendHeightFraction, box.height, box.height*(1-LegendHeightFraction)])
    OutFileName = ''.join(x for x in os.path.join(DirName, 'Output_' + PlotTitle + '.png') if x in valid_chars)

    # And output it to the disk
    print('Saving', OutFileName)
    mysavefig(OutFileName)

def PlotPhaseMasses(PhaseData, DirName):
    """ PlotPhases() takes the phase data output by MELTS and produces a csv file and plots from it.
        Input:
            PhaseData (str): string with the section of text from the MELTS output file.
            DirName (str): Location to put output files.
        Output:
            Phase_mass_percents.csv: CSV file containing all the information.
            Phase_mass_percentspng: Plot of the phases as a function of temperature.
    """

    # Verify that MELTS hasn't changed the output since we were programmed.
    # This section has variable headers, but the first 3 should be the same.
    if (PhaseData[0,:3] != np.array([b'Pressure', b'Temperature', b'mass'])).any():
        print ("alphaMELTS output format for phase masses has changed.")
        return

    # The first line is the name of the columns.
    ColNames = np.copy(PhaseData[0,:].astype(str))
    PhaseData = PhaseData[1:,:]

    # Convert PhaseData to floats so we can do math.
    PhaseData = PhaseData.astype(float)

    # Convert Kelvin to celcius.
    T = PhaseData[:,1].astype(float) - 273.15

    # Capitalize the phase names so they look nice.  Get rid of the _0 from the names.
    Columns = [str(i).capitalize().split('_')[0] for i in ColNames]

    # Get all the numbers and call them indices.  We don't need the numbers for the first three columns: Press, temp, mass.
    Indices = range(len(ColNames))
    Indices = Indices[3:]

    # Sometimes the total weight doesn't add up to 100 so let's normalize.
    for i in Indices:
        PhaseData[:,i] /= PhaseData[:,2]
        PhaseData[:,i] *= 100
    PhaseData[:,2] = 100

    # Plot all phases
    PlotMultiWtPct(T, Columns, Indices, PhaseData, DirName, 'Phase mass percents')
    
    # Write the phases to a csv file so we can manipulate it later.
    np.savetxt(os.path.join(DirName, 'Output_PhaseMassPercents.csv'), PhaseData, delimiter=',', header=','.join(ColNames))

    plt.close('all')

    return

def ExtractAndPlotOnePhase(MELTSData, PhaseName, DirName, PlotAxis='Temperature', TargetCompositions=None):
    # Get the chunk of text for this phase.
    header = PhaseName.lower() + '_0 thermodynamic data and composition:'
    DataRaw = GetalphaMELTSSectionAsTxt(MELTSData, header)

    print(PhaseName)

    # Check if this MELTS computation includes this phase.
    if DataRaw is not None:
        # Read text as a CSV and default everything to floats, except for a couple fields that are strings.
        Data = pd.read_csv(DataRaw, header=1, delimiter=' ')
        for c in Data.columns:
            if c not in ['formula', 'structure']:
                Data[c] = Data[c].astype(float)

        # Convert Kelvin to Celcius.
        Data['Temperature'] -= 273.15

        # Determine if this phase participates in the FitIndex compute.
        if PhaseName in TargetCompositions.keys():
            PhaseCompo = TargetCompositions[PhaseName]
        else:
            PhaseCompo = None

        # Plot the phase.
        if PhaseName == 'Olivine':
            PhaseData = PlotOlivine(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Spinel':
            PhaseData = PlotSpinel(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Liquid':
            PhaseData = PlotLiquid(Data, DirName, FitCompo=PhaseCompo)
        # elif PhaseName == 'Feldspar':
        #     PhaseData = PlotFeldspar(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Clinopyroxene':
            PhaseData = PlotClinopyroxene(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Orthopyroxene':
            PhaseData = PlotOrthopyroxene(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Alloy-Solid':
            PhaseData = PlotAlloySolid(Data, DirName, FitCompo=PhaseCompo)
        elif PhaseName == 'Alloy-Liquid':
            PhaseData = PlotAlloyLiquid(Data, DirName, FitCompo=PhaseCompo)
        # elif PhaseName == 'Nepheline':
        #     PhaseData = PlotNepheline(Data, DirName, FitCompo=PhaseCompo)
        else:
            print('Brwooop!  Brwooop!  Phase {} not recognized!  Please program it in.'.format(PhaseName))
    else:
        # If not, note that this mineral is not here.
        PhaseData = None
    return PhaseData

def ExtractFormulaComponent(Formula, FormulaStr):
    """ ExtractFormulaComponent gets a substring of the formula to give the formula unit for a given element.
        Input:
            Formula (pd.Series): Strings containing a whole formula (such as 'Fe''0.32Mg0.68Fe'''0.01Al0.19Cr1.79Ti0.01O4' for spinel.
            FormulaStr (str): A regex for the substring we want (such as 'Mg([0-9.]*)Fe').
        Output:
            pd.Series: Just like formula, but it is a float datatype with the number that was extracted.
    """
    # Extract the substring which corresponds to this element in the formula string.
    ElNum = Formula.str.findall(FormulaStr)
    # Findall returns a list of all matches -- there will only be one so return it.
    for i,o in enumerate(ElNum):
        # If there is no match, then the string for this value would be blank. 
        if len(o) == 0:
            ElNum[i] = ''
        else:
            ElNum[i] = o[0]
    # Now we can convert these to numbers
    ElNum = pd.to_numeric(ElNum, errors='coerce')
    return ElNum

def GetPlotAxis(Data, PlotAxis):
    """ GetPlotAxis figures out what the text of the plot axis should be and gets the appropriate series out of the data frame.
        Input:
            Data (pd.DataFrame): The data frame for the MELTS simulation including the x-axis.
            PlotAxis (str): The name of the series with the plot axis.
        Output:
            x (pd.Series): The series with the name PlotAxis.
            xtext (str): A nice string to populate the plot x-label with.
    """

    # First configure the x axis values and label
    x = Data[PlotAxis]
    if PlotAxis == 'Temperature':
        xtext = 'Temperature ($^{\circ}$C)'
    else:
        xtext = PlotAxis
    return x, xtext
 
def ComputeFitIndex(MELTSCompo, FitCompo):
    """ ComputeFitIndex computes a fit index for one phase given a target composition.
        Input:
            MELTSCompo (pd.DataFrame): A df with the compositions that were computed by MELTS.  We will compare each for a fit.
            FitCompo (dict): A dict of the format {'SiO2': 45.5, 'MgO':33.0 ...} which gives the target composition for comparison.
        Output:
            FitIndex (np.array[n]): A fit index for each row in MELTSCompo
    """

    # The length of the fit index array should be the number of rows in MELTSCompo.  One fit index for each MELTS compute.
    FitIndex = np.zeros(MELTSCompo.shape[0])

    # Loop through all the oxides in FitCompo, and compute the fit index for that.
    for El, Val in FitCompo.items():
        FitIndex += np.abs((MELTSCompo[El] - Val) / Val)
    
    # Now we have to normalize against the number of DOFs, which is the number of oxides we used for the fit.
    FitIndex /= len(FitCompo)

    return FitIndex

def PlotOlivine(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """ PlotOlivine() takes the olivine data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            Olivine.csv: CSV file containing all the information.
            Olivine.png: Plot of the phases as a function of temperature.
    """

    # Compute the Fo number and add it to the data frame.
    MgNum = ExtractFormulaComponent(Data['formula'], 'Mg([0-9.]*)Fe')
    FeNum = ExtractFormulaComponent(Data['formula'], "Fe''([0-9.]*)Mn")
    Fo = MgNum/(MgNum+FeNum)*100
    Data['Fo'] = Fo

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_Olivine.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(3,1, figsize=(9,9))
    else:
        fig, ax = plt.subplots(2,1, figsize=(9,6))
    ax[0].plot(x, Data['Fo'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Fo')
    ax[0].set_title('Fo')
    ax[1].plot(x, Data['Cr2O3'], x, Data['MnO'], x, Data['CaO'], x, Data['NiO'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('Wt %')
    ax[1].set_title('Minors')
    ax[1].legend(['Cr2O3', 'MnO', 'CaO', 'NiO'])
    if FitCompo is not None:
        ax[2].plot(x, FitIndex)
        ax[2].set_xlabel(xtext)
        ax[2].set_ylabel('Fit index')
        ax[2].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_OlivineComposition.png'))

    return Data

def PlotSpinel(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """ PlotSpinel() takes the spinel data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            Spinel.csv: CSV file containing all the information.
            Spinel.png: Plot of the phases as a function of temperature.
    """

    # Compute formula numbers which are relevant.
    MgNum = ExtractFormulaComponent(Data['formula'], 'Mg([0-9.]*)Fe')
    Fe2Num = ExtractFormulaComponent(Data['formula'], "Fe''([0-9.]*)Mg")
    Fe3Num = ExtractFormulaComponent(Data['formula'], "Fe'''([0-9.]*)Al")
    AlNum = ExtractFormulaComponent(Data['formula'], "Al([0-9.]*)Cr")
    TiNum = ExtractFormulaComponent(Data['formula'], "Ti([0-9.]*)O")
    CrNum = ExtractFormulaComponent(Data['formula'], "Cr([0-9.]*)Ti")
    # Mg/Tet
    MgOverTet = MgNum / (MgNum+Fe2Num)
    Data['MgOverTet'] = MgOverTet
    # Al/Oct
    AlOverOct = AlNum / (AlNum+CrNum+Fe3Num)
    Data['AlOverOct'] = AlOverOct
    # Fe3+ / sum(Fe)
    Fe3OverSumFe = Fe3Num / (Fe3Num+Fe2Num)
    Data['Fe3OverSumFe'] = Fe3OverSumFe

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_Spinel.csv'))

    # Plot the olivine stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(3,1, figsize=(9,9))
    else:
        fig, ax = plt.subplots(2,1, figsize=(9,6))
    ax[0].plot(x, Data['MgOverTet'], x, Data['AlOverOct'], x, Data['Fe3OverSumFe'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Site Occupancies')
    ax[0].set_title('Site Occupancies')
    ax[0].legend(['Mg/$\Sigma$Tet', 'Al/$\Sigma$Oct', 'Fe$^{3+}$/$\Sigma$Fe'])
    ax[1].plot(x, Data['MgO'], x, Data['Al2O3'], x, Data['FeO'], x, Data['Fe2O3'], x, Data['Cr2O3'], x, Data['TiO2'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('Wt %')
    ax[1].set_title('Minors')
    ax[1].legend(['MgO', 'Al2O3', 'FeO', 'Fe2O3', 'Cr2O3', 'TiO2'])
    if FitCompo is not None:
        ax[2].plot(x, FitIndex)
        ax[2].set_xlabel(xtext)
        ax[2].set_ylabel('Fit index')
        ax[2].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_SpinelComposition.png'))

    return Data

def PlotLiquid(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """ PlotLiquid() takes the liquid data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            Liquid.csv: CSV file containing all the information.
            Liquid.png: Plot of the phases as a function of temperature.
    """

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_Liquid.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(4,1, figsize=(9,12))
    else:
        fig, ax = plt.subplots(3,1, figsize=(9,9))
    # Basic properties subplot
    ax[0].plot(x, Data['Cp'], x, Data['viscosity'], x, Data['Mg#'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Various units')
    ax[0].set_title('Liquid properties')
    ax[0].legend(['Cp', 'Viscosity', 'Mg Number'])
    # Majors
    ax[1].plot(x, Data['SiO2'], x, Data['Al2O3'], x, Data['Fe2O3'], x, Data['FeO'], x, Data['MgO'], x, Data['CaO'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('Wt %')
    ax[1].set_title('Majors')
    ax[1].legend(['SiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO'])
    # Minors
    ax[2].plot(x, Data['TiO2'], x, Data['Cr2O3'], x, Data['MnO'], x, Data['NiO'])
    ax[2].set_xlabel(xtext)
    ax[2].set_ylabel('Wt %')
    ax[2].set_title('Minors')
    ax[2].legend(['TiO2', 'Cr2O3', 'MnO', 'NiO'])
    if FitCompo is not None:
        ax[3].plot(x, FitIndex)
        ax[3].set_xlabel(xtext)
        ax[3].set_ylabel('Fit index')
        ax[3].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_LiquidComposition.png'))

    return Data

def PlotClinopyroxene(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """  PlotClinopyroxene() takes the clinopyroxene data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            Clinopyroxene.csv: CSV file containing all the information.
            Clinopyroxene.png: Plot of the phases as a function of temperature.
    """

    # Compute formula numbers which are relevant.
    MgNum = ExtractFormulaComponent(Data['formula'], 'Mg([0-9.]*)Fe')
    Fe2Num = ExtractFormulaComponent(Data['formula'], "Fe''([0-9.]*)Mg")
    Fe3Num = ExtractFormulaComponent(Data['formula'], "Fe'''([0-9.]*)Ti")
    FeNum = Fe2Num+Fe3Num
    AlNum = ExtractFormulaComponent(Data['formula'], "Al([0-9.]*)Si")
    SiNum = ExtractFormulaComponent(Data['formula'], "Si([0-9.]*)O")
    # En Number
    En = MgNum/(MgNum+FeNum)*100
    Data['Mg#'] = En
    # Al/sum(T)
    AlOverTSite = AlNum / (AlNum+SiNum)
    Data['AlOverTSite'] = AlOverTSite
    # Fe3+ / sum(Fe)
    Fe3OverSumFe = Fe3Num / (Fe3Num+Fe2Num)
    Data['Fe3OverSumFe'] = Fe3OverSumFe

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_Clinopyroxene.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(4,1, figsize=(9,12))
    else:
        fig, ax = plt.subplots(3,1, figsize=(9,9))
    # Basic properties subplot
    ax[0].plot(x, Data['Mg#']/100, x, Data['AlOverTSite'], x, Data['Fe3OverSumFe'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Various units')
    ax[0].set_title('Clinopyroxene properties')
    ax[0].legend(['En Num/100', 'Al/(Al+Si)', 'Fe/$\Sigma$Fe'])
    # Majors
    ax[1].plot(x, Data['SiO2'], x, Data['Al2O3'], x, Data['Fe2O3'], x, Data['FeO'], x, Data['MgO'], x, Data['CaO'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('Wt %')
    ax[1].set_title('Majors')
    ax[1].legend(['SiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO'])
    # Minors
    ax[2].plot(x, Data['TiO2'], x, Data['Cr2O3'], x, Data['MnO'], x, Data['NiO'])
    ax[2].set_xlabel(xtext)
    ax[2].set_ylabel('Wt %')
    ax[2].set_title('Minors')
    ax[2].legend(['TiO2', 'Cr2O3', 'MnO', 'NiO'])
    if FitCompo is not None:
        ax[3].plot(x, FitIndex)
        ax[3].set_xlabel(xtext)
        ax[3].set_ylabel('Fit index')
        ax[3].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_ClinopyroxeneComposition.png'))

    return Data

def PlotOrthopyroxene(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """  PlotOrthopyroxene() takes the orthopyroxene data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            Orthopyroxene.csv: CSV file containing all the information.
            Orthopyroxene.png: Plot of the phases as a function of temperature.
    """

    # Compute formula numbers which are relevant.
    MgNum = ExtractFormulaComponent(Data['formula'], 'Mg([0-9.]*)Fe')
    Fe2Num = ExtractFormulaComponent(Data['formula'], "Fe''([0-9.]*)Mg")
    Fe3Num = ExtractFormulaComponent(Data['formula'], "Fe'''([0-9.]*)Ti")
    FeNum = Fe2Num+Fe3Num
    AlNum = ExtractFormulaComponent(Data['formula'], "Al([0-9.]*)Si")
    SiNum = ExtractFormulaComponent(Data['formula'], "Si([0-9.]*)O")
    # En Number
    En = MgNum/(MgNum+FeNum)*100
    Data['Mg#'] = En
    # Al/sum(T)
    AlOverTSite = AlNum / (AlNum+SiNum)
    Data['AlOverTSite'] = AlOverTSite
    # Fe3+ / sum(Fe)
    Fe3OverSumFe = Fe3Num / (Fe3Num+Fe2Num)
    Data['Fe3OverSumFe'] = Fe3OverSumFe

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_Orthopyroxene.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(4,1, figsize=(9,12))
    else:
        fig, ax = plt.subplots(3,1, figsize=(9,9))
    # Basic properties subplot
    ax[0].plot(x, Data['Mg#']/100, x, Data['AlOverTSite'], x, Data['Fe3OverSumFe'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Various units')
    ax[0].set_title('Orthopyroxene properties')
    ax[0].legend(['En Num/100', 'Al/(Al+Si)', 'Fe/$\Sigma$Fe'])
    # Majors
    ax[1].plot(x, Data['SiO2'], x, Data['Al2O3'], x, Data['Fe2O3'], x, Data['FeO'], x, Data['MgO'], x, Data['CaO'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('Wt %')
    ax[1].set_title('Majors')
    ax[1].legend(['SiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO'])
    # Minors
    ax[2].plot(x, Data['TiO2'], x, Data['Cr2O3'], x, Data['MnO'], x, Data['NiO'])
    ax[2].set_xlabel(xtext)
    ax[2].set_ylabel('Wt %')
    ax[2].set_title('Minors')
    ax[2].legend(['TiO2', 'Cr2O3', 'MnO', 'NiO'])
    if FitCompo is not None:
        ax[3].plot(x, FitIndex)
        ax[3].set_xlabel(xtext)
        ax[3].set_ylabel('Fit index')
        ax[3].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_OrthopyroxeneComposition.png'))

    return Data

def PlotAlloySolid(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """ PlotAlloySolid() takes the solid metal data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            AlloySolid.csv: CSV file containing all the information.
            AlloySolid.png: Plot of the phases as a function of temperature.
    """
    FeNum = ExtractFormulaComponent(Data['formula'], "Fe([0-9.]*)Ni")
    NiNum = ExtractFormulaComponent(Data['formula'], "Ni([0-9.]*)$")
    Data['Fe'] = FeNum*100
    Data['Ni'] = NiNum*100
    Data['FeOverNi'] = FeNum/NiNum

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_AlloySolid.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(3,1, figsize=(9,12))
    else:
        fig, ax = plt.subplots(2,1, figsize=(9,9))
    # Basic properties subplot
    ax[0].plot(x, Data['FeOverNi'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Dimensionless')
    ax[0].set_title('Fe/Ni')
    ax[1].plot(x, Data['Fe'], x, Data['Ni'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('At %')
    ax[1].set_title('Composition')
    ax[1].legend(['Fe', 'Ni'])
    if FitCompo is not None:
        ax[2].plot(x, FitIndex)
        ax[2].set_xlabel(xtext)
        ax[2].set_ylabel('Fit index')
        ax[2].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_AlloySolidComposition.png'))

    return Data

def PlotAlloyLiquid(Data, DirName, PlotAxis='Temperature', FitCompo=None):
    """ PlotAlloyLiquid() takes the liquid metal data output by MELTS and produces a csv file and plots from it.
        Input:
            Data (pd.DataFrame): The data for this phase in this MELTS simulation.
            DirName (str): Location to put output files.
            PlotAxis (str): Column name which comprises the x-axis of plots.
            FitCompo (dict): Oxide Wt% values to use to compute a fit index.  If None, no fit index is computed.
                The format is like: {'SiO2': 45.05, 'MgO': 30.50 ...}
        Output:
            AlloyLiquid.csv: CSV file containing all the information.
            AlloyLiquid.png: Plot of the phases as a function of temperature.
    """
    FeNum = ExtractFormulaComponent(Data['formula'], "Fe([0-9.]*)Ni")
    NiNum = ExtractFormulaComponent(Data['formula'], "Ni([0-9.]*)$")
    Data['Fe'] = FeNum*100
    Data['Ni'] = NiNum*100
    Data['FeOverNi'] = FeNum/NiNum

    # If a composition has been given for computing the fit index, then we will compute it now.
    if FitCompo is not None:
        FitIndex = ComputeFitIndex(Data, FitCompo)
        Data['FitIndex'] = FitIndex

    # Now output a csv to disk for future reference.
    Data.to_csv(os.path.join(DirName, 'Output_AlloyLiquid.csv'))

    # Plot the stats in some pretty charts here.
    x, xtext = GetPlotAxis(Data, PlotAxis)
    # We will either have a two pane or three pane plot depending on whether we computed a fit index.
    if FitCompo is not None:
        fig, ax = plt.subplots(3,1, figsize=(9,12))
    else:
        fig, ax = plt.subplots(2,1, figsize=(9,9))
    # Basic properties subplot
    ax[0].plot(x, Data['FeOverNi'])
    ax[0].set_xlabel(xtext)
    ax[0].set_ylabel('Dimensionless')
    ax[0].set_title('Fe/Ni')
    ax[1].plot(x, Data['Fe'], x, Data['Ni'])
    ax[1].set_xlabel(xtext)
    ax[1].set_ylabel('At %')
    ax[1].set_title('Composition')
    ax[1].legend(['Fe', 'Ni'])
    if FitCompo is not None:
        ax[2].plot(x, FitIndex)
        ax[2].set_xlabel(xtext)
        ax[2].set_ylabel('Fit index')
        ax[2].set_title('Fit index with {} DOFs'.format(len(FitCompo)))
    plt.tight_layout()
    mysavefig(os.path.join(DirName, 'Output_AlloyLiquidComposition.png'))

    return Data

def PlotCPX(CPXData, DirName):
    """

    :rtype : T (ndarray) and FitIndex (ndarray)
    """

    # Verify that MELTS hasn't changed the output since we were programmed.
    if (CPXData[0,:] != array([b'Pressure', b'Temperature', b'mass', b'S', b'H',
                               b'V', b'Cp', b'structure', b'formula', b'SiO2',
                               b'TiO2', b'Al2O3', b'Fe2O3', b'Cr2O3', b'FeO',
                               b'MnO', b'MgO', b'NiO', b'CaO'])).any():
                               # b'MnO', b'MgO', b'CaO', b'Na2O', b'K2O'])).any():
        print ("alphaMELTS output format for clinopyroxene has changed.")
        return

    # Get rid of that first header.
    ColNames = copy(CPXData[0,:].astype(str))
    CPXData = CPXData[1:,:]

    # Get rid of the formulas column
    Formulas = copy(CPXData[:,8].astype(str))
    CPXData[:,8] = 0

    # Check that the structure column always says cpx, and then get rid of it.
    if any(CPXData[:,7].astype(str) != 'cpx'):
        print ('alphaMELTS output has a clinopyroxene section for which structure is not cpx.')
        return
    CPXData[:,7] = 0

    # Convert CPXData to floats so we can do math.
    CPXData = CPXData.astype(float)

    # Convert Kelvin to celcius.
    T = CPXData[:,1].astype(float) - 273.15

    # Nice names for the columuns.
    P, SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O = [0] + range(9, 20)

    # Calculate total Fe as FeO.
    FeContent = CPXData[:,FeO] + CPXData[:,Fe2O3]*0.9
    # And Fe3+/sum(Fe)
    Fe3OverFe = CPXData[:,Fe2O3]*0.9 / FeContent
    # Add them to the CPXData
    FeContent, CPXData, ColNames = AddColumnToMineralData(FeContent, CPXData, 'All Fe as FeO', ColNames)
    Fe3OverFe, CPXData, ColNames  = AddColumnToMineralData(Fe3OverFe*100, CPXData, 'Fe$^{3+}$/$\sum$Fe$ x $100', ColNames)

    # Plot major components
    indices = [SiO2, MgO, FeContent]
    PlotMultiWtPct(T, ColNames, indices, CPXData, DirName, 'CPX Majors')

    # Plot minor components and Fe3+
    indices = [CaO, Al2O3, MnO, TiO2, Cr2O3, Na2O, Fe3OverFe]
    PlotMultiWtPct(T, ColNames, indices, CPXData, DirName, 'CPX Minors and Fe$^{3+}$')

    # Output a fit index if appropriate:
    if ('CPXCompo' in globals()) and (sum(CPXCompo) > 0):
        FitIndex = zeros(shape(CPXData)[0])
        indices = [MgO, Al2O3, CaO, Na2O, TiO2, Cr2O3] # Except for Fe.

        # Compute the fit index at this point.
        FitIndex = [abs(CPXData[:,i].astype(float)-CPXCompo[i])/CPXCompo[i] for i in indices]

        # Add in the Fe.
        FitIndex = vstack((FitIndex, abs(CPXData[:,FeContent].astype(float)-CPXCompo[FeContent])/CPXCompo[FeContent]))
        FitIndex = sum(FitIndex,0)/(len(indices)+1)

        ylabeltext = 'Fit index using Mg, Al, Ca, Na, Ti, Cr, and Fe'
        titletext ='CPX fit index'
        savefilename = 'CPXFitIndex.png'
        DrawFitIndexPlot(DirName, FitIndex, T, savefilename, titletext, ylabeltext)
    else:
        FitIndex = None

    close('all')

    return T, FitIndex

# def PlotOPX(OPXData, DirName):
    # """

    # :rtype : T (ndarray) and FitIndex (ndarray)
    # """

    # # Verify that MELTS hasn't changed the output since we were programmed.
    # if (OPXData[0,:] != array([b'Pressure', b'Temperature', b'mass', b'S', b'H',
    #                            b'V', b'Cp', b'structure', b'formula', b'SiO2',
    #                            b'TiO2', b'Al2O3', b'Fe2O3', b'Cr2O3', b'FeO',
    #                            b'MnO', b'MgO', b'NiO', b'CaO'])).any():
    #     print ("alphaMELTS output format for orthopyroxene has changed.")
    #     return

    # # Get rid of that first header.
    # ColNames = copy(OPXData[0,:].astype(str))
    # OPXData = OPXData[1:,:]

    # # Get rid of the formulas column
    # Formulas = copy(OPXData[:,8])
    # OPXData[:,8] = 0

    # # Check that the structure column always says cpx, and then get rid of it.
    # if any(OPXData[:,7].astype(str) != 'opx'):
    #     print ('alphaMELTS output has an orthopyroxene section for which structure is not opx.')
    #     return
    # OPXData[:,7] = 0

    # # Convert OPXData to floats so we can do math.
    # OPXData = OPXData.astype(float)

    # # Convert Kelvin to celcius.
    # T = OPXData[:,1].astype(float) - 273.15

    # # Nice names for the columuns.
    # P, SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O = [0] + list(range(9, 20))

    # # Calculate total Fe as FeO.
    # FeContent = OPXData[:,FeO] + OPXData[:,Fe2O3]*0.9
    # # And Fe3+/sum(Fe)
    # Fe3OverFe = OPXData[:,Fe2O3]*0.9 / FeContent
    # # Add them to the OPXData
    # FeContent, OPXData, ColNames = AddColumnToMineralData(FeContent, OPXData, 'All Fe as FeO', ColNames)
    # Fe3OverFe, OPXData, ColNames  = AddColumnToMineralData(Fe3OverFe*100, OPXData, 'Fe$^{3+}$/$\sum$Fe$ x $100', ColNames)

    # # Plot major components
    # indices = [SiO2, MgO, FeContent]
    # PlotMultiWtPct(T, ColNames, indices, OPXData, DirName, 'OPX Majors')

    # # Plot minor components and Fe3+
    # indices = [CaO, Al2O3, MnO, TiO2, Cr2O3, Na2O, Fe3OverFe]
    # PlotMultiWtPct(T, ColNames, indices, OPXData, DirName, 'OPX Minors and Fe$^{3+}$')

    # # # Output a fit index if appropriate:
    # # if 'OPXCompo' in globals() and (sum(OPXCompo) > 0):
    # #     FitIndex = zeros(shape(OPXData)[0])
    # #     indices = [MgO, Al2O3, CaO, Na2O, TiO2, Cr2O3] # Except for Fe.

    # #     # Compute the fit index at this point.
    # #     FitIndex = [abs(OPXData[:,i].astype(float)-OPXCompo[i])/OPXCompo[i] for i in indices]

    # #     # Add in the Fe.
    # #     FitIndex = vstack((FitIndex, abs(OPXData[:,FeContent].astype(float)-OPXCompo[FeContent])/OPXCompo[FeContent]))
    # #     FitIndex = sum(FitIndex,0)/(len(indices)+1)

    # #     ylabeltext = 'Fit index using Mg, Al, Ca, Na, Ti, Cr, and Fe'
    # #     titletext ='OPX fit index'
    # #     savefilename = 'OPXFitIndex.png'
    # #     DrawFitIndexPlot(DirName, FitIndex, T, savefilename, titletext, ylabeltext)
    # # else:
    # #     FitIndex = None
    # FitIndex=None

    # close('all')

    # return T, FitIndex

# def PlotFeldspar(FeldsparData, DirName):
    # """

    # :rtype : T (ndarray) and FitIndex (ndarray)
    # """

    # # Verify that MELTS hasn't changed the output since we were programmed.
    # if (FeldsparData[0,:] != array(['Pressure', 'Temperature', 'mass', 'S', 'H',
    #                            'V', 'Cp', 'formula', 'SiO2', 'TiO2',
    #                            'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO', 'MnO',
    #                            'MgO', 'CaO', 'Na2O', 'K2O'])).any():
    #     print ("alphaMELTS output format for feldspar has changed.")
    #     return

    # # Get rid of that first header.
    # ColNames = copy(FeldsparData[0,:])
    # FeldsparData = FeldsparData[1:,:]

    # # Get rid of the formulas column
    # Formulas = copy(FeldsparData[:,7])
    # FeldsparData[:,7] = 0

    # # Convert FeldsparData to floats so we can do math.
    # FeldsparData = FeldsparData.astype(float)

    # # Convert Kelvin to celcius.
    # T = FeldsparData[:,1].astype(float) - 273.15

    # # Nice names for the columuns.
    # P, SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O = [0] + range(8, 19)

    # # Plot major components
    # indices = [SiO2, Al2O3, CaO, Na2O, K2O]
    # PlotMultiWtPct(T, ColNames, indices, FeldsparData, DirName, 'Feldspar Majors')

    # # # Plot minor components and Fe3+
    # # indices = [CaO, Al2O3, MnO, TiO2, Cr2O3, Na2O, Fe3OverFe]
    # # PlotMultiWtPct(T, ColNames, indices, FeldsparData, DirName, 'Feldspar Minors and Fe$^{3+}$')

    # # Output a fit index if appropriate:
    # if ('FeldsparCompo' in globals()) and (sum(FeldsparCompo) > 0):
    #     FitIndex = zeros(shape(FeldsparData)[0])
    #     indices = [Al2O3, CaO, Na2O]

    #     # Compute the fit index at this point.
    #     FitIndex = [abs(FeldsparData[:,i].astype(float)-FeldsparCompo[i])/FeldsparCompo[i] for i in indices]
    #     FitIndex = sum(FitIndex,0)/(len(indices)+1)

    #     ylabeltext = 'Fit index using Al, Ca, Na'
    #     titletext ='Feldspar fit index'
    #     savefilename = 'FeldsparFitIndex.png'
    #     DrawFitIndexPlot(DirName, FitIndex, T, savefilename, titletext, ylabeltext)
    # else:
    #     FitIndex = None

    # close('all')

    # return T, FitIndex

# def PlotNepheline(NephelineData, DirName):
    # """

    # :rtype : T (ndarray) and FitIndex (ndarray)
    # """

    # # Verify that MELTS hasn't changed the output since we were programmed.
    # if (NephelineData[0,:] != array(['Pressure', 'Temperature', 'mass', 'S', 'H',
    #                                 'V', 'Cp', 'formula', 'SiO2', 'TiO2',
    #                                 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO', 'MnO',
    #                                 'MgO', 'CaO', 'Na2O', 'K2O'])).any():
    #     print ("alphaMELTS output format for nepheline has changed.")
    #     return

    # # Get rid of that first header.
    # ColNames = copy(NephelineData[0,:])
    # NephelineData = NephelineData[1:,:]

    # # Get rid of the formulas column
    # Formulas = copy(NephelineData[:,7])
    # NephelineData[:,7] = 0

    # # Convert NephelineData to floats so we can do math.
    # NephelineData = NephelineData.astype(float)

    # # Convert Kelvin to celcius.
    # T = NephelineData[:,1].astype(float) - 273.15

    # # Nice names for the columuns.
    # P, SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O = [0] + range(8, 19)

    # # Plot major components
    # indices = [SiO2, Al2O3, CaO, Na2O, K2O]
    # PlotMultiWtPct(T, ColNames, indices, NephelineData, DirName, 'Nepheline Majors')

    # # # Plot minor components and Fe3+
    # # indices = [CaO, Al2O3, MnO, TiO2, Cr2O3, Na2O, Fe3OverFe]
    # # PlotMultiWtPct(T, ColNames, indices, NephelineData, DirName, 'Nepheline Minors and Fe$^{3+}$')

    # # Output a fit index if appropriate:
    # if ('NephelineCompo' in globals()) and (sum(NephelineCompo) > 0):
    #     FitIndex = zeros(shape(NephelineData)[0])
    #     indices = [Al2O3, CaO, Na2O, K2O]

    #     # Compute the fit index at this point.
    #     FitIndex = [abs(NephelineData[:,i].astype(float)-NephelineCompo[i])/NephelineCompo[i] for i in indices]
    #     FitIndex = sum(FitIndex,0)/(len(indices)+1)

    #     ylabeltext = 'Fit index using Al, Ca, Na and K'
    #     titletext ='Nepheline fit index'
    #     savefilename = 'NephelineFitIndex.png'
    #     DrawFitIndexPlot(DirName, FitIndex, T, savefilename, titletext, ylabeltext)
    # else:
    #     FitIndex = None

    # close('all')

    # return T, FitIndex

if __name__ == '__main__':
    # print(platform.python_version())
    # print(os.system("which python"))
    # print(os.system("pwd"))


    # Create a dictionary for each phase that we want to include in a fit index.
    # Each phase has a dictionary for all the oxides to include.
    TargetCompositions = dict()
    TargetCompositions['Olivine'] = {'SiO2':41.626, 'MgO':48.536, 'FeO':7.849}#, 'MnO':1.494, 'CaO':0.101, 'Cr2O3':0.394}
    TargetCompositions['Orthopyroxene'] = {'SiO2':54.437, 'MgO':31.335, 'FeO':4.724}
    # TargetCompositions['Alloy-Liquid'] = {'Fe':91.428, 'Ni':8.572}
    TargetCompositions['Liquid'] = {'SiO2':48.736, 'MgO':25.867}
    
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
        Dirs = glob(os.path.join(ThisDir, '../ComputeScratchSpace/*/'))

        @dask.delayed
        def DoOneDir(DirName):
            print('Dask for {}'.format(DirName))
            ProcessAlphaMELTS(DirName=DirName, TargetCompositions=TargetCompositions)
        Computes = [DoOneDir(Dir) for Dir in Dirs]
        dask.compute(Computes)
