# alphaMELTSEnsemble
Code to run a large number of alphaMELTS simulations and to plot the outputs from those simulations as a combined object.

There is a quick video introduction to the code available here:  https://www.youtube.com/watch?v=AyIksocsp4M&t=810s

[alphaMELTS](https://magmasource.caltech.edu/alphameltys/) is an interface to the MELTS software for equilibrium mineral thermodynamics.  
While alphaMELTS is very capable at setting up elaborate thermochemical calculations there are some cases where it would be useful to run multiple calculations in alphaMELTS and combine them into one dataset for analysis.
For example, one alphaMELTS calculation may track the equilibrium composition of a melt as it cools and its tied to an oxygen fugacity buffer.
However, it may be useful to try the came calculation over 1000 to 10,000 different compositions in a Monte Carlo fashion, or over 10-100 compositions to explore the variation of one or two parameters.
We would call this an ensemble calculation.
This package is for setting up ensemble calculations, and recombinind the results into a single high-dimensional data cube for analysis.
Simple plots can be made from cross-sections of the hypercube, or statistical analysis can be done.
For example, if one wishes to know the errors introduced by uncertainty in the mineral composition of the inputs, one can make a Monte Carlo ensemble with the experimental error bars and plot the output compositions as a function of those inputs.
Alternately, if it is suspected that different fugacities may be at play, then an ensemble varying fo2 parameters and one or two elemental parameters can be made.

The required basic directory structure is:

alphamelts (link to location of the alphaMELTS software which should be downloaded from the CalTech site).
code/ (contains all the code for this module.)
ComputeScratchSpace/ directory which contains an ensemble calculation which will be made when running the code.

In order to set up an ensemble, use the file GenMELTSEnsemble.py.  
A simple example of a simple ensemble can be found at the bottom of the file.
Each calclulation in the ensemble will inherit from alphamelts_settings.txt, mybatch and input.melts.
alphamelts_settings.txt describes the parameters for the calculation and is described in the alphamelts documentation.
As alphamelts is a command line GUI, one normally runs it by typing a sequence of menu commands after starting the program.
mybatch is the sequence of keystrokes necessary to run the calculation so it can be automated.
Both alphamelts_settings.txt and mybatch should be edited before setting up the ensemble.
input.melts is the input file for alphamelts which varies for each computation.  This is automatically genertaed by GenMELTSEnsemble.py.

After an ensemble is set up, a number of directories will be present within ComputeScratchSpace (or another location if specified in GenMELTSEnsemble.py).  
It will be necessary to run alphaMELTS for each simulation, and a file named runall.sh is created within ComputeScratchSpace/ in order to facilitate that.
If the number of simulations is large, we recommend using GNU Parallels to distribute the load efficiently across multiple CPU cores on a fast machine.

Once the ensemble of calculations is complete, ProcessMELTS.py can be run on each of the directories to extract the numerical data from the simulation.  
Within each directory, there will be a csv file and a graphical plot for each phase, and for the phase abundances.
The csv file will contain the compositional and thermodynamic outputs from MELTS, as well as a few additional computed values based on those outputs (e.g. Fo# for olivine, or Fe3+/sum(Fe) for spinel).
If desired, a FitIndex will be computed for each phase.
FitIndex is described in [Gainsforth, Z., et al. Meteoritics & planetary science 50.5 (2015): 976-1004.](https://doi.org/10.1111/maps.12445)

