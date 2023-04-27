# broad_neutrino

This is an adaptation of the program [DayaBaySterileDecoherence](https://github.com/Harvard-Neutrino/DayaBaySterileDecoherence) (written by Toni Bertólez-Martínez) with additional [broad sterile neutrino model](https://arxiv.org/abs/2209.11270) implemented.

This program is designed to analyse different low-energy sterile neutrino experiments, and which is connected to the paper [here](https://arxiv.org/abs/2201.05108). Namely, the program is capable of analysing the [DayaBay](https://arxiv.org/abs/1610.04802v1), [NEOS](https://arxiv.org/abs/1610.05134v4), [PROSPECT](https://arxiv.org/abs/2006.11210v2) and [BEST](https://arxiv.org/abs/2109.11482v1) data. The analyses focus on studying sterile neutrino oscillations, addressing possible differences between the delta function neutrino mass states and the 3 delta function + 1 "broadened" sterile states formalism. However, this program can be extended to arbitrary oscillation formulas.

In this ReadMe we briefly introduce the program and the different analyses and plots one can do with it.

## Before starting
In order for most scripts to work, one needs to compile the following file:

```
user@user: ~/broad_neutrino$ cd Common_cython
user@user: ~/broad_neutrino/Common_cython$ sh compile.sh
```

This will prepare the necessary libraries, which are written in Cython. You will need `python3-dev` in order to run these files. This same command must be done each time a modification is done to the file `Models.pyx` or `HuberMullerFlux.py`.

## General structure
The main folder of the program contains 5+2 directories. Namely:
 - BEST, DayaBay, GlobalFit, NEOS, PROSPECT: the analysis of the respective experiments.
 - Common_cython: contains the libraries which are used for every experiment, written in Cython for a better performance.
 - Misc: some miscellaneous files to plot some interesting graphics, namely those of our paper.
All the Python files `.py` can be run with the `python3` command on your terminal.

```
user@user: ~/broad_neutrino$ cd NEOS
user@user: ~/broad_neutrino/NEOS$ python3 FitPlotsBroad.py
```
or
```
user@user: ~/broad_neutrino$ python3 NEOS/FitPlotsBroad.py
```

## Experiment analysis directories
This section describes the content inside BEST, DayaBay, GlobalFit, NEOS and PROSPECT directories. For clarity, we will use NEOS as the example.

`NEOS.py` is the main file. It defines a class with all the methods necessary to compute event expectations and compare them with the data using a test statistic. Usually this program is never compiled, but accessed through the rest of the programs.

`NEOSData.py` and `NEOSParameters.py` are auxiliary files to `NEOS.py` which read all the data of the experiment (found in the subdirectory `/Data`) and  holds the parameters of the analysis. This allows for a better tuning and easier variation of the numbers.

`EventExpectationPlots.py` is, as the same name says, a program to compute event expectation and plot them different ways: measured spectrum, ratio to the standard oscillations, the value of the test statistic per bin... The figures are saved in the subdirectory `/Figures`

`FitClass.py` defines a new class `BroadFit` to ease the task of computing the test statistic for different values of the mass and the mixing. This class is then called by `BroadFitTable.py`, which produces 4 numpy array files that stores the value of mass, mixing, breadth and statistic in the subdirectory `/PlotData`, namely `BroadSterileMass_frac.npy`, `BroadSterileAngle_frac.npy`, `BroadSterileb_frac.npy`, and `BroadSterileChi2_frac.npy`.

Note that `BroadFitTable.py` is now optimized for multi-core parallel computation.

`FitPlotsBroad.py` draws the exclusion contours using the data written by these files, and found in `/PlotData/*.npy`.

Specifically, the data files being used for standard delta function contours are `PWSterileMass_noint.npy`(DayaBay and PROSPECT) and `PWSterileMass_int.npy`(NEOS). For broad sterile model, the data being used are `BroadSterileChi2_0.08-2.npy` and `BroadSterileChi2_2-10.npy` in all experiments.

`/PlotData/read.py` reads the array in `*.npy` and its shape.

`FitPlotsChi2b.py` produces chi2 vs. b plot while marginalizing over mass and mixing.

## Common
The `Common_cython` directory includes different programs which are required by all other programs and analysis.

`HuberMullerFlux.pyx` defines a class which returns the Huber-Muller flux for some given nuclear isotopes.

`InverseBetaDecayCrossSection.py` defines different functions to be compute the IBD cross-sections, as its own name states.

`Models.pyx` defines different classes of models which define different oscillation probabilities. For example, there is a class for the standard oscillations (with parameters from nu-fit.org), or some classe for oscillations with a sterile neutrino. The only requirement for this classes is that they have a method `oscProbability` and `oscProbability_av` which return the full and averaged oscillation probabilities at distance L and energy E, respectively. This program has implemented 3 new classes: `BroadSterile`, `BroadSterileFrac`, and `BroadSterileNull`.

`BroadSterile` calculates probability with sterile neutrino with breadth `b`.
`BroadSterileFrac` calculates probability with sterile neutrino with fractional breadth `bfrac`.
`BroadSterileNull` calculates the null hypothesis when oscillation parameters and b all equal to 0.


## Miscellaneous
Finally, the `Misc` folder contains different and diverse files.

`FitPlotsGlobal.py` provides a 2 sigma contour comparison of standard 3 neutrinos and 3+1 broad model under `Misc/GlobalFitPlots/`.

`2sigmaComparison.py` produces 2 sigma contours of all reactor experiments under `Misc/2sigmaPlots`.

`Plot Neutrino Experiments.ipynb` produces the position in (L,E) of relevant neutrino oscillation experiments, and the relevant scales.

`SuperFitPlots.py` computes the total chi2 of all nuclear reactor experiments and of BEST, and plots the exclusion contours and preferred regions. These are saved in the subdirectory `/AllFitFigures`.

`Probability.py` is simply used to draw toy plots of the oscillation probabilities defined in `Models.py`.
