# InGene
InGene contains python software tools for Sparse Estimation of Mutual Information Landscapes (SEMIL) and postprocessing tools to correct for finite-sampling bias and to obtain Kernel Density Estimates from bootstrapping.

To construct a mutual information landscapes using the example data use:

$python3 SEMIL/landmaker.py -i example/input.txt -f l

The inputfile (input.txt) contains: (1) the set of input values as which the conditional output distributions are available, (2) the discretized conditional output distributions for all the input values in a single .csv file, and (3) information about the boundaries of the design space for the mutual information landscape.
