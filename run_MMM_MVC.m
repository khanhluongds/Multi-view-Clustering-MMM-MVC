% Demo script for MMM-MVC on the MNIST_10K_nonzeros dataset.
%
% This script uses the one-parameter setting used for the convergence
% experiment in the revised Neurocomputing manuscript.

clear;
clc;

projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(genpath(fullfile(projectRoot, 'refs')));
addpath(genpath(fullfile(projectRoot, 'datasets')));

datasetName = 'Threesources';
weightMode = 'HeatKernel';

[NMIs_spec, ACCs_spec, NMIs, ACCs, derror, NMIs_intra_Hstar_Spec, ...
    AResult_NMIparas, AResult_ACCparas, AResult_ARparas, ...
    AResult_Recallparas, AResult_Precisionparas, AResult_Fscoreparas, ...
    NMIs_intra_Hstar, NMIs_intra, NMIs_intra_HstarHd, H_final, ...
    Hstar_final, Hd_final, nIter_final, W_final, objhistory_final, ...
    nIteration] = Converge_MainFunc_MMM_2026( ...
    '2026_', datasetName, weightMode, 2, 0, 0, 0, 1);

fprintf('\nMMM-MVC finished on %s.\n', datasetName);
fprintf('Iterations: %d\n', nIteration);
fprintf('Final Hstar NMI: %.4f\n', NMIs_intra_Hstar(1, 6));
fprintf('Final Hstar ACC: %.4f\n', NMIs_intra_Hstar(1, 9));
