% Date 7 Jan 2024
% Copy from Converge_New_Spec_MainFunc_CutType_DiverKernel
% Z:\111_MAIN_CODE_Proposed Methods\MinMaxManifold_2021\Code\Converge_New_Spec_MainFunc_CutType_DiverKernel.m
function [NMIs_spec,ACCs_spec,NMIs, ACCs, derror,NMIs_intra_Hstar_Spec, AResult_NMIparas, AResult_ACCparas, AResult_ARparas, AResult_Recallparas, AResult_Precisionparas, AResult_Fscoreparas, NMIs_intra_Hstar, NMIs_intra, NMIs_intra_HstarHd, H_final, Hstar_final, Hd_final, nIter_final, W_final, objhistory_final, nIteration, timeNMF_final, timeMainFunc_final] = ...
    MainFunc_MMM_2026(namematfile, datasetname, mode, normFea, normW, normH, normHstar, normHd)
% function [NMIs_spec,ACCs_spec,NMIs, ACCs, derror,NMIs_intra_Hstar_Spec, AResult_NMIparas, AResult_ACCparas, AResult_ARparas, AResult_Recallparas, AResult_Precisionparas, AResult_Fscoreparas, NMIs_intra_Hstar, NMIs_intra, NMIs_intra_HstarHd, H_final, Hstar_final, Hd_final, nIter_final, W_final, objhistory_final, nIteration, timeNMF_final] = ...
%     Converge_MainFunc_MMM_2026(namematfile, datasetname, mode, normFea, normW, normH, normHstar, normHd)
%Load data
mainTimer = tic;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(genpath(fullfile(projectRoot, 'refs')));
addpath(genpath(fullfile(projectRoot, 'datasets')));
warning off;
datasetFile = fullfile(projectRoot, 'datasets', [datasetname '.mat']);
if exist(datasetFile, 'file')
    load(datasetFile);
else
    load(datasetname);
end
timeNMF_final = NaN;
 
 
normonFea = normFea;
normonW = normW;
normonH = normH;%default value 0;
normonHd = normHd;
normonHstar = normHstar;
m = nviews; %number of views 
 
% initialise R is a matrix of mxm matrix storing inter relationships
% between object types
R = cell(m,1);
n = size(fea{1,1},1);
nfea = zeros(m,1); %nfea{1} stores the number of objects in object type 1
%nfea(1) = size(fea{1},1);
 
for i = 1:m
    nfea(i) = size(fea{i,1},2);
end
  
for i = 1:m
        temp = zeros(n,nfea(i));
        R{i,1} = temp;
 
end 
 
%% parameter setting norm or no norm, and for calculation of affinity matrix and laplacian matrix
options = [];
options.alpha = 0;
options.WeightMode = mode;  
% mode = options.WeightMode;
options.maxIter = 100;
rand('twister',5489);
 
for i = 1:m
    if normonFea == 2
        R{i,1} = NormalizeFea(fea{i,1});
    else 
        R{i,1} = fea{i,1};
    end
end
 
 
%% 
% timerun = datestr(datetime('now'));
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
filename_mat = fullfile(resultsDir, strcat(namematfile, datasetname, 'MMM','_','_FeaNorm', num2str(normonFea),'_WNorma',num2str(normonW),'_HNorm',num2str(normonH),'_', mode));
% filename_mat(regexp(filename_mat,'[.]'))=[];
% filename_mat(regexp(filename_mat,'[ ]'))=[];
% filename_mat(regexp(filename_mat,'[-]'))=[];
% filename_mat(regexp(filename_mat,'[:]'))=[];
 
fprintf('Dataset name: %s\n',datasetname);
 
nClass = length(unique(gnd)); %document cluster number
 
 
%Clustering in the original space
rand('twister',5489);
 
%First view matrix or concatenate all views to be clustered
firstView = R{1,1};
concat = [R{1,1} R{2,1}];
label = litekmeans(concat,nClass,'Replicates',20);
NMI_Kmeans = MutualInfo(gnd,label);
disp(['Clustering in the original space. NMI: ',num2str(NMI_Kmeans)]);
 
 d = [gnd, label];
%     fscore = FScr(d);
gnd1 = gnd;
labelnew = bestMap(gnd1, label);
AC_Kmeans = length(find(gnd == labelnew))/length(gnd);
 
% initialise and assign intra-type relationship
W = cell(m,1);
L = cell(m,1);
tempW = cell(m,1);
alpha = 1;
options.alpha = alpha;
options.normW = 1;
for i = 1:m
    W{i} = constructW(R{i},options);
    L{i} = constructL(W{i}, alpha, options);
end
 
H = cell(m,1);
for i = 1:m
    H{i,1} = zeros(n, nClass);
end
% G = initializeMulti_view_anyviews(R, nClass);
H = initializeMV2018(R, nClass, m);
H = H';
rand('twister',5489);
ncols = 21; 
% alphas = [0.1];
% betas = [1];
% lambdas = [1];
% ks = [10 5 15];
alphas = [0.001 ];
betas  = [0.001 ];
lambdas  = [0.05 ];

ks = [10]; %[5 10 20 30 50 100]; 
%%
options.eta = 1;
 
 
NMIs_intra = para_matrix(alphas, betas, lambdas, ks, ncols);
NMIs_intra_Hstar_Spec = para_matrix(alphas, betas, lambdas, ks, ncols);
NMIs_intra_Hstar = para_matrix(alphas, betas, lambdas, ks, ncols);
NMIs_intra_HstarHd = para_matrix(alphas, betas, lambdas, ks, ncols);
note_NMIs_intra_new = 'NMIs_intra_new has been ordered as NMI and NMI_std ACC and ACC_std F-score and F-score_std Precision and Precision_std Recall and Recall_std -- time';
 
para_coupled = [1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0];
 
NMIparas = zeros(size(NMIs_intra,1),length(para_coupled));
ACCparas = zeros(size(NMIs_intra,1),length(para_coupled));
ARparas = zeros(size(NMIs_intra,1),length(para_coupled));
Fscoreparas = zeros(size(NMIs_intra,1),length(para_coupled));
Precisionparas = zeros(size(NMIs_intra,1),length(para_coupled));
Recallparas = zeros(size(NMIs_intra,1),length(para_coupled));
 
NMIparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
ACCparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
Fscoreparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
ARparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
Recallparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
Precisionparas_std = zeros(size(NMIs_intra,1),length(para_coupled));
for iNMI = 1:size(NMIs_intra,1)
    options.alpha = NMIs_intra(iNMI,1);
    options.beta = NMIs_intra(iNMI,2);
    options.lambda = NMIs_intra(iNMI,3);
    options.k = NMIs_intra(iNMI,4);
    
    % initialise and assign intra-type relationship
    A = cell(m,1);
    tempA = cell(m,1);
    alpha = 1;
    options.normW = 1;
    
    for v = 1:m
        A{v,1} = full(constructA(R{v},options));
        L{v,1} = full(constructL(A{v}, alpha, options));
    end
    %% Calculate AMinMax and LMinMax
    
    AMinMax = MinMaxMatrices(A, m);
    manifold_type = 'Min Max';
    LMinMax = constructL(AMinMax, alpha, options);
    nSmp = size(AMinMax,1);
    DCol = full(sum(AMinMax,2));
    DMinMax = spdiags(DCol,0,nSmp,nSmp);
    
    %%
    tic
    [NMIs_spec,ACCs_spec,NMIs, ACCs, derror, H_final, Hstar_final, Hd_final, nIter_final, W_final, objhistory_final, nIteration] = MMM_func_2026(normonW, normonH, normonHd,normonHstar, gnd, m, n, R, H, LMinMax, DMinMax, AMinMax, nClass, nfea, options);
    timeNMF = toc;
    timeNMF_final = timeNMF;
    rand('twister',5489);
    H_dis = zeros(n,nClass);
    
    for t=1:m
        H_dis = H_dis + 1/m*Hd_final{t};
    end
    Hfuse = zeros(n,nClass);
    for t=1:m
        Hfuse = Hfuse + 1/m*H_final{t};
    end
    H_final = [Hfuse];
    if ~(any(any(isnan(H_final'))) || any(any(isinf(H_final'))))
        [CA F P Recall nmi AR] = performance_kmeans(H_final, nClass, gnd); 
        NMIs_intra(iNMI,[6,7]) = nmi;
        NMIs_intra(iNMI,[9,10]) = CA;
        NMIs_intra(iNMI,[12,13]) = F;
        NMIs_intra(iNMI,[15,16]) = P;
        NMIs_intra(iNMI,[18,19]) = Recall;
        NMIs_intra(iNMI,[20,21]) = AR;
        NMIs_intra(iNMI,22) = timeNMF;
    end
    
    %%H_final_2 = H_star 
  
    H_final_2 = [Hstar_final];
    if ~(any(any(isnan(H_final_2'))) || any(any(isinf(H_final_2'))))
        [CA F P Recall nmi AR] = performance_kmeans(H_final_2, nClass, gnd);
  
        NMIs_intra_Hstar(iNMI,[6,7]) = nmi;
        NMIs_intra_Hstar(iNMI,[9,10]) = CA;
        disp(['From Hstar. NMI and ACC = ', num2str(nmi(1)*100), '----', num2str(CA(1)*100)]);
        NMIs_intra_Hstar(iNMI,[12,13]) = F;
        NMIs_intra_Hstar(iNMI,[15,16]) = P;
        NMIs_intra_Hstar(iNMI,[18,19]) = Recall;
        NMIs_intra_Hstar(iNMI,[20,21]) = AR;
        NMIs_intra_Hstar(iNMI,22) = timeNMF;
    end
    %% Spectral clustering on Hstar
    if ~(any(any(isnan(H_final_2'))) || any(any(isinf(H_final_2'))))
        [CA F P Recall nmi AR] = evalResults_multiview_K(H_final_2', gnd);
        NMIs_intra_Hstar_Spec(iNMI,[6,7]) = nmi;
        NMIs_intra_Hstar_Spec(iNMI,[9,10]) = CA;
        NMIs_intra_Hstar_Spec(iNMI,[12,13]) = F;
        NMIs_intra_Hstar_Spec(iNMI,[15,16]) = P;
        NMIs_intra_Hstar_Spec(iNMI,[18,19]) = Recall;
        NMIs_intra_Hstar_Spec(iNMI,[20,21]) = AR;
        NMIs_intra_Hstar_Spec(iNMI,22) = timeNMF;
    end
    %% Use both Hc and Hd_v, H_final = [H_star Hd]
    H_final_3 = [Hstar_final H_dis];
    if ~(any(any(isnan(H_final_3'))) || any(any(isinf(H_final_3'))))
        [CA F P Recall nmi AR] = performance_kmeans(H_final_3, nClass, gnd);
 
        NMIs_intra_HstarHd(iNMI,[6,7]) = nmi;
        NMIs_intra_HstarHd(iNMI,[9,10]) = CA;
        NMIs_intra_HstarHd(iNMI,[12,13]) = F;
        NMIs_intra_HstarHd(iNMI,[15,16]) = P;
        NMIs_intra_HstarHd(iNMI,[18,19]) = Recall;
        NMIs_intra_HstarHd(iNMI,[20,21]) = AR;
        NMIs_intra_HstarHd(iNMI,22) = timeNMF;
    end
    %% Use both Hc and Hd_v, H_final = [H_star Hd] combine with a parameter
    
    for itest = 1:length(para_coupled)
        para_1 = para_coupled(itest);
        para_2 = 1 - para_1;
        H_2C = [para_1*Hstar_final para_2*H_dis];
        
        %%
        if sum(any(isnan(H_2C), 2))~=n
            [CA F P Recall nmi AR] = performance_kmeans(H_2C, nClass, gnd);
 
    %         label = litekmeans(Gstar,nClass,'Replicates',20);
 
            NMIparas(iNMI,itest) = nmi(1);
            NMIparas_std(iNMI,itest) = nmi(2);
 
            ACCparas(iNMI,itest) = CA(1);
            ACCparas_std(iNMI,itest) = CA(2);
 
            Fscoreparas(iNMI,itest) = F(1);
            Fscoreparas_std(iNMI,itest) = F(2);
 
            Recallparas(iNMI,itest) = Recall(1);
            Recallparas_std(iNMI,itest) = Recall(2);
 
            Precisionparas(iNMI,itest) = P(1);
            Precisionparas_std(iNMI,itest) = P(2);
 
            ARparas(iNMI,itest) = AR(1);
            ARparas_std(iNMI,itest) = AR(2);
        end
    end
      %% New measurement result
        AResult_NMIparas = [NMIparas NMIs_intra_HstarHd NMIparas_std];
        AResult_ACCparas = [ACCparas NMIs_intra_HstarHd ACCparas_std];
        AResult_ARparas = [ARparas NMIs_intra_HstarHd ARparas_std];
        AResult_Recallparas = [Recallparas NMIs_intra_HstarHd Recallparas_std];
        AResult_Precisionparas = [Precisionparas NMIs_intra_HstarHd Precisionparas_std];
        AResult_Fscoreparas = [Fscoreparas NMIs_intra_HstarHd Fscoreparas_std];     
        save(filename_mat,'Hstar_final','NMIs_spec','ACCs_spec','NMIs', 'ACCs', 'derror','AResult_NMIparas', 'NMIs_intra_Hstar_Spec', 'AResult_ACCparas', 'AResult_ARparas','AResult_Recallparas','AResult_Precisionparas','AResult_Fscoreparas', 'NMIs_intra','W_final', 'H_final', 'datasetname','fea','gnd', 'note_NMIs_intra_new','NMI_Kmeans', 'normonFea','normonH', 'normonW', 'normonHd','normonHstar', 'mode', 'manifold_type', 'NMIs_intra_HstarHd', 'NMIs_intra_Hstar', 'timeNMF_final');
        timeMainFunc_final = toc(mainTimer);
end
