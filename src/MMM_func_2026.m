%Normal diver on Hd_v and Hd_s
%Orthogonal constraint on Hc and Hd_v                                                                                                                                
function [NMIs_spec,ACCs_spec,NMIs, ACCs, derror, H_final, Hstar_final, Hd_final, nIter_final, W_final, objhistory_final, nIteration] = MMM_func_2026(normonW, normonH, normonHd, normonHstar, gnd, m, n,  R, H, LMinMax, DMinMax, AMinMax, nClass, nfea, options)
if ~isfield(options,'error')
    options.error = 1e-6;
end
if ~isfield(options, 'maxIter')
    options.maxIter = [];
end
 
if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end
 
if ~isfield(options,'minIter')
    options.minIter = 30;
end
 
if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end
 
if ~isfield(options,'alpha')
    options.alpha = 100;
end
 
if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end
 
if ~exist('G1','var') %k
    H1 = [];
    H2 = [];
    W = [];
end
derror = [];   
objhistory = 0;
objhistory_final = 0;
differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;
 
% Innitialize all factor matrices
H = l1_norm(H,m);
for i = 1:m
    H_final{i} = H{i};
end
 
H_star = zeros(n,nClass);
for i = 1:m
    H_star = H_star + 1/m*H{i};
end
H_star = l1_norm_onematrix(H_star);
 
Hstar_final = H_star;
 
%Innitialize Hd to be H at the begining
Hd = cell(m,1);
for i = 1:m
    Hd{i,1} = H{i,1};
end
Hd_final = Hd;
 
%construct L based on u
u = ones(1, 1);
u = [1];
q = size(u,2);%number of W
sumu = sum(u);
for i = 1:size(u,2)
    u(:,i) = u(:,i)/sumu;
end
%% calculate and use new Lcompatible
LMinMax1 = cell(m,1);
LMinMax0 = cell(m,1);
 
LMinMax1 = zeros(size(LMinMax,1),size(LMinMax,2));
LMinMax0 = zeros(size(LMinMax,1),size(LMinMax,2));
LMinMax1 = (abs(LMinMax) + LMinMax)*0.5;
LMinMax0 = (abs(LMinMax) - LMinMax)*0.5;
 
%% end chuan bi L
 
selectInit = 1;
Rd_t = R{1,1};
nSmp = size(Rd_t,1);
mFea = size(Rd_t,2);
% Initialize the data and feature matrices
H = initializeMV2018(R, nClass, m);
 
% initialise matrix S
W = cell(m,1);
for i=1:m
    W{i}=ones(nfea(i),nClass);
end
tryNo = 0;
nIter = 0;
nIteration = 0;
ACCs = [];
NMIs = [];
ACCs_spec = [];
NMIs_spec = [];
% eta = options.eta
 
while tryNo < nRepeat
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        nIteration = nIteration + 1;
        derror(nIteration) = objhistory_final;
% %         if (mod(nIteration,20)==0)
%             [CA F P Recall nmi AR] = evalResults_multiview_K(Hstar_final', gnd);
%             ACCs_spec(nIteration) = CA(1);
%             NMIs_spec(nIteration) = nmi(1); 
% 
%             [CA F P Recall nmi AR] = performance_kmeans(Hstar_final, nClass, gnd); 
%             ACCs(nIteration) = CA(1);
%             NMIs(nIteration) = nmi(1);
% %         end
% %        
%          
%         if (nIteration==102) 
%            disp('stop');
%         end
        
        % ===================== update all W_v========================
        W = updateW(W, H, R, m, nfea, nClass);
        
        if normonW == 1
            W = l1_norm(W,m);
        else 
            if normonW == 2
                for v = 1:m
                    W{v,1} = NormalizeFea(W{v,1});
                end
            end
        end
        
%         S = l1_norm(S,m);
        % ===================== update G ~~ update all H_v========================
        H_star = updateH_star(H, Hd, H_star, DMinMax, AMinMax, LMinMax, LMinMax1, LMinMax0, m, nClass, nSmp, options);
 
%         H_star = NormalizeFea(H_star);
%         
        if normonHstar ==1
            H_star = l1_norm_onematrix(H_star);
        end
            
        H = updateH(W, H, Hd, H_star, R, m, nClass, nSmp, options);
        % ===================== update H_star ~~ update H_star ========================
        
        if normonH ==1
            H = l1_norm(H,m);
        end
        
        % ===================== update Hd_v ~~ distinct v ========================
        Hd = updateHd(H, Hd, H_star, DMinMax, m, nClass, nSmp, options);
        if normonHd ==1
            Hd = l1_norm(Hd,m);
        end
%         derror(nIteration) = objhistory_final;
% %         if (mod(nIteration,20)==0)
%             [CA F P Recall nmi AR] = evalResults_multiview_K(Hstar_final', gnd);
%             ACCs_spec(nIteration) = CA(1);
%             NMIs_spec(nIteration) = nmi(1); 
% 
%             [CA F P Recall nmi AR] = performance_kmeans(Hstar_final, nClass, gnd); 
%             ACCs(nIteration) = CA(1);
%             NMIs(nIteration) = nmi(1);
        nIter = nIter + 1;
        
        
        %  When U, V run nIter times
        if nIter > minIter
            if selectInit
                objhistory =  CalculateObj(R, W, H, Hd, H_star, LMinMax, AMinMax, m, options);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj =  CalculateObj(R, W, H, Hd, H_star, LMinMax, AMinMax, m, options);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj =  CalculateObj(R, W, H, Hd, H_star, LMinMax, AMinMax, m, options);
                        
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                        meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;%k
                        maxErr = (meanFit-newobj)/meanFit;% k
                        
                    end
                    %                     maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                                             
                    end
                end
            end
        end
        
%         end
       
    end
    
    %     When nIter achieves minIter, run the following code segment
    if tryNo == 1
        for i = 1:m
            H_final{i} = H{i};
            Hd_final{i} = Hd{i};
            W_final{i} = W{i};
        end
        Hstar_final = H_star;
        nIter_final = nIter;
        objhistory_final = objhistory;
        
    else
        if objhistory(end) < objhistory_final(end)
            for i = 1:m
                H_final{i} = H{i};
                Hd_final{i} = Hd{i};
                W_final{i} = W{i};
            end
            Hstar_final = H_star;
            nIter_final = nIter;
            objhistory_final = objhistory;
            
%         end

        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            H = initializeMV2018(R, nClass,m);
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            for i = 1:m
                H{i} = H_final{i};
                Hd{i} = Hd_final{i};
                W{i} = W_final{i};
            end
            H_star = Hstar_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
    
end
    %==========================================================================
function objhistory_final = CalculateObj(R, W, H, Hd, H_star, LMinMax, AMinMax, m, options)
    alpha = options.alpha;
    beta = options.beta;
    lambda = options.lambda;
    
    obj_NMF = 0;
    for i=1:m
        obj_NMF = obj_NMF + norm(R{i} - H{i}*W{i}','fro');
    end
 
    obj_fuse = 0;
    for i=1:m
        obj_fuse = obj_fuse + norm(H{i} - H_star - Hd{i},'fro');
    end
    %calculate diver term cost
    obj_diver = 0;
    
%% diver term on kernel
%     for v = 1:m
%         for s = 1:m
%             if s~=v 
%                 obj_diver = obj_diver + trace(Hd{v}*Hd{v}')*(Hd{s}'*Hd{s});
%             end
%         end
%     end
%% Normal diver term on Hd_v and Hd_s
    for v = 1:m
        for s = 1:m
            if s~=v 
                obj_diver = obj_diver + trace(Hd{v}*Hd{s}');
            end
        end
    end
    
    %manifold regularization
    obj_Manifold = 0;
    obj_Manifold = obj_Manifold + trace(H_star'*AMinMax*H_star);
 
    %% norm on H or not
%     obj_l2norm = 0; 
%     for v = 1:m
%         obj_l2norm = obj_l2norm + trace(H{v}'*H{v});
%     end
    
    
    objhistory_final = obj_NMF + alpha*obj_fuse + beta*obj_diver - lambda*obj_Manifold; % + beta*obj_norm;
 
 
%%    
function H = l1_norm(H,m)
    for p = 1:m
        for i = 1:size(H{p},1)
            if sum(H{p}(i,:))~= 0
                H{p}(i,:) = H{p}(i,:)/sum(H{p}(i,:));
            else
                for j = 1:size(H{p},2)
                    H{p}(i,j) = 1/(size(H{p},2));
                end
            end
        end
    end
    %%
    function H = l1_norm_onematrix(H)
         for i = 1:size(H,1)
            if sum(H(i,:))~= 0
                H(i,:) = H(i,:)/sum(H(i,:));
            else
                for j = 1:size(H,2)
                    H(i,j) = 1/(size(H,2));
                end
            end
         end
 
  %%
function W = updateW(W, H, R, m, nfea, nClass) %update all W_v
 
for v=1:m
    tempup = zeros(nfea(v),nClass);
    tempun = zeros(nfea(v),nClass);
    
    tempup = tempup + R{v}'*H{v};
    tempun = tempun + W{v}*H{v}'*H{v};
    W{v} = W{v}.*power((tempup./tempun),(0.5));
end
 
%% Update H_star
function H_star = updateH_star(H, Hd, H_star, DMinMax,  AMinMax, LMinMax, LMinMax1, LMinMax0, m, nClass, nSmp, options)
%alpha = options.alpha; 
alpha = options.alpha;
lambda = options.lambda;
 
tempup_Hstar = zeros(nSmp, nClass);
tempun_Hstar = zeros(nSmp, nClass);
 
sumH_v = zeros(nSmp, nClass);
for v = 1:m
   sumH_v = sumH_v + H{v};     
end
 
sumHd_v = zeros(nSmp, nClass);
for v = 1:m
   sumHd_v = sumHd_v + Hd{v};     
end
 
for v = 1:m
    tempup_Hstar = tempup_Hstar + alpha*sumH_v + lambda*AMinMax*H_star;
    tempun_Hstar = tempun_Hstar + alpha*sumHd_v + alpha*H_star;
end
 
%% Calculate Ortho term
%Calculate Lambda, named PP
    isOrtho = 1;
    PP = alpha*H_star'*(sumH_v - H_star - sumHd_v) + lambda*H_star'*AMinMax*H_star;
    PP = 1/2*(PP + PP');
    PP1 = (abs(PP)+PP)./2;
    PP0 = (abs(PP)-PP)./2;
    %DMinMax = eye(nSmp,nSmp);
    orthterm0 = DMinMax*H_star*PP0;
    orthterm1 = DMinMax*H_star*PP1;
    tempup_Hstar = tempup_Hstar + isOrtho*orthterm0;
    tempun_Hstar = tempun_Hstar + isOrtho*orthterm1;
    
for j = 1:size(H_star,2)
    for i = 1:size(H_star,1)
        if tempun_Hstar(i,j)~=0
            H_star(i,j) = H_star(i,j)*(tempup_Hstar(i,j)/tempun_Hstar(i,j))^(0.5);
        else
            H_star(i,j) = 0;
        end
    end
end
 
%% update all H_v
function H = updateH(W, H, Hd, H_star, R, m, nClass, nSmp, options) 
alpha = options.alpha;
beta = options.beta;
 
for v = 1:m   
    tempup = zeros(nSmp,nClass);
    tempun = zeros(nSmp,nClass);
    tempup = tempup + R{v}*W{v} + alpha*H_star + alpha*Hd{v}; 
    
%     sumH_s = zeros(nSmp, nClass);
%     for s = 1:m
%         if s~=v
%             sumH_s = sumH_s + H{s};
%         end
%     end
 
    tempun = tempun + H{v}*W{v}'*W{v}+ alpha*H{v}; % + 1/2*beta*sumH_s;
 
    for j = 1:size(H{v},2)
        for i = 1:size(H{v},1)
            if tempun(i,j)~=0
                H{v}(i,j) = H{v}(i,j)*(tempup(i,j)/tempun(i,j))^(0.5);
            else
                H{v}(i,j) = 0;
            end
        end
    end
end
%% Update all Hd_v
function Hd = updateHd(H, Hd, H_star, DMinMax, m, nClass, nSmp, options) 
alpha = options.alpha;
beta = options.beta;
 
for v = 1:m   
    tempup = zeros(nSmp,nClass);
    tempun = zeros(nSmp,nClass);
    tempup = tempup + alpha*H{v}; 
 
%% Normal diver term on Hd_v and Hd_s
%     sumHd_s = zeros(nSmp, nClass);
%     for s = 1:m
%         if s~=v
%             sumHd_s = sumHd_s + Hd{s};
%         end
%     end
%     diverTerm = 0.5*beta*sumHd_s;
    
%% Calculate Diverterm on Kernel
    sumKs = zeros(nSmp, nSmp);
    for s = 1:m
        if s~=v
            sumKs = sumKs + Hd{s}*Hd{s}';
        end
    end
    diverTerm = 0.5*beta*sumKs'*Hd{v};
    
%% Orthogonal constraint
    isOrth = 0; %No orthogonal constraint on Hd_v
    %Calculate Lambda, named PP
    PP = alpha*Hd{v}'*(H{v} - H_star + Hd{v})-Hd{v}'*diverTerm;
    PP1 = (abs(PP)+PP)./2;
    PP0 = (abs(PP)-PP)./2;
    
    orthterm0 = DMinMax*Hd{v}*PP0;
    orthterm1 = DMinMax*Hd{v}*PP1;
    
    tempup = tempup + isOrth*orthterm0;
    tempun = tempun + alpha*H_star + alpha*Hd{v} + diverTerm;
    tempun = tempun + isOrth*orthterm1;
    for j = 1:size(Hd{v},2)
        for i = 1:size(Hd{v},1)
            if tempun(i,j)~=0
                Hd{v}(i,j) = Hd{v}(i,j)*(tempup(i,j)/tempun(i,j))^(0.5);
            else
                Hd{v}(i,j) = 0;
            end
        end
    end
end
                    
