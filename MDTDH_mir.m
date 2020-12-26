clear all;
warning off; 
clc;

%% Load data
load './datasets/mir_cnn.mat';

%% Parameters setting
run = 5;
para.MAX_iter = 30;
para.bits = 16;
para.beta = 1000;
para.lambda = 5000;
sigmaI = 95;
sigmaT = 60;
para.omega1 = 0.5;
para.omega2 = 0.5;
para.theta1 = 0.5;
para.theta2 = 0.5;
map = zeros(run, 1);

%% Display parameter settings
fprintf('%d-bits: MAX_iter = %d, beta = %.4f, lambda = %.5f, sigmmaI = %d, sigmmaT = %d\n', ...
        para.bits, para.MAX_iter, para.beta, para.lambda, sigmaI ,sigmaT);
    
%% Data preparing
n_train = size(I_tr, 1);
n_anchors = 1000;
sample = randsample(n_train, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);

%% nonlinear mapping
Phi_trI = exp(-sqdist(I_tr, anchorI)/(2*sigmaI*sigmaI));
Phi_trT = exp(-sqdist(T_tr, anchorT)/(2*sigmaT*sigmaT));
Phi_testI = exp(-sqdist(I_te, anchorI)/(2*sigmaI*sigmaI));
Phi_testT = exp(-sqdist(T_te, anchorT)/(2*sigmaT*sigmaT));
Phi_dbI = exp(-sqdist(I_db, anchorI)/(2*sigmaI*sigmaI));
Phi_dbT = exp(-sqdist(T_db, anchorT)/(2*sigmaT*sigmaT));

%% centralization
Phi_trI = bsxfun(@minus, Phi_trI, mean(Phi_trI, 1));
Phi_trT = bsxfun(@minus, Phi_trT, mean(Phi_trT, 1));
Phi_testI = bsxfun(@minus, Phi_testI, mean(Phi_testI, 1));
Phi_testT = bsxfun(@minus, Phi_testT, mean(Phi_testT, 1));
Phi_dbI = bsxfun(@minus, Phi_dbI, mean(Phi_dbI, 1));
Phi_dbT = bsxfun(@minus, Phi_dbT, mean(Phi_dbT, 1));

for i = 1 : run
    I_temp = Phi_trI';
    T_temp = Phi_trT';
    Im_te =  Phi_testI';
    Te_te = Phi_testT';
    Im_db = Phi_dbI';
    Te_db = Phi_dbT';
    
    %% Training model
    [P1, P2] = solveMDTDH(I_temp, T_temp, para);
    
    %% Test model
    [Y_te] = queryMDTDH(Im_te, Te_te, P1,P2, para.bits)';
    [Y_db] = queryMDTDH(Im_db, Te_db, P1,P2, para.bits)';
    
    %% evaluate
    B_db   = compactbit(Y_db>0);
    B_test = compactbit(Y_te>0);
    Dhamm  = hammingDist(B_db, B_test);
    [map_tmp] = perf_metric4Label(L_db, L_te, Dhamm);
    map(i) = map_tmp;
    topN = TOPK(Dhamm,L_db, L_te, para.bits);
% 	save my_topk_mir_16 topN
	fprintf('=========== run = %d, mAP = %.4f ===========\n', i, map_tmp);
end
fprintf('*********** bits = %d, mAP : %.4f ***********\n',para.bits, mean(map));
