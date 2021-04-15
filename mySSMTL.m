function results = mySSMTL(X,Y,T,D_lab_test,predictors,X_u)

addpath('MTL/')
addpath('MTL/utils/')
addpath('SLA/')
mex flsa.c;

rng(1); %Fix random seed

in = 5; % Internal fold
out = 10; % External fold

% Hyperparameters grid-search
rho1 = [10^-6 10^-5 10^-4 10^-3 10^-2 10^-1]; 
rho2 = [10^-5 10^-4 10^-3 10^-2 10^-1 1];
rho3 = 10^-2;

M = 300; % Percentage of data-augmentation
k = 1; % Number of nearest neighbors to consider while performing augmentation

frac = 1; % Select the fraction of labeled training set (default frac = 1)

idx_ext = crossvalind('Kfold',Y{1},out);
for i = 1:out
    disp('Fold:')
    disp(i)
    for t = 1:T 
    train_ext_pre{t,1} = X{t,1}(idx_ext~=i,:);
    labtrain_ext_pre{t,1} = Y{t,1}(idx_ext~=i);

    %% Processing the fraction of labeled training set
    if frac ~= 1
    if t == 1
    ind_neg = find(labtrain_ext_pre{t,1}==-1);
    ind_pos=find(labtrain_ext_pre{t,1}==1);
    randx_neg = sort(randperm(length(ind_neg),round(frac*length(ind_neg))));
    randx_pos = sort(randperm(length(ind_pos),round(frac*length(ind_pos))));
    ind_neg=ind_neg(randx_neg,1);
    ind_pos=ind_pos(randx_pos,1);
    ind_tot=sort([ind_neg;ind_pos]);
    end
    train_ext_pre{t,1} = train_ext_pre{t,1}(ind_tot,:);
    labtrain_ext_pre{t,1} = labtrain_ext_pre{t,1}(ind_tot,1);
    end
    %%
    
    labtest_ext{t,1} = Y{t,1}(idx_ext==i);
    test_ext{t,1} = X{t,1}(idx_ext==i,:);   
        
    %% SMOTE start
    a = sum(labtrain_ext_pre{1,1}==-1);
    b = sum(labtrain_ext_pre{1,1}==1);
    if a<b
    auxilium = (find(labtrain_ext_pre{t,1}==1))';
    else
    auxilium = (find(labtrain_ext_pre{t,1}==-1))';
    end
    train_ext_control{t,1} = train_ext_pre{t,1};
    train_ext_control{t,1}(auxilium,:) = [];
    X_aux{t,1} = train_ext_control{t,1};
    X_smote{t,1} = mySMOTE(X_aux{t,1}, M, k);
    train_ext{t,1} = [train_ext_pre{t,1}; X_smote{t,1}];
    if a<b
    labtrain_ext{t,1} = [labtrain_ext_pre{t,1};zeros(size(X_smote{t,1},1),1)-1];
    diff = abs(sum(labtrain_ext{t,1}==-1)-sum(labtrain_ext{t,1}==1));
    rand_idx_ext = sort(randperm(size(X_smote{t,1},1),diff));
    X_smote{t,1}(rand_idx_ext,:) = [];
    train_ext{t,1} = [train_ext_pre{t,1}; X_smote{t,1}];
    labtrain_ext{t,1} = [labtrain_ext_pre{t,1};zeros(size(X_smote{t,1},1),1)-1];
    
    else
    labtrain_ext{t,1} = [labtrain_ext_pre{t,1};ones(size(X_smote{t,1},1),1)];
    diff = abs(sum(labtrain_ext{t,1}==-1)-sum(labtrain_ext{t,1}==1));
    rand_idx_ext = sort(randperm(size(X_smote{t,1},1),diff));
    X_smote{t,1}(rand_idx_ext,:) = [];
    train_ext{t,1} = [train_ext_pre{t,1}; X_smote{t,1}];
    labtrain_ext{t,1} = [labtrain_ext_pre{t,1};ones(size(X_smote{t,1},1),1)];
    end
    %% SMOTE end
    
    %% Normalization ext start 
    [train_ext_norm_esito{t,1}, mu_ext{t,1}, sigma_ext{t,1}] = zscore(train_ext{t,1}(:,end-D_lab_test:end));
    sigma_ext{t,1}(sigma_ext{t,1}==0) = eps;
    CC_ext{t,1} = bsxfun(@minus, test_ext{t,1}(:,end-D_lab_test:end), mu_ext{t,1});
    CCu_ext{t,1} = bsxfun(@minus, X_u{t,1}(:,end-D_lab_test:end), mu_ext{t,1});
    test_ext_norm_esito{t,1} = bsxfun(@rdivide, CC_ext{t,1}, sigma_ext{t,1});
    X_u_norm_esito{t,1} = bsxfun(@rdivide, CCu_ext{t,1}, sigma_ext{t,1});
    
    train_ext_norm{t,1} = [train_ext{t,1}(:,1:end-(D_lab_test+1)) train_ext_norm_esito{t,1}];
    test_ext_norm{t,1} = [test_ext{t,1}(:,1:end-(D_lab_test+1)) test_ext_norm_esito{t,1}];
    X_u_norm{t,1} = [X_u{t,1}(:,1:end-(D_lab_test+1)) X_u_norm_esito{t,1}];
    %% Normalization ext end
    end
    
    number_of_labeled_train=length(labtrain_ext{1,1});
 
    %% SLA start
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [x_l, y_l] = SLA_SVMlasso_majvot(train_ext_norm, labtrain_ext, X_u_norm);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    number_of_labeled_train_after_SLA = length(y_l{1,1});
    %% SLA end   
      
    %% RANDOM UNDERSAMPLING pseudolabels start
    if number_of_labeled_train ~= number_of_labeled_train_after_SLA
    for t = 1:T
    y_l_pre{t,1} = y_l{t,1}(1:number_of_labeled_train);
    x_l_pre{t,1} = x_l{t,1}(1:number_of_labeled_train,:);
    y_l_post{t,1} = y_l{t,1}(number_of_labeled_train+1:end);
    x_l_post{t,1} = x_l{t,1}(number_of_labeled_train+1:end,:);
    end  
    e = sum(y_l_post{1,1}==-1);
    f = sum(y_l_post{1,1}==1);
    if e<f
    maj = (find(y_l_post{1,1}==1))';
    min = (find(y_l_post{1,1}==-1))';
    else
    maj = (find(y_l_post{1,1}==-1))';
    min = (find(y_l_post{1,1}==1))';   
    end
    randIdcs = sort(randperm(length(maj),round(length(min))));
    maj = maj(randIdcs);
    maj = [maj min];
    maj = sort(maj);
    for t = 1:T
    y_l_post{t,1} = y_l_post{t,1}(maj);
    x_l_post{t,1} = x_l_post{t,1}(maj,:);  
    y_l_pre{t,1} = [y_l_pre{t,1}; y_l_post{t,1}];
    x_l_pre{t,1} = [x_l_pre{t,1}; x_l_post{t,1}];
    end
    end
    %% RANDOM UNDERSAMPLING pseudolabels end 
    
    number_of_labeled_train_after_downsampling = size(y_l_pre{1,1},1);
    
    %% Normalization ext bis start
    for t = 1:T
    [x_l_pre_norm_esito{t,1}, mu_bis{t,1}, sigma_bis{t,1}] = zscore(x_l_pre{t,1}(:,end-D_lab_test:end));
    sigma_bis{t,1}(sigma_bis{t,1}==0) = eps;
    CC_bis{t,1} = bsxfun(@minus, test_ext_norm{t,1}(:,end-D_lab_test:end), mu_bis{t,1});
    test_ext_norm_esito_bis{t,1} = bsxfun(@rdivide, CC_bis{t,1}, sigma_bis{t,1});
    
    x_l_pre_norm{t,1} = [x_l_pre{t,1}(:,1:end-(D_lab_test+1)) x_l_pre_norm_esito{t,1}];
    test_ext_norm_bis{t,1} = [test_ext_norm{t,1}(:,1:end-(D_lab_test+1)) test_ext_norm_esito_bis{t,1}];
    end
    %% Normalization ext bis end
    
    %% Validation start
    idx_int = crossvalind('Kfold',y_l_pre{1},in);
    for h = 1:in 
        for t = 1:T
            train_int{t,1} = x_l_pre{t,1}(idx_int~=h,:);
            labtrain_int{t,1} = y_l_pre{t,1}(idx_int~=h);
            test_int{t,1} = x_l_pre{t,1}(idx_int==h,:); 
            labtest_int{t,1} = y_l_pre{t,1}(idx_int==h);
            %% Normalization int start
            [train_int_norm_esito{t,1}, mu_int{t,1}, sigma_int{t,1}] = zscore(train_int{t,1}(:,end-D_lab_test:end));
            sigma_int{t,1}(sigma_int{t,1}==0) = eps;
            CC_int{t,1} = bsxfun(@minus, test_int{t,1}(:,end-D_lab_test:end), mu_int{t,1});
            test_int_norm_esito{t,1} = bsxfun(@rdivide, CC_int{t,1}, sigma_int{t,1});
    
            train_int_norm{t,1} = [train_int{t,1}(:,1:end-(D_lab_test+1)) train_int_norm_esito{t,1}];
            test_int_norm{t,1} = [test_int{t,1}(:,1:end-(D_lab_test+1)) test_int_norm_esito{t,1}];
            %% Normalization int end
        end
            
        for j = 1:length(rho1)
            for jj = 1:length(rho2)
                
            %% MTL start  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [W_int, C_int, ~] = Logistic_CFGLasso(train_int_norm, labtrain_int, rho1(j), rho2(jj), rho3);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            for t=1:T
            f_opt(:,t) = test_int_norm{t,1}*W_int(:,t)+C_int(1,t);
            posterior_int(:,t) = sigmf(f_opt(:,t),[1 0]);
            end
            yp_opt_mean = sign(mean(f_opt,2));
            yp_opt_mean(yp_opt_mean==0) = -1;
            %% MTL end
            
            acc_opt{h}(j,jj) = sum(yp_opt_mean==labtest_int{t,1})/numel(labtest_int{t,1});
           [macro_opt{h}(j,jj),precision_opt{h}(j,jj),recall_opt{h}(j,jj)] = my_micro_macro(yp_opt_mean,labtest_int{t,1});
           [~,~,~,AUC_opt{h}(j,jj)] = perfcurve(labtest_int{t,1},mean(posterior_int,2),1);
            f_opt = [];
            yp_opt = [];
            yp_opt_mean = [];
            posterior_int = [];
            aux_opt = [];
            end
       end      
    end
    recall_opt_mean = zeros(length(rho1), length(rho2));
    for tt = 1:in
        recall_opt_mean = recall_opt_mean+recall_opt{tt};
    end
    recall_opt_mean = recall_opt_mean/in;
    [~,l] = max(recall_opt_mean(:));
    [opt_rho1, opt_rho2] = ind2sub(size(recall_opt_mean),l);
    idx_opt_rho1(i) = opt_rho1;
    idx_opt_rho2(i) = opt_rho2;
    %% Validation end
        
    %% Training start
    [W_ext, C_ext, ~] = Logistic_CFGLasso(x_l_pre_norm, y_l_pre, rho1(opt_rho1), rho2(opt_rho2), rho3);
    weights{i} = abs(W_ext);
    pseudolabels_added(i) = number_of_labeled_train_after_downsampling-number_of_labeled_train;
    %% Training end       
    
    %% Testing start
    for t = 1:T
    f_ext(:,t) = test_ext_norm_bis{t,1}*W_ext(:,t)+C_ext(1,t);
    posterior_ext(:,t) = sigmf(f_ext(:,t),[1 0]);
    end
    yp_ext_mean = sign(mean(f_ext,2));
    yp_ext_mean(yp_ext_mean==0) = -1;

    acc_ext(i) = sum(yp_ext_mean==labtest_ext{t,1})/numel(labtest_ext{t,1});
    [macro_ext(i), precision_ext(i), recall_ext(i)] = my_micro_macro(yp_ext_mean,labtest_ext{t,1});
    CC{i,1} = confusionmat(labtest_ext{t,1},yp_ext_mean);
    [~,~,~,AUC_ext(i)] = perfcurve(labtest_ext{t,1},mean(posterior_ext,2),1);    
    %% Testing end  
    f_ext = [];
    yp_ext = [];
    aux_ext = [];
    posterior_ext = [];  
end

%% Output:
results.accTest = acc_ext;
results.macroTest = macro_ext;
results.precisionTest = precision_ext;
results.recallTest = recall_ext;
results.AUCTest = AUC_ext;
results.MEAN = [mean(acc_ext) mean(macro_ext) mean(precision_ext) mean(recall_ext) mean(AUC_ext)];

results.Conf = CC;
CC_tot=zeros(2,2);
for i = 1:out
    CC_tot = CC_tot+CC{i};
end
results.Conf_tot = CC_tot;

results.features = predictors;
