function [x_l, y_l] = SLA_SVMlasso_majvot(x_l, y_l, x_u)

%     A margin-based self-learning algorithm.
%     :param x_l: Labeled observations.
%     :param y_l: Labels.
%     :param x_u: Unlabeled data. Will be used for learning.
%     :return: Labeled and the pseudo-labeled unlabeled examples.

T = 5;
b = true;
thetas = [];
l = size(x_l{1},1);
sample_distr = 1/l*ones(l,1);
count = 0;

risk_gibbs = 0.5; % An upper bound of the Gibbs risk is set to 0.5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SLA start 
while b
count = count+1;
disp('Iteration SLA: ')
disp(count)

for t = 1:T
%% Margin estimation
Lambda = 0.001;
H = fitclinear(x_l{t,1},y_l{t,1},'Learner','svm','Solver','sparsa','Regularization','lasso','Lambda',Lambda,'Weight',sample_distr);

[labels_pre, probs_pre] = predict(H,x_u{t,1});

posterior = sigmf(probs_pre(:,1),[1 0]);
posterior(:,2) = 1-posterior(:,1);
margins(:,t) = abs(posterior(:,1)-posterior(:,2));

labels(:,t) = labels_pre;
labels_pre = [];
probs_pre = [];
posterior = [];

%% Find a threshold minimizing Bayes conditional error
theta = optimal_threshold(x_u, margins(:,t), risk_gibbs);
thetas(count,t) = theta;

%% Select patients onnly if all their observations have margins more than theta
idx{t} = find(margins(:,t)>=thetas(count,t));
end

margins = [];
idx_intersect = mintersect(idx{1,1},idx{1,2},idx{1,3},idx{1,4},idx{1,5});
labels_intersect = labels(idx_intersect,:);

%% Assign the label to selected observations
cont_yes = 0;
y_s_pre = [];
for i = 1:size(labels_intersect,1)
    if sum(labels_intersect(i,:),2)>=1
        cont_yes = cont_yes+1;
        y_s_pre(cont_yes,1) = 1;
        idx_sel(cont_yes,1) = idx_intersect(i);
    end
    if sum(labels_intersect(i,:),2)<=-1
        cont_yes = cont_yes+1;
        y_s_pre(cont_yes,1) = -1;
        idx_sel(cont_yes,1) = idx_intersect(i);
    end
end

for t = 1:T      
x_s{t,1} = x_u{t,1}(idx_sel,:);
y_s{t,1} = y_s_pre; 
end

%% Stop if there is no anything to add:
if isempty(x_s)
    b = false;
end

%% Move them from the test set to the train one
for t = 1:T
x_l{t,1} = [x_l{t,1}; x_s{t,1}];
y_l{t} = [y_l{t}; y_s{t}];
end

%% Update unlabeled observations
flag1 = size(x_u{1},1);
for t = 1:T
x_u{t}(idx_sel,:) = [];
end
flag2 = size(x_u{1},1);
s = size(x_l{1},1)-l;
sample_distr = [1/l*ones(l,1); 1/s*ones(s,1)];

idx_sel = [];
labels = [];
        
%% Stop criterion
if isempty(x_u{1}) || flag1==flag2
    b = false;
end
end
%% SLA end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end