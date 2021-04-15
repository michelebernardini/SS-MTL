clc;clear;

load drug_exam_lab_test_list

%% Input required:

% X: Labeled observations
% Y: Labeled targets
% T: Number of tasks (time-windows)
% D_lab_test: Number of predictors of lab tests field
% X_u: Unlabeled observations
% predictors_drug: list of predictors of drugs field
% predictors_exam: list of predictors of exams field
% predictors_lab_test: list of predictors of lab tests field
% predictors: total list of predictors of OVERALL* field (in the following order)--> [ Gender + predictors_drug + predictors_exam + predictors_lab_test + Age]

rng(1)

m_l = 1833; % number of total labeled patients
m_u = 4996; %number of total unlabeled patients
n = 496; % number of total predictors
T = 5; % number of tasks (time-windows) 

% Create X with random positive values
X = mat2cell(randi(100,5*m_l,n),[m_l,m_l,m_l,m_l,m_l],n);

% Create X_u with random positive values
X_u = mat2cell(randi(100,5*m_u,n),[m_u,m_u,m_u,m_u,m_u],n);

% Create Y [20:80]
neg = zeros(round(0.2*m_l),1)-1; % negative label: -1
pos = ones(round(0.8*m_l),1); % positive label: +1
Y = [neg;pos];
Y = [Y Y Y Y Y];
Y=(mat2cell(Y, [m_l], [1,1,1,1,1]))';

% Create predictors
age = {'Age'};
gender = {'Gender'};
predictors = horzcat(predictors_drug', predictors_exam');
predictors = horzcat(predictors, predictors_lab_test');
predictors = horzcat(gender, predictors);
predictors = horzcat(predictors, age);

D_lab_test = length(predictors_lab_test); 


%% SS-MTL [SVMlasso - majvot - OVERALL* configuration]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results = mySSMTL(X,Y,T,D_lab_test,predictors,X_u);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
