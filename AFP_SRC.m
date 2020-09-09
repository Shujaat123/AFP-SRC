%% AFP-SRC: AntiFreeze Proteins (AFPs) Classification using Sparse Reconstruction Classifier
% Author: Shujaat Khan (shujaat123@gmail.com)

clc
clear all
close all

%% Load training and test dataset
data_dir = ['data\'];
train_file = ['train1.csv'];
test_file = ['test1.csv'];
train_db = [data_dir,train_file];
test_db = [data_dir,test_file];

X_train = csvread(train_db,1,1);
X_test = csvread(test_db,1,1);

%% Preprocessing
T = readtable(train_db,'ReadVariableNames',false);
T = table2cell(T(2:end,1));
for i = 1:length(T)
    if (strcmp(T{i},'AFP'))
        class_train(i) = 1;
    elseif (strcmp(T{i},'NON_AFP'))
        class_train(i) = 2;
    end
end

T = readtable(test_db,'ReadVariableNames',false);
T = table2cell(T(2:end,1));
for i = 1:length(T)
    if (strcmp(T{i},'AFP'))
        class_test(i) = 1;
    elseif (strcmp(T{i},'NON_AFP'))
        class_test(i) = 2;
    end
end

classes=2;
train=300;
n=classes*train;

%% Finding principal components usinf training dataset and selecting the top [num_PCs] number of PCs
num_PCs = 50;
[V,D] = eig(cov(X_train));
[D, Fr] = sort(diag(D),'descend');

%% Projecting Test data on PCs
A=(X_train*V(:,Fr(1:num_PCs)))';
test_1 = (X_test*V(:,Fr(1:num_PCs)))';


%% Sparse Representation Classification (SRC)
count=0;
TP=0;
TN=0;
total_afps = numel(find(class_test==1));
total_non_afps = numel(find(class_test==2));
for i=1:size(test_1,2)
    y=test_1(:,i);
    x0=A\y;         % initial L2 solution
    xp=l1qc_logbarrier(x0,A,[],y,0.05, 0.01); % findc L1 solution
    
delta=zeros(n,1);
score=zeros(classes,1);
k=1;

% Delta rule for reconstruction
for u=1:train:train*classes
    delta(u:u+train-1,1)=xp(u:u+train-1,1);
    
    yp=A*delta;
    r_y=sum((y-yp).^2).^0.5;
    score(k,1)=r_y;
    delta=zeros(n,1);
    k=k+1;     
end

%%%Decision rule
[value,class]=min(score);
if (class_test(i)==1)
    if class==class_test(i)
        count=count+1;
        TP = TP+1;
    end
elseif (class_test(i)==2)
    if class==class_test(i)
        count=count+1;
        TN = TN+1;
    end
end
[count i]    
[TP/total_afps TN/total_non_afps count/i]
end


%% Calculation Performance Statistics

FN = total_afps - TP;
FP = total_non_afps - TN;

Sensitivity = TP/total_afps;
Specificity = TN/total_afps;
Youden_index = Sensitivity + Specificity - 1;
Accuracy = (TP+TN)./(total_afps+total_non_afps);
Balanced_Accuracy = (Sensitivity+Specificity)/2;
F1_score = 2*TP./(2*TP + FP + FN);
MCC = ((TP.*TN) - (FP.*FN))./sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN));

%% Display Performance Statistics
display(['--------Display Performance Statistics---------'])
display(['Sensivitity: ',num2str(Sensitivity)])
display(['Specificity: ',num2str(Specificity)])
display(['Youden_index: ',num2str(Youden_index)])
display(['Accuracy: ',num2str(Accuracy)])
display(['Balanced_Accuracy: ',num2str(Balanced_Accuracy)])
display(['F1_score: ',num2str(F1_score)])
display(['MCC: ',num2str(MCC)])
