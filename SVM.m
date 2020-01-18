close all;
clear all;
%load dataset
load('trainset.mat');
load('testset.mat');
load('Self_trainset.mat');
load('Self_testset.mat');

%data preprocessing
trainset = [trainset;Self_trainset];
r=randperm( size(trainset,1) );   
train_set=trainset(r, :);
train_data=train_set(:,1:1024);%random
train_labels=train_set(:,1025);
test_set=[testset;Self_testset];
test_data=test_set(:,1:1024);
test_labels=test_set(:,1025);
num=500;

%calculate the coveriance matrix
num_example=size(train_data,1);
num_example_test=size(test_data,1);
num_pixel=size(train_data,2);
mean_train=mean(train_data);
mean_test=mean(test_data);
X = train_data-mean_train;
X_test=test_data-mean_train;
S=X'*X;
% find the eigenvalue
[E, U, V]=svd(S);
lameda=diag(U);

% train svm model with raw data
train_data=(mapminmax(train_data'))';
test_data=(mapminmax(test_data'))';
model = svmtrain(train_labels,train_data,'-t 0 -c 0.01 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_data, model);
fprintf('accuracy with raw data and C is 0.01 is: %f\n', accuracy(1));

model = svmtrain(train_labels,train_data,'-t 0 -c 0.1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_data, model); % test the training data
fprintf('accuracy with raw data and C is 0.1 is: %f\n', accuracy(1));

model = svmtrain(train_labels,train_data,'-t 0 -c 1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_data, model); % test the training data
fprintf('accuracy with raw data and C is 1 is: %f\n', accuracy(1));

%train svm model with data which dimension is 80
E_80=E(:,1:80);
train_80=X*E_80;
test_80=X_test*E_80;
train_80=(mapminmax(train_80'))';
test_80=(mapminmax(test_80'))';
model = svmtrain(train_labels,train_80,'-t 0 -c 0.01 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_80, model);
fprintf('accuracy with K = 80 and C is 0.01 is: %f\n', accuracy(1));

model = svmtrain(train_labels,train_80,'-t 0 -c 0.1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_80, model); % test the training data
fprintf('accuracy with K = 80 and C is 0.1 is: %f\n', accuracy(1));

model = svmtrain(train_labels,train_80,'-t 0 -c 1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_80, model); % test the training data
fprintf('accuracy with K = 80 and C is 1 is: %f\n', accuracy(1));

%train svm model with data which dimension is 200
E_200=E(:,1:200);
train_200=X*E_200;
test_200=X_test*E_200;
train_200=(mapminmax(train_200'))';
test_200=(mapminmax(test_200'))';
model = svmtrain(train_labels,train_200,'-t 0 -c 0.01 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_200, model);
% test the training data
fprintf('accuracy with K = 200 and C is 0.01 is: %f\n', accuracy(1));
model = svmtrain(train_labels,train_200,'-t 0 -c 0.1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_200, model); % test the training data
fprintf('accuracy with K = 200 and C is 0.1 is: %f\n', accuracy(1));
model = svmtrain(train_labels,train_200,'-t 0 -c 1 -q');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_200, model); % test the training data
fprintf('accuracy with K = 200 and C is 1 is: %f\n', accuracy(1));
