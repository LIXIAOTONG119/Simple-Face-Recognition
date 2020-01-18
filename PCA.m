%% load image
close all;
clear all;
%load dataset
load('trainset.mat');
load('testset.mat');
load('Self_trainset.mat');
load('Self_testset.mat');
%data preprocessing
trainset = [trainset;Self_trainset];
r=randperm( size(trainset,1) );   %randomly generate array
train_set=trainset(r, :);
train_data=train_set(:,1:1024);%random
train_labels=train_set(:,1025);
test_set=[testset;Self_testset];
test_data=test_set(:,1:1024);
test_labels=test_set(:,1025);
num=500;

%% calculate the coveriance matrix
num_example=size(train_data,1);
num_example_test=size(test_data,1);
num_pixel=size(train_data,2);
mean_train=mean(train_data);
mean_test=mean(test_data);
X = train_data-mean_train;
X_test=test_data-mean_train;
PIE_test=X_test(1:1020,:);
Self_test=X_test(1021:1023,:);
S=X'*X;

%% compute SVDs and find the eigenvalue
[E, U, V]=svd(S);
lameda=diag(U);

%% visualize the data to 2D
E_2=E(:,1:2);
train_2=X*E_2;
figure(1)
train_label=train_labels(1:num,:);
k=find(train_label==69);
scatter(train_2(k,1),train_2(k,2),45,'h','k','filled'),hold on;
text(train_2(k,1),train_2(k,2),'\leftarrow Selfies');
train_2(k,:)=[];
train_label(k,:)=[];
gscatter(train_2(1:num-size(k,1),1),train_2(1:num-size(k,1),2),train_label(1:num-size(k,1),1));

%% visualize the data to 3D
E_3=E(:,1:3);
train_3=X*E_3;
figure(2)
train_label=train_labels(1:num,:);
k=find(train_label==69);
scatter3(train_3(k,1),train_3(k,2),train_3(k,3),45,'d','k','filled'),hold on;
text(train_3(k,1),train_3(k,2),train_3(k,3),'\leftarrow Selfies');
train_3(k,:)=[];
train_label(k,:)=[];
scatter3(train_3(1:num-size(k,1),1),train_3(1:num-size(k,1),2),train_3(1:num-size(k,1),3),[],train_label(1:num-size(k,1),1),'filled');

%% visualize the eigenface of different dimension
figure(3)
eigen=reshape(E(:,1),32,32);
imshow(eigen',[]);
figure(4)
eigen=reshape(E(:,2),32,32);
imshow(eigen',[]);
figure(5)
eigen=reshape(E(:,3),32,32);
imshow(eigen',[]);

%% reduce the dimension of 1024 to 40 80 200
D = [40 80 200];
for i = 1:3
    E_d=E(:,1:D(i));
    train_d=X*E_d;
    PIE_test_d=PIE_test*E_d;
    index_d=knnsearch(train_d,PIE_test_d);
    predict_labels_d=train_labels(index_d);
    accuracy_d(i,1)=sum(predict_labels_d==test_labels(1:1020,:))/1020;
   
    Self_test_d=Self_test*E_d;
    sindex_d=knnsearch(train_d,Self_test_d);
    predict_labels_selfie_d=train_labels(sindex_d);
    accuracy_d(i,2)=sum(predict_labels_selfie_d==test_labels(1021:1023,:))/3;
    display(['When reduced dimension=', num2str(D(i)),', Accuracy on PIE set = :', num2str(accuracy_d(i,1)),', Accuracy on Selfie set = :', num2str(accuracy_d(i,2))]);
end

