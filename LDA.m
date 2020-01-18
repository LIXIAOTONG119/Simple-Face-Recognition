close all;
clear all;
%load dataset
load('trainset.mat');
load('testset.mat');
load('Self_trainset.mat');
load('Self_testset.mat');

%data preprocessing as train_data train_labels test_data test_labels
trainset = [trainset;Self_trainset];
r=randperm( size(trainset,1) );   
train_set=trainset(r, :);
train_data=train_set(:,1:1024);%random
train_labels=train_set(:,1025);
test_set=[testset;Self_testset];
test_data=test_set(:,1:1024);
test_labels=test_set(:,1025);
num=500;
num_example=size(train_data,1);
num_pixel=size(train_data,2);

%% compute Sw and Sb matrix
%Compute Mean
mean_train=mean(train_data);           %Mean of all data
STD=std(train_data,[],'all');           %Mean of all data

Mu=zeros(21,1024);                 %Mean of each class 
Sw=zeros(1024,1024);
Sb=zeros(1024,1024);
class=[1,11,15,27,21,28,63,39,37,35,23,41,34,19,20,46,55,58,6,50,69];   %the folder number I choose

for i=1:21
    index=find(train_labels==class(i));
    X=train_data(index,:);
    Mu(i,:)=mean(X);
    
    %Compute Sw & Sb
    Si=(X-Mu(i,:))'*(X-Mu(i,:))./size(X,1);
    Sw=Sw+size(X,1)./num_example.*Si;
    Sb=Sb+size(X,1)./num_example.*(Mu(i,:)-mean_train)'*(Mu(i,:)-mean_train);
end

%calculate the eigenvalue of Sw-1Sb matrix
S=pinv(Sw)*Sb;
%E contain eigenvector and S contain eigenvalue 
[E, S, V]=svd(S);
X = train_data-mean_train; 
X_test=test_data-mean_train;
PIE_test=X_test(1:1020,:);
Self_test=X_test(1021:1023,:);

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

%% visualize the projected data to 3D   
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

%% project the data to 9 15 30 dimension
D=[2,3,9,15,30];
for i=1:5
    dim = D(i);
    E_d = E(:,1:dim);
    train_d = X*E_d;
    
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
