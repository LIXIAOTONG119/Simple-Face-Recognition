clc;clf;clear;close all;

% transfer selfies to grayscale 32*32 format as PIE
% Path=dir('.\selfie\*.jpg');
% for i=1:10
%     I=imread(strcat('.\selfie\',Path(i).name));
%     I=rgb2gray(imresize(I,[32,32]));%generate grayscale format
%     jpgname=['selfie',num2str(i),'.jpg'];
%     imwrite(I,jpgname);
% end
%load image from folder
Path=dir('.\selfie\*.jpg');
image_1024 = zeros(10,1024);
for i=1:10
    I{1,i}=imread(strcat('.\selfie\',Path(i).name));
    te = I{1,i}';
    image_1024(i,:) = te(:)';
end

%let label as 69
self_label = 69*ones(10,1);
dataset = [image_1024 self_label];

%random operation and generate selfie dataset
r=randperm( size(dataset,1) );   
dataset_set=dataset(r, :);
Self_trainset = dataset(1:7,:);
Self_testset = dataset(8:10,:);
save('Self_trainset.mat','Self_trainset');
save('Self_testset.mat','Self_testset');

