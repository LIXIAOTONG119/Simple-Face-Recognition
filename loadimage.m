% obtain all subfolder address in cell path
p = genpath('.\PIE');      
length_p = size(p,2);   
path = {};  
temp = [];
file_list = []; 
for i = 1:length_p % 
    if p(i) ~= ';'
        temp = [temp p(i)];
    else 
        temp = [temp '\']; 
        path = [path ; temp];
        temp = [];
    end
end

% Write a label for 69 folders from 1-69 and choose 20 folders from them
label=1:1:68;
all_label = repmat(label,170,1)';
select = [1,11,15,27,21,28,63,39,37,35,23,41,34,19,20,46,55,58,6,50];
rando = [1,3,7,20,14,21,60,33,31,29,16,36,28,11,13,41,51,54,56,46];
clabel = zeros(length(rando),170);
image = cell(length(rando),170);
Vect = zeros(1,1024);
for i = 1:length(rando)
    ind = rando(i);
    t = strcat(path(ind+1),'*.jpg');
    mm=dir(strcat(t{1,1}));
    for j = 1:length(mm)
        image{i,j} =  imread(strcat(mm(j).folder,strcat('\',mm(j).name)))%read all images
        %transfer 32*32 unit8 to 1024 dimensional vector
        te = image{i,j}';
        image{i,j} = te(:)';
    end
    clabel(i,:) = all_label(select(i),:);
end

% preprocess the data and split into trainingset and testingset
image_1024 = zeros(3400,1024);
trainset=zeros(119*20,1025);
testset=zeros(51*20,1025);
tem = clabel';
labelnew = tem(:);
for i = 1:length(rando)
    for j=1:length(mm)
        image_1024((i-1)*170+j,:) = image{i,j};
        dataset = [image_1024 labelnew];
    end
end
for i = 1:length(rando)
    data_per = dataset((i-1)*170+1:i*170,:);
    r=randperm( size(data_per,1) );   
    data_per_cell=data_per(r, :);
    dataset((i-1)*170+1:i*170,:)=data_per_cell;%random arrange
    %take 70% of every folder(170*1025) as trainset and the rest as testset
    trainset((i-1)*119+1:i*119,:) = dataset((i-1)*170+1:(i-1)*170+119,:);
    testset((i-1)*51+1:i*51,:) = dataset((i-1)*170+120:i*170,:);
end

save('trainset.mat','trainset');
save('testset.mat','testset');

