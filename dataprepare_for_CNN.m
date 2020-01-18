%For CNN
%randomly transfer images from PIE folders  into CNN_PIE folders
select = [1,11,15,27,21,28,63,39,37,35,23,41,34,19,20,46,55,58,6,50];%Folder number I choose
r=randperm(170);
for i=1:20
    Path=dir(strcat('PIE\',num2str(select(i)),'\*.jpg'));
    train_Path=strcat('.\CNN_PIE\TrainSet\',num2str(select(i)),'\');
    test_Path=strcat('.\CNN_PIE\TestSet\',num2str(select(i)),'\');
    mkdir(train_Path)
    mkdir(test_Path)
    for j=1:119
    filename=strcat('PIE\',num2str(select(i)),'\',Path(r(j)).name);
    copyfile(filename,train_Path);
    end
    for j=120:170
    filename=strcat('PIE\',num2str(select(i)),'\',Path(r(j)).name);
    copyfile(filename,test_Path);
    end   
end

%ADD selfie
Path=dir(strcat('.\selfie\','\*.jpg'));
train_Path=strcat('.\CNN_PIE\TrainSet\',num2str(69),'\');
test_Path=strcat('.\CNN_PIE\TestSet\',num2str(69),'\');
mkdir(train_Path)
mkdir(test_Path)
r=randperm(10)
    for j=1:7
    filename=strcat('.\selfie\','\',Path(r(j)).name);
    copyfile(filename,train_Path);
    end
    for j=8:10
    filename=strcat('.\selfie\','\',Path(r(j)).name);
    copyfile(filename,test_Path);
    end   