# Simple-Face-Recognition  
PCA, LDA, SVM, CNN  
Based on CMU PIE set and the selfie set made by myself with the same resolution as CMU PIE set  
> PCA based data distribution visualization  
>PCA plus nearest neighbor classification results  
>LDA based data distribution visualization  
>LDA plus nearest neighbor classification results  
>SVM classification results with different parameter values  
>CNN classification results with different network architectures  
  
'loadimage.m' can randomly choose 20 out of the subjects from CMU PIE set and use 70% of the provided images for training and use the remaining 30% for testing.   
'img2mat.m' can resize 10 selfies(32*32) into the same resolution as CMU PIE set and save them as .mat file.  
'PCA.m' can finish the PCA for feature extraction, visualization and classification.  
'LDA.m' can finish the LDA for feature extraction.  
'SVM.m' can train svm model to fulfill face recognition.  
'dataprepare_for_CNN.m' can combine the 20 subjects into the 'CNN_PIE' folder for code 'CNN.py'  
'CNN.py' trains a CNN with two convolutional layers and a fully connected layer, with the architecture specified as follows:   
>number of nodes: 20-50-500-21.  
>Convolutional kernel sizes are set as 5.   
>Each convolutional layer is followed by a max pooling layer with a kernel size of 2 and stride of 2.  
>The fully connected layer is followed by ReLU.  
