clc;clear all;close all;

% Load Dataset
imgd = imageDatastore('cell_images','IncludeSubfolders',true,'LabelSource','foldernames');

% Count Data Label
labelCount = countEachLabel(imgd);

% Data preprocessing
imageSize = 115;
imgd.ReadFcn = @(filename)Malaria_ImageProcessing(filename,imageSize);


% Data partitioning
[imdsTrain, imdsTest, imdsValid] = splitEachLabel(imgd,0.6,0.2, 0.2, 'randomize');

trainLabelCount = countEachLabel(imdsTrain)
testLabelCount = countEachLabel(imdsTest)
validLabelCount = countEachLabel(imdsValid)

% Design network architecture 
layers = [
    imageInputLayer([115 115 3])
    
    convolution2dLayer(2, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    
    convolution2dLayer(2, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer ];

analyzeNetwork(layers)  

% Training Option
options = trainingOptions('adam', ...
    'MiniBatchSize',24, ...
    'MaxEpochs',10, ...
    'InitialLearnRate', 2e-4, ...
    'ValidationData', imdsValid, ...
    'ValidationFrequency',10, ...
    'ValidationPatience',6, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(imdsTrain, layers, options);

% Test the network
[predicted_label, ~] = classify(net, imdsTest);

% Actual Label
actual_label = imdsTest.Labels;

% Confusion matrix
figure;
plotconfusion(actual_label, predicted_label);
title('Confusion Matrix');


