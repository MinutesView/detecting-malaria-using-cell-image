# detecting-malaria-using-cell-image

Introduction:

Malaria is an infectious disorder caused by Plasmodium group individual Protozoan Parasite. The sickness is spread in particularly via the bite of Anopheles Mosquito, an infected female. With almost 240 million cases reported each year, the disease puts nearly forty percent of the world population at threat. Typical symptoms of malaria comprise fever, nausea, headaches and, in severe instances, yellow skin, seizures, coma that leads to death. Every year trained professionals examine several million blood films to detect malaria infection. Malaria detection involves manually numbering the parasites and infected red blood cells. However, this relies totally upon the microscopist's experience and expertise. It has been proven in several area studies that manual microscopy isn’t always a reliable screening approach when accomplished through non-experts due to lack of training particularly in the rural regions where malaria is endemic.Operating in a limited resource set-up with no supportive program for capacity maintenance can affect the quality of the diagnosis. That leads to erroneous diagnostic decisions. Henceforth we have built up this method to give a precise diagnosis. Convolutional Neural Networks (CNN), a category obtained from deep learning (DL) models is used to obtain superior end results with feature extraction and categorization. CNN based DL models are a characteristic extractors towards classifying the blood cells. CNN is a promising device for feature extraction. Automated malaria screening using DL techniques, consequently, function an effective diagnostic aid. An automated Artificial Intelligence (AI) system targets at performing this task without human intervention and to offer a goal, reliable, and efficient tool to accomplish that.All this makes deep learning models incredibly a precise option for performing computer vision tasks.

Dataset Description:

The dataset is organized into a folder (cell_images) and contains subfolders for each image category (Parasitized, Uninfected).  There are 27,5560 images of malaria cell image of  blood  and  2 categories of infected and uninfected images. The  two  levels of cell images are,
i)	Parasitized, which shows the amount of infected cell image of blood.
ii)	Uninfected, which shows the amount of uninfected cell image of blood. 

The numbers of different category pictures for train are Parasitized 8,267  images and Uninfected 8,267 images.
The numbers of different category pictures for test are Parasitized 2,756  images and Uninfected 2,756 images.


Specifying Training and Validation Sets
We divided the imds into three datastores as imdsTrain, imdsTest, imdsValid. We divided the imds into three datastores randomly with 60 % imdsTrain, 20% imdsTest and 20% imdsValid images.



Preprocessing:

To help the AI with detecting the malaria parasite we will resize all the images to same size. For this we used a function named image preprocessing in which we resized all the pictures to 115  by 115 size. We used the function to resize all the images. 
Network Architecture
Image Input Layer: An imageInputLayer is where you specify the image size, which, in this case, is 115 by 115 by 3. These numbers correspond to the height, width, and the channel size. The digit data consists of grayscale images, so the channel size (color channel) is 3. For a color image, the channel size is 3, corresponding to the RGB values. You do not need to shuffle the data because trainNetwork, by default, shuffles the data at the beginning of training. trainNetwork can also automatically shuffle the data at the beginning of every epoch during training.
Convolutional Layer: In the convolutional layer, the first argument is filterSize, which is the height and width of the filters the training function uses while scanning along the images. For example, in the first convolutional layer, the number 2 indicates that the filter size is 2-by-2. You can specify different sizes for the height and width of the filter. The second argument 16 is the number of filters, numFilters, which is the number of neurons that connect to the same region of the input. This parameter determines the number of feature maps. Use the 'Padding' name-value pair to add padding to the input feature map. For a convolutional layer with a default stride of 1, 'same' padding ensures that the spatial output size is the same as the input size. You can also define the stride and learning rates for this layer using name-value pair arguments of convolution2dLayer. We added 4 convolutional layers in the code of varying sizes and filter. 
Batch Normalization Layer: Batch normalization layers normalize the activations and gradients propagating through a network, making network training an easier optimization problem. Use batch normalization layers between convolutional layers and nonlinearities, such as ReLU layers, to speed up network training and reduce the sensitivity to network initialization. Use batchNormalizationLayer to create a batch normalization layer. We used 4 batch normalization layers after each convolutional layer.
ReLU Layer: The batch normalization layer is followed by a nonlinear activation function. The most common activation function is the rectified linear unit (ReLU). Use reluLayer to create a ReLU layer. We used 4 batch ReLU layers after each convolutional layer.
Max Pooling Layer: Convolutional layers (with activation functions) are sometimes followed by a down-sampling operation that reduces the spatial size of the feature map and removes redundant spatial information. Down-sampling makes it possible to increase the number of filters in deeper convolutional layers without increasing the required amount of computation per layer. One way of down-sampling is using a max pooling, which you create using maxPooling2dLayer. The max pooling layer returns the maximum values of rectangular regions of inputs, specified by the first argument, poolSize. In this example, the size of the rectangular region is [2,2]. The 'Stride' name-value pair argument specifies the step size that the training function takes as it scans along the input. We used 4 batch Max pooling layers of varying size and strides after each convolutional layer.
Fully Connected Layer: The convolutional and down-sampling layers are followed by one or more fully connected layers. As its name suggests, a fully connected layer is a layer in which the neurons connect to all the neurons in the preceding layer. This layer combines all the features learned by the previous layers across the image to identify the larger patterns. The last fully connected layer combines the features to classify the images. Therefore, the OutputSize parameter in the last fully connected layer is equal to the number of classes in the target data. In this example, the output size is 2, corresponding to the 2 classes. We used fullyConnectedLayer to create a fully connected layer.
Softmax Layer: The softmax activation function normalizes the output of the fully connected layer. The output of the softmax layer consists of positive numbers that sum to one, which can then be used as classification probabilities by the classification layer. Create a softmax layer using the softmaxLayer function after the last fully connected layer.
Classification Layer: The final layer is the classification layer. This layer uses the probabilities returned by the softmax activation function for each input to assign the input to one of the mutually exclusive classes and compute the loss. To create a classification layer, we used classificationLayer.

![Layer](https://user-images.githubusercontent.com/48564403/158805824-84f19053-9a64-457e-8189-c52485937db6.png)

Here we can see the total number of layers which is 24 here and how they are connected. 


Specify Training Options

After defining the network structure, specify the training options. We trained the network with optimizer adam with an initial learning rate of 0.0002. We used MiniBatchSize of  24 which means the AI will work with batched of 24 files. We set the maximum number of epochs to 10. An epoch is a full training cycle on the entire training data set. We monitored the network accuracy during training by specifying validation data which is imdsValid here and validation frequency which is 10 here. Shuffle the data every epoch. The software trains the network on the training data and calculates the accuracy on the validation data at regular intervals during training. The validation data is not used to update the network weights. We turned on the training progress plot. 

![Training](https://user-images.githubusercontent.com/48564403/158805717-8fa44b4a-c13b-4c99-aaa9-74621a3fa45f.png)

Training Progress:

We can see that the training progress took 250 min 4 sec with 86.85% validation accuracy. Training cycle had 1 epoch and 290 out of 6880 iterations were completed. Per epoch there were 688 iterations. Validation frequency was 10 iterations which mean the graph was plotted after 10 iterations with a patience of 6. Here we can also see the learning rate was .0003.

Confusion Matrix: 

At the end of the training progress, the confusion matrix will be used to show the result of the test done by the AI. A combination of predicted class and actual class result is shown in confusion matrix. In the confusion matrix given below, we can see that the AI predicted correctly 82.3% of actual parasitized pictures, 92.2% of uninfected pictures. This result is shown vertically. On the other hand, if we look at the confusion matrix horizontally we can see that, 91.3% pictures were identified accurately parasitized and 83.9% were identified accurately as uninfected. In the bottom right corner we can see the average accuracy which is 87.2%. This confusion matrix helps us to understand how good the AI is and if we need to improve it even more. AI can be called good if it has accuracy higher than 95%. So, we need to improve the AI a little bit more to get higher accuracy and make it useful for medical purposes. 

![Confusion](https://user-images.githubusercontent.com/48564403/158805930-17846de6-6cd7-49eb-9417-3b2d37a24b88.png)

Conclusion:

The main aim of the proposed research work is to develop an image assisted examination procedure to extract and assess malaria affected cells. If more accuracy can be achieved then we will be able to detect the malaria damaged cells in patients and take measures accordingly. Malaria parasite is big problem in today’s world. If we can develop the AI further , we will be able to help the doctor working with this problem.

