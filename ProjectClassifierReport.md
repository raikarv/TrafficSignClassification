# **Traffic Sign Recognition Report** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### Data Set Summary & Exploration

#### 1. Data Summary

I used the numpy library to calculate summary statistics of the traffic signs data set from pickle file:
* Training set size 34799
* Validation set size 4410
* Test set size 12630
* Traffic sign image size 32x32x3
* Number of unique classes/labels in the data set 43

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. Six random images were choosen from the training data set and displayed. As it is random plus training and validation data set is shuffled, output images shown here will be different from the ones displayed when the code file is executed. See the results in cell 3

[image1]: ./outputImages/21142.jpg "Narrow lane"
[image2]: ./outputImages/4076.jpg "Speed Limit 50"
[image3]: ./outputImages/20341.jpg "Speed Limit 80"
[image4]: ./outputImages/16470.jpg "Speed Limit 120"


### Design and Test a Model Architecture

#### 1. Preprocess Image

First step, I normalized the image from (0, 255) to (0, 1), then converted normailized image to gray image, reason is convert color channels(3) to 1. The reason being, we are interested in shape of the image for classifing rather than the colors. Converting to grayscale also helps in compute time. Results of the normalization can be seen in cell 5. Again, as the data set shuffled images stored will be different from the images displayed in cell 5

Here few examples of a traffic sign images before and after grayscaling.
[image5]: ./outputImages/Speed80.jpg "Speed Limit 80, before processing"
[image6]: ./outputImages/GraySpeed80.jpg "After Normalized & Grayscale"
[image7]: ./outputImages/childrenCrossong.jpg "Before Processing"
[image8]: ./outputImages/GraychildrenCrossing.jpg "After Normalized & Grayscale"




#### 2. Model Architecture

I basically followed 5-layer LeNet Lab exercise executed in the class, with some changes. My initial model did not produce the satisfactory validation accuracy but with minor changes I was able to get reasonable 96% accuracy saw an immediate improvement.
My final model consisted of the following layers:

#Layer 1: Convolutional 
    Input: Size 32x32x1
    Stride: 1,1,1 
    Padding: Valid
    Output: 28x28x12
  
  Followed by
  Relu Activation
  Pooling
      Stride: 1, 2, 2
      Output: 14x14x12
   
#Layer 2 Convolutional
    Input: Size 14x14x12
    Stride: 1,1,1 
    Padding: Valid
    Output: 10x10x25
  
  Followed by
  Relu Activation
  Pooling
      Stride: 1, 2, 2
      Output: 5x5x25    
    
#Flatten: Flatten output shape of the final pooling layer such that it's 1D instead of 3D i.e. 5x5x25=625

#Layer 3: Fully Connected
    Input: Size 625
    Output: Size 300
    #Relu Activation

#Layer 4: Fully Connected
    Input: Size 300
    Output: Size 180
    #Relu Activation

#Layer 5: Fully Connected (Logits)
    Input: Size 180
    Output: Size 43, Total classifiers
    
Above LeNet is defined in cell 6


#### 3. Training LeNet Model

I did not change hyper parameters used in the normal distribution of weights and bias, mean(mu=0) and standard deviation(sigma=0.1). I used cross entropy (softmax) and its mean to calculate error. Finally, Adam Optimizer was used to minimize error. I did play with EPOCHS, learning rate and batch size which basically gives better accuracy at the expense of compute time.

Training, Validation & Tes Model Pipeline can be seen in 7,8,9 & 10


#### 4. Approach to get validation set accuracy to be at least 0.93
Iterative process was used, varied learning rate (reduced learning rate to see changing weights and bias slowly), EPOCHS and batch size (reduce sample size in CNN for accuracy). Below is the log of the iterative process. 

Learning Rate	EPOCH	Batch Size	Validation Accuracy 	Test Accuracy
0.0008      	10  	128     	0.899               	0.905
0.0008      	15  	128     	0.911               	0.919
0.001       	10   	128     	0.919               	0.901
0.001       	15   	128     	0.928               	0.911
0.0008      	10   	64      	0.924               	0.903
0.0008      	15   	64      	0.932               	0.9259
0.001       	10   	64      	0.938               	0.917
0.001       	15  	64      	0.948               	0.92
0.0008      	10  	32      	0.931               	0.931
0.0008      	15   	32      	0.94                	0.933
0.001       	10  	32      	0.943               	0.919
0.001       	15  	32      	0.951               	0.938

Summary: Lowering learning rate, increasing EPOCHS and reducing batch size led me to direction of increased accuracy. However, they come at expense of time. For a large data set this is not efficient. So I chose learning rate 0.001, EPOCHS 10 & Batch Size 32 on the current LeNet to give validation accuracy > 0.93. However, there were other methods like dropout which might have improved overall accuracy, in the interest of time did not have the opportunity to experiment.

My final model results were:
* Validation set accuracy 0.935 
* Test set accuracy 0.92

Results can be seen in cell 12 & 13.
 

### Test a Model on New Images

#### 1. New Six German Traffic Signs
Here are five German traffic signs that I found on the web. Images were assigned numbers to file name such that sign matches with respective classifier(43), then images were reduce to 32x32x3. This is represented in cell 14 and output

[image9]: ./GermanSignImages/12.prorityLane.png
[image10]: ./GermanSignImages/24.roadNarrow.png
[image11]: ./GermanSignImages/26.trafficSign.png
[image12]: ./GermanSignImages/28.childrenCrossing.png
[image13]: ./GermanSignImages/29.bicycleCrossing.png
[image14]: ./GermanSignImages/34.turnLeft.png

Priority sign image, the shape resembles a traingle in top half. This can be an issue compared to other signs. Road Narrow sign image, I see the two line are thin and very close, this might to different classifier like traffic signals. Traffic sign image, same as previous case. Children crossing and bicycle crossing image, if the images are noisy this classifer might not work, as seen in my results below. Turn left sign, I can't think of any issue with this image.


#### 2. Summary of results on new test images
Before the image was tested, new images were normalized and converted to gray scale.

Here are the results of the prediction:

Orignal Images                  	Prediction
Road narrows on the right       	Road narrows on the right
Turn left ahead                 	Turn left ahead
Priority road                    	Priority road
Bicycle crossing                	Speed Limit (60km/h)
Traffic Signals                 	Traffic Signals
Children Crossing               	Children Crossing

The model was able to correctly guess 5 out of the 6 traffic signs, which gives an accuracy of 83.3%. Bicycle crossing sign could not be classifed correctly. Cell 17 gives the results.

#### 3. Top 5 Softmax Probabilities

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is confident that this is a "Road narrows on the right" sign (probability of 0.9999), and the image does contain same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99       			| Road narrows on the right     				| 
| .00    				| Pedistrain 									|
| .00					| Dangerous curve on left						|
| .00	      			| Traffic Signals       		 				|
| .00				    | Slippery Road      							|


For the second image, the model is confident that this is a "Turn Left Ahead" sign (probability of 0.9999), and the image does contain same sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99       			| Turn Left Ahead               				| 
| .00005   				| General Caution								|
| .0000017				| Yeild                 						|
| .00	      			| Round about Mandatory    		 				|
| .00				    | Traffic Signals      							|

For the third image, the model is confident that this is a "Priority Road" sign (probability of 0.9999), and the image does contain same sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99       			| Priority Road                 				| 
| .00011   				| Ahead Only    								|
| .00004				| End of all speed and passing limits			|
| .000037      			| Keep right            		 				|
| .00				    | Turn left ahead      							|

For the fourth image, the model is relatively sure that this is a "BicyCle crossing" sign (probability of 0.915), and the image does not contain same sign instead it is "Speed limit (60km/h)". Third highest proabliity matches the sign but probability(0.039) does NOT give any confidence that it is Bicycle crossing. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .915       			| Speed limit (60km/h)            				| 
| .045   				| Children crossing								|
| .039  				| Bicycles crossing                 			|
| .00023      			| Go straight or right     		 				|
| .00				    | Right-of-way at the next intersection			|

For the fifth image, the model is confident that this is a "Traffic signals" sign (probability of 0.995), and the image does contain same sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9945       			| Traffic signals                 				| 
| .0055   				| General caution  								|
| .0000 				| Go straight or left                			|
| .0000      			| Speed limit (70km/h)     		 				|
| .0000				    | Road narrows on the right						|

For the sxth image, the model is extremely confident that this is a "Traffic signals" sign (probability of 1.0), and the image does contain same sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Children crossings              				| 
| .0000   				| Right-of-way at the next intersection			|
| .0000 				| End of all speed and passing limits  			|
| .0000      			| Speed limit (60km/h)     		 				|
| .0000				    | Speed limit (20km/h)  						|
