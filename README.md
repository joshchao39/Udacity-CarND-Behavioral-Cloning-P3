#**Behavioral Cloning Project** 

[//]: # (Image References)

[image1]: ./writeup_files/flip.png
[image2]: ./writeup_files/distribution.png
[image3]: ./writeup_files/model.png
[image4]: ./writeup_files/loss.png

###Project Structure

My project includes the following files:
* model.py containing the script to create and train the model
* data_processor.py containing the functions for data operations
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* README.md summarizing the results

To train the model
```
python model.py
```

To drive autonomously (along with the simulator)  
```
python drive.py model.h5
```

###Data Generation
About 20 minutes of smooth driving and about 3 minutes of recovery driving were used as the training/validation data.

For recovery recording I let the car go almost off course with zero steering angle before recovering and filter out the samples with zero steering angle. This avoids leaking bad actions (drifting off course) into the model.

In total 16,606 samples were collected, each with three images (center, left, and right cameras) and steering angle.

###Data Pre-processing/Augmentation
####1. Image size reduction
The top 65 and bottom 5 pixels were removed, leaving the image size to be (90, 320, 3). This area should contain all the information the model needs to make decision for track 1.

After cropping, the image is downsized by half along the width and the height, making the effective image size into the model (45, 160, 3).
The image size reduction improves the training time dramatically without losing much accuracy. The same reduction is done in autonomous mode for input consistency. 

####2. Use side cameras
During training the side cameras were used as input with 0.25 steering angle correction. This helps the car to steer toward the center.

Note this should not be used as the sole method for recovery training, as the steering angle not only needs to depend on the position of the car, but also the angle of the car. In large amount it causes the correction to overshoot and the car fish-tails. However with minor amount this method is helpful as recovery driving data is much harder to collect.

####3. Flip images
In this problem domain flipping the image also means flipping the steering angle. This helps the model to generalize.
![alt text][image1]

####4. Data redistribution
The driving samples are naturally unevenly distributed. Since track 1 are mostly long sweeping corners the model would bias toward low steering angle for lower error. Making sure the data being evenly distributed ensures the car is competent in all driving situation. 
This is what the distributions look like before/after the data is balanced (note absolute steering angle is taken here since each image can be flipped to change the sign of the steering angle)
![alt text][image2]

Note1: The spike at 0.5 is from the recovery driving. Emergency steering has large magnitude.

Note2: This also reduced the number of samples from 16,606 to 2,517, further reducing the training time.

###Model Architecture and Training
####1. Model
The model contains 3 convolutional layers with 3x3 kernels and 2x2 max-pool each with 24, 32, 64 features respectively, and 4 fully-connected layers with size 300, 50, 20, and 1.
The model shows very little overfitting but I added some dropouts to let the model learn redundant features hoping for more reliability in autonomous driving.
![alt text][image3]

####2. Training
Adam optimizer with 1e-4 learning rate was used. Since training is fast due to image size reduction I let the model train until validation loss does not reduce for over 50 epochs and pick the parameters with the best validation loss. This also gives me a better idea on whether the model has enough capability to overfit to tune the model size and regularization. 
![alt text][image4]

Note: The final model was trained with a lot of epochs with small learning rate to obtain best accuracy. The training takes less than 10 minutes on GTX750Ti.


###Results
The mode allows the car to drive autonomously on track 1 endlessly (I let it drive overnight) at 25 mph:
<p align="center">
  <img src="writeup_files/track1.gif" alt="Driving autonomously on track 1"/>
</p>

Full video: https://youtu.be/BwD-pfWQ9xA

The car view: https://youtu.be/vGt9wMu_CS8

The car can also drive endlessly at max speed 30 mph but it fish tails a lot.

###Other Notes
####1. Ensure the data is consistent between training and production!
It was found after hours of training models that the input image in autonomous mode was in BGR while the training input image is in RGB. This teaches the lesson that one should make no assumption of the system and always check for production/training data consistency.

####2. Control feedback due to lag
Due to the asynchronous nature of data processing in autonomous mode, the car drives much smoother on my desktop PC than on my laptop. Two things noted:
1. Small delay in model response in production can cause large impact in performance
2. Instability in dynamic control system needs to be taken into consideration when designing the architecture