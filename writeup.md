**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/1.png
[image5]: ./output_images/2.png
[image6]: ./output_images/3.png
[image7]: ./output_images/frame_0.png
[image8]: ./output_images/frame_1.png
[image9]: ./output_images/frame_2.png
[image10]: ./output_images/frame_3.png
[image11]: ./output_images/frame_4.png
[image12]: ./output_images/frame_5.png
[image13]: ./output_images/labels.png
[image14]: ./output_images/final_bb.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained `train.py` (lines 13 to 50) with most function definitions placed in `functions.py`. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and through trial and error, decided on `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. These value provided the best performance in training and testing. Additionally, I also utilized both bin spatial and color histogram features (`functions.py` lines 77 through 84). It's important to note that I normalized the feature vectors for both car and not-car classes:
```
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I split my data set using:
```
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
This ensured that the data was sufficiently randomized and that 20% of the data set was set aside as a test set (`train.py` lines 54-55). Following that, I trained and tested the SVM using:
```
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```
On average, my testing set was correctly classified in the range of 99% accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
For this application, it seemed prudent to perform a sliding window search in specific areas of interest. Furthermore, in the lower section of the camera view, the cars are closer to the camera and appear larger. As the we approach the vanishing point of the image the cars will appear smaller in the camera view due to the perspective geometry of the scene. In order to take advantage of this, I set up two ranges of sliding window search, one large and one smaller scale. The larger scale window search spans the lower section of the image, with thew smaller scale spanning a higher portion. After several rounds of testing, I concluded that having the search areas overlap provided the most robust search method. The image below shows the bounding boxes for each window in the sliding window search. The multiscale window search is set up in my `process_frame` function in `funtions.py` (lines 220 - 223)  

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images and their corresponding heat maps (Note: the second image has NO cars, and NO false detections as appropriate). In order to optimize the pipeline, I utilized the HOG sub-sampling strategy. This allowed the HOG feature map to be coimputed only once per frame and then subsampled accordingly during the window search. This greatly reduced the processing time per frame. (`functions.py` lines 104-180)

![alt text][image4]
![alt text][image5]
![alt text][image6]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=4Gf01Zqji_U&feature=youtu.be)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Additionally, I maintained a heat_map history in order to smooth the detection and preserve detection during bad frames.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
### Here are six frames and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image13]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image14]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For this project, I used many of the strategies outlined in the lecture modules. Overall, I found the SVM approach to be very dependant on many factors with significant difficulty in achieving both good performance and robust/correct detection. HOG subsampling helped decrease the computational load per frame, but my laptop still struggled to process the frames anywhere close to 'real-time'. Additionally, much of the process was trial and error. Trying various color-spaces, HOG parameters, bin spacings, window search ranges, and heat map thresholding was a very laborious process. After much trial and error, I fear that although I was able to find parameters that performed well on the project video, testing the pipeline on other scenes may yield unsatisfactory misses or false positives.

In future work on this project, I would likely take a R-CNN approach like YOLO to perform the vehicle detection. I believe this approach would perform well, especially if we were to take advantage of embedded GPU hardware like the NVIDIA TX2. Additionally, pre-trained models would likely provide good performance, either out-of-the-box or by allowing us to take advantage of transfer learning. Additionally, I feel that this approach may yield a more generalized pipeline that will perform well in many conditions.
