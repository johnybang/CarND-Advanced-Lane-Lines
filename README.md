## Advanced Lane Finding
John Bang - 4/21/17

### As a part of the Udacity Self Driving Car Engineer Nanodegree program, we use computer vision to again implement lane finding, but using a more advanced set of techniques than the first project of the course.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/calibration1_undistortion.jpg "calibration1.jpg undistortion"
[image2]: ./examples/test1_undistortion.jpg "test1.jpg undistortion"
[image3]: ./examples/test5_binary.jpg "Binary Example"
[image4]: ./examples/straight_lines_warped.jpg "Warp Example"
[image5]: ./examples/test2_polyfit.jpg "Fit Visual"
[image6]: ./examples/test3_curverad.jpg "Curvature Calc"
[image7]: ./examples/test6_regionoverlay.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 0. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in [code cells 1 and 2](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#camera_calibration) of the [Jupyter notebook](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In [code cell 3](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#apply_distortion) I undistorted some test images. Here's how distortion correction (as described above) looks on a test image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at [code cells 4 and 5](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#create_binary) of the jupyter notebook).  Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in [code cells 6-8](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#perspective) of the jupyter notebook.  In cell 6, I hard code source (`srcpoints`) and destination (`dstpoints`) points.  I chose to hardcode the source and destination points in the following manner:

```python
# src points rectangle vertices
srcpoints = np.float32(
    [[(imgsize[0] / 2) - 60, imgsize[1] / 2 + 100],
    [((imgsize[0] / 6) - 10), imgsize[1]],
    [(imgsize[0] * 5 / 6) + 40, imgsize[1]],
    [(imgsize[0] / 2 + 64), imgsize[1] / 2 + 100]])

# dst points rectangle vertices
dstpoints = np.float32(
    [[(imgsize[0] / 4), 0],
    [(imgsize[0] / 4), imgsize[1]],
    [(imgsize[0] * 3 / 4), imgsize[1]],
    [(imgsize[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| srcpoints                       | dstpoints     |
|:--------------------------------|:--------------|
| [ 580.  460.]                   | [ 320.    0.] |
| [ 203.33332825  720.        ]   | [ 320.  720.] |
| [ 1106.66662598   720.        ] | [ 960.  720.] |
| [ 704.  460.]                   | [ 960.    0.] |

Further in cell 6, I computed the perspective transform matrix (`M`) to be used for perspective warping.

In cell 7, I verified the validity of my perspective transform by warping the undistorted straight road images, then drawing the `srcpoints` and `dstpoints` polygons onto them to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In [code cells 9 and 10](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#fit_lane), I identify lane line pixels for the left and right and fit their positions with a 2nd order polynomial.  I implemented three approaches, two of which are actually used in the pipeline and one which was simply considered and plotted:

1. A sliding window using a convolution which maximizes pixel count within some margin of the previous window position (not used in pipeline) 
2. A sliding window which recenters based on pixel mean of the previous window if enough pixels are present in the previous window (used in pipeline when no valid previous fit exists) 
3. A fixed margin surrounding a previous polynomial fit (used in the pipeline when a valid previous fit exists)

When tuned for the test images, both methods 1 and 2 performed similarly.  I opted for 2. because it seemed a little less complex, but I'm open to using 1. if 2. starts having trouble at some point in the future.  Here's a visualization of the three methods on one test image ("hot" pixels fall inside the green boundaries and the yellow lines are the 2nd order polynomial fit on the "hot" pixels):

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In [code cells 11 and 12](https://github.com/johnybang/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb#curvature) I computed the radius of curvature in meters by converting the lane pixels index positions in to meter units, fitting a 2nd degree polynomial curve to these points yielding the polynomial in meters, and finally computing the radius of curvature according to the formula [shown here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). Within the `compute_curvature()` function, I hardcoded the first and second derivatives assuming a 2nd degree polynomial function.

![alt text][image6]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
