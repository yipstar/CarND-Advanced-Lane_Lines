##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Any Jupyter notebook code cells referenced in the following discussion are from the `./Advanced Lane Finding.ipynb` notebook.

The code for this step is contained in cells 3-5.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here the chessboard is 9x6, so the object points will look like (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0). We assume the chessboard is fixed on the (x, y) plane at z=0, which leads to the object points being the same for each calibration image. I then loop through all the calibration images, add on successful chessboard detection, add the (x, y) pixel positions of detected corners to the `impoints` array. Now I can use `objpoints` and `imgpoints` to compute the camera maxtrix and distortion coeffifients using the cv2.calibrateCamera() function. 

The output cell 5 contains an example image and its undistorted version next to it. Cell 6 shows 2 more example images in their undistorted state.

###Pipeline (single images)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I combined two threshold binary images to use for lane detection. One was a color threshold from the s channel of an HLS color space converted image. The other was a Sobel gradient along the x direction. 

The code for my pipeline procedure is in `./pipeline.py`. The file `./color_and_gradient.py` contains helper functions for computing sobel gradients for x, y, direction, and magnitude. It also contains helper functions for computing gradients of HLS color space channels, and a color mask designed to detect yellow and white colors.

Lines 23-36 of `pipeline.py` show how the combined threshold binary image was created. Cell 9 in the referenced notebook show examples of two test images with the sobel x gradient threshold applied. Cell 10 shows these 2 test images with sobel x gradient and perspective transformation applied.

Cell 11-21 show many examples of experiments I conducted searching for the best binary image combination of thresholding. In the end I opted for the simple combination I described above.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Line 43 of `pipeline.py` performs the perspective transform on the combined binary threshold image. It calls a function `perspective_transform` from the `./perspective_transform.py` file which contains the code that does the actual perspective warping. I hardcoded the source and destination points exactly as described in the example writeup template, which I've copied below for reference.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

An example of the perspective_transform function working is in Cell 7, which shows an undistorted test image and its warped perspective version next to it.

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The remaining lines in the `pipeline` function (lines 52-90) contain the high level logic for identifying lane line pixels. This code checks if a previous line detection was successful, and if so it searches for lane line pixels using the `find_from_previous` function from the `./find_lines.py` file. Otherwise if a successful line detection had not occurred in the previous frame (or we're processing the first frame) it searches for line pixels using the `find_new` function. In either case, after trying to find the lane lines, I perform a sanity check on the detected lines (`sanity_check` function) and if they pass the check I store the lines as the current fit and proceed to draw the lines on the original image. If the sanity check fails, then I use the last previously detected successful fit and draw those lines instead.

The `find_new` function uses a histogram approach (lines 5-99) to identify approximately the x locations of the left and right lines on the bottom of the image. Using these locations as a starting point, it then uses the sliding window approach described in the lesson, to detect the pixel positions of the left and right lines respectively. The found line pixels are then used to fit 2nd order polynomials.

The `find_from_previous` function performs a more efficient search (lines 139-165), as it uses the previously found lines as a starting point instead of the histogram approach, and it just searches within a window around these previously found points.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The `./find_lines.py` file also contains the code for calculating the radius of curvature, and the position of the vehicle with respect to center.

Lines 223-260 contain these calculations, this code was essentially copied from the project instructions example code. Both `find_new` and `find_from_previous` call out to the `calculate_curvature_and_center_diff` function to calculate the radius of curvature for the left and right lines, and to calculate the center position.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `draw_current_fit` function (lines 93-128 in `./pipeline.py`) draws the lane onto the original image. It uses the left and right fit polynomials to generate points that form a polygon that maps to the outlined shape of the lane that encompasses the left and right detected lines. It draws this polygon onto a color version of the the binary warped image. It then unwarps this image and layers it on top of the original color image, resulting in the original image frame with the lane drawn on it.

`draw_current_fit` also adds text to the top of image displaying the average radius of curvature of the 2 lines, and the center position of the vehicle with respect to the camera.

Cell 24 contains example images with the full pipeline applied.

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My project video is saved in `./project_video_output.mp4` It can be viewed in Cell 35.

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried many different experiments to find the right binary threshold combination to detect the lines under all the test images provided, as well as against images I exported from the challenge videos. I was unable to find one combination that worked well for all the different lighting and road conditions. I found that using a color mask tuned to detect yellow and white colors was robust in many circumstances but degraded when bright light conditions changed and led to janky detections in problem cases. I tried an implementation that would cycle between these 2 different binary image versions however it did not work well in all conditions on the original project video and due to time constraints I commented out that out of my code. You can see the idea commented out in the `pipeline` function on lines 72-78.

I also spent a bit of time trying to figure out how to smooth the lines over the past 5 frames however I could never get it to work. 

My sanity check function (`sanity_check_fit` lines 130-159) is also not as robust as I would have liked. It currently only checks the lines are a reasonable distance of pixels away from each other, given the 700 pixel estimate provided in the project notes. I tried comparing slopes of the detected lines however I couldn't find a metric that performed well across the challenge videos. I also was unable to figure out how to compare the curvature of the polynomials as suggested in the tips section of the project notes.

For future efforts I would definately like to implement a more robust sanity check function. Ideally I would setup detailed logs of the pipeline run, which would help me more quickly generate test images from the problem frames in the videos and better fine tune the sanity check parameters.

The general approach I took was to get the full pipeline in place first, using the project example code provided. Once a full pipeline was in place I spent a lot of time experimenting with binary threshold image approaches and combinations but my intuitions are not quite there yet with these ideas and I was unable to produce something clever enough to work for all the different cases in the videos provided. For future explorations I would conduct many more experiments in different color spaces and using different combinations. I would also further explore the idea of cycling between different binary image strategies when the sanity check for a frame failed.

In closing, the pipeline is likely to fail in conditions that don't match the project video, but I look forward to spending more time on it in the future and making it much more robust. 