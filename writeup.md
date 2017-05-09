
# Project 4 Advanced Lane Finding Project

###The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

###Camera Calibration

####1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The code for this part is:

```
def calc_points(images_name):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_name)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()
    return objpoints,imgpoints
```

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

The code for this part is:

```
def undistort_image(img,objpoints,imgpoints):
    
    #convert image to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #compute the camera calibration matrix and distortion coefficients
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

    #distortion correction
    dst_img = cv2.undistort(img,mtx,dist,None,mtx)
    
    return dst_img
```

The following are some examples of the result(left are the originals and right are the undistorted ones):
 
 <img src="./examples/pic/cal1.jpg" width="300"/> <img src="./examples/pic/out1.jpg" width="300"/> 
 
 <img src="./examples/pic/cal2.jpg" width="300"/> <img src="./examples/pic/out2.jpg" width="300"/>
 
 <img src="./examples/pic/cal3.jpg" width="300"/> <img src="./examples/pic/out3.jpg" width="300"/>
 

###Pipeline (single images)

####2. Use color transforms, gradients, etc., to create a thresholded binary image
The purpose of this part is to generate the binary image and highlight the lane lines. I used a combination of color and gradient thresholds.

The code for this part is:

```
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1  
    
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    return dir_binary


# Gradient threshold combined
def grad_thres(image):

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 150))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined
```

```
#Combined color and gradient thresholding
def color_grad_thres(image, s_thresh=(120, 255), r_thres=(220,255)):
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    #threshold S color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    #threshold R color channel
    r_channel = image[:,:,0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thres[0]) & (r_channel <= r_thres[1])] = 1
    
    #threshold grad,mag,dir channel
    grad_binary = grad_thres(image)
    
    #combine all thresholds together
    #result_binary = np.dstack(( np.zeros_like(grad_binary), \
    grad_binary, s_binary, r_binary))
    result_binary = np.zeros_like(s_binary)
    result_binary[(grad_binary == 1) | (s_binary == 1) | (r_binary == 1)] = 1

    return result_binary
    
```

The following is examples of the result:
 
 <img src="./examples/pic/1-2.png" width="800"/> 
 
Note that detecting the sky and cars is fine since we will crop uncessary part of the image later.

####3. Apply a perspective transform to rectify binary image ("birds-eye view")
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)



####3. Detect lane pixels and fit to find the lane boundary

The code for my perspective transform includes a function called `persp_trans()`. The function takes as inputs an image (`img`), as well as source (`srcp`) and destination (`dstp`) points. The hardcoded source and destination points are:

| Source              | Destination     |
|:-------------------:|:---------------:|
| 300.47,658.238      | 300.47,658.238  |
| 584.888,459.251     | 300.47,0        |
| 697.796,459.251     | 1001.77,0       |
| 1001.77,658.238     | 1001.77,658.238 |


And the code of the function is:

```
def persp_trans(image, srcp, dstp):

    M = cv2.getPerspectiveTransform(srcp,dstp)
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image,M,img_size,flags=cv2.INTER_LINEAR)
    
    return warped, M
```

I verified that my perspective transform was working as expected by drawing the `srcp` and `dstp` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image, which is shown below:

<img src="./examples/pic/1-3.png" width="800"/>


####4. Detect lane pixels and fit to find the lane boundary

The method I use for the line finding is by sliding windows of histogram. After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image where the lane lines stand out clearly. I need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line. In order to solve this, I first take a histogram along all the columns in the lower half of the image. With this histogram I added up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I used that as a starting point for where to search for the lines. From that point, I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

The code of the function is:

```
def sliding_windows(binary_warped, plot = True):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 5

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 


    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    center_offset = abs((left_fitx[-1] + right_fitx[-1])/2 - 640.0)

    if plot:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
```

####5. Determine the curvature of the lane and vehicle position with respect to center

I did this following the previous code, which makes a full function:

```
	y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space

    #left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvated = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvated = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]) 
    center_offset = center_offset * xm_per_pix 
    
    return left_curvated, right_curvated, left_fitx, right_fitx, ploty, center_offset
```

####6. Warp the detected lane boundaries back onto the original image

I implemented with the following code:

```
def plot_img(bird_img, img, persp_M, left_c, right_c, left_fitx, right_fitx, ploty, offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bird_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(persp_M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = 'Left curvature: ' + str(round(left_c,3)) +  'm' + \
            '  Right curvature: ' + str(round(right_c,3)) + 'm' + \
            '  Center offset: ' + str(round(offset,3)) + 'm'
    cv2.putText(result,words,(10,50), font, 0.7,(255,255,255),2)
    
    return result
```

Here is an example of my result on a test image:

<img src="./examples/pic/1-4.jpg" width="500"/>

---

###Pipeline (video)

When processing the video, the sliding window function is modified to take into account of the line quality for each frame and smoothing over the last 4 frames of the video. The detail information is shown under '7.Pipeline(video)' in the jupyter notebook. 

Please refer to file /output/project_video_out.mp4

---

###Discussion

In this project, I find out two things very import for the lane finding. The first thing is without a doubt multiple parameters within each processing step. Given extra time, the parameters should be fine tuned in the future.

Another very import thing for processing the video is how to define the quality of the results(fitted polynomial, lane curvature etc.) for each frame, and how to combine historic results with different quality and the current frame to come up with a more accurate lane curvature. The concepts of filtering, smoothing and sanity check are important points to keep in mind for future work.
