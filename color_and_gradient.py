import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100)):

    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # scale to 8-bit
    scale_factor = np.max(sobel_mag) / 255
    scaled_sobel = (sobel_mag / scale_factor).astype(np.uint8)

    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
    scaled = np.arctan2(abs_sobely, abs_sobelx)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # scaled = np.uint8(255 * grad_dir / np.max(grad_dir))

    # Apply threshold
    dir_binary = np.zeros_like(scaled)
    dir_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return dir_binary

def color_thresh(img, channel=2, color_cvt=cv2.COLOR_RGB2HLS, thresh=(170, 255)):
    if color_cvt == None:
        channel = img[:,:,channel]
    else:
        hls = cv2.cvtColor(img, color_cvt)
        channel = hls[:,:,channel]

    color_binary = np.zeros_like(channel)
    color_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return color_binary

def hsv_color_thresh(img, thresh=(50, 100)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 1]
    return h

    # color_binary = np.zeros_like(h)
    # color_binary[(h > thresh[0]) & (h <= thresh[1])] = 1

    # return color_binary

def color_mask(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # create a mask for white
    lower_white = np.array([20, 0, 180])
    upper_white = np.array([255, 80, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # create a mask for yellow, tricky to find the right yellow boundaries
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([50, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # combine the masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # mask the image
    masked_image = cv2.bitwise_and(hsv, hsv, mask = combined_mask)

    result = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB)
    # return result

    result = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    binary = np.zeros_like(result)
    binary[(result > 0) & (result <= 255)] = 1

    return binary

# result = pipeline(image)

# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()

# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)

# ax2.imshow(result)
# ax2.set_title('Pipeline Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
