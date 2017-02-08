import numpy as np
import cv2

from color_and_gradient import *
from perspective_transform import perspective_transform
from find_lines import *

def pipeline(img, mtx, dist):

    img = np.copy(img)

    # distortion correction
    result = cv2.undistort(img, mtx, dist, None, mtx)

    # color threshold
    min = 90
    color_binary = color_thresh(result, thresh=(min, 255))

    # gradient threshold
    kernel = 15
    grad_x_binary = abs_sobel_thresh(result, orient='x', sobel_kernel=kernel, thresh=(20, 100))

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(grad_x_binary)
    combined_binary[(color_binary == 1) | (grad_x_binary == 1)] = 1

    # perspective transform
    transformed_binary, M_inverse = perspective_transform(combined_binary)

    # find lanes, fit polynomial, and calculate radius of curvature
    left_fit, right_fit, left_curverad, right_curverad, center_diff = find_and_fit_lanes(transformed_binary, visualize=False)
    # print("left_fit: ", left_fit)

    result = draw_lane_on_original_image(img, transformed_binary, left_fit, right_fit, M_inverse)

    # Print radius of curvature on video
    radius = int(left_curverad + right_curverad) / 2
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(radius), (120,80), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)

    print("center_diff ", center_diff)

    cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(center_diff), (100,140), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)

    return result

def draw_lane_on_original_image(img, binary_warped, left_fit, right_fit, M_inverse):

    # create color image to draw the lane on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

    color_unwarped = cv2.warpPerspective(out_img, M_inverse, (binary_warped.shape[1], binary_warped.shape[0]))

    # print("binary_warped shape ", binary_warped.shape)
    # print("color_unwarped shape ", color_unwarped.shape)

    result = cv2.addWeighted(img, 1, color_unwarped, 0.3, 0)
    return result
