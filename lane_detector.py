import numpy as np
import cv2

from lane import Line
from color_and_gradient import *
from perspective_transform import perspective_transform
from find_lines import *

class LaneDetector:

    def __init__(self):
        self.left_lane = Line()
        self.right_lane = Line()

    def process_image(self, image):
        image_copy = np.copy(image)

        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        min = 90
        color_binary = color_thresh(result, thresh=(min, 255))

        # gradient threshold
        kernel = 15
        grad_x_binary = abs_sobel_thresh(result, orient='x', sobel_kernel=kernel, thresh=(20, 100))

        # gradient dir threshold
        grad_dir_binary = dir_threshold(result, sobel_kernel=kernel, thresh=(0.7, 1.3))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(grad_x_binary)
        combined_binary[(color_binary == 1) | (grad_x_binary == 1)] = 1

        # perspective transform
        transformed_binary, M_inverse = perspective_transform(combined_binary)

        if self.left_lane.detected:

            fit = find_from_previous(transformed_binary, self.left_lane.current_fit, self.right_lane.current_fit, visualize=False)

            if self.sanity_check_fit(fit):
                self.set_current_fit(fit)
            else:
                fit = self.get_last_detected_fit()
                self.set_undetected_fit()

        else:
            fit = find_new(transformed_binary, visualize=False)

            if self.sanity_check_fit(fit):
                self.set_current_fit(fit)
                # pass
            else:
                fit = self.get_last_detected_fit()
                self.set_undetected_fit()

        result = self.draw_current_fit(fit, image_copy, transformed_binary, M_inverse)

        return result

    def set_current_fit(self, fit, binary_type = None):
        if binary_type != None:
            self.left_lane.current_binary_type = binary_type
            self.right_lane.current_binary_type = binary_type

        self.left_lane.line_detected(fit['left_fit'], fit['left_fitx'], fit['left_curverad'], fit['center_diff'], fit['leftx'], fit['lefty'])

        self.right_lane.line_detected(fit['right_fit'], fit['right_fitx'], fit['right_curverad'], fit['center_diff'], fit['rightx'], fit['righty'])

    def set_undetected_fit(self):
        self.left_lane.detected = False
        self.right_lane.detected = False

    def get_last_detected_fit(self):
        return {'left_fit': self.left_lane.best_fit,
            'right_fit': self.right_lane.best_fit,
            'left_curverad': self.left_lane.radius_of_curvature,
            'right_curverad': self.right_lane.radius_of_curvature,
            'center_diff': self.left_lane.line_base_pos}

    def sanity_check_fit(self, fit):
        if fit == None:
            # print("iteration: ", self.left_lane.iteration)
            # print("no fit found")
            return False

        # Sanity check lines distance from each other
        xm_per_pix = 3.7/700
        line_diff = fit['rightx'][0] - fit['leftx'][0]
        line_diff_m = xm_per_pix * line_diff

        if (line_diff > 720 or line_diff < 540):
            # print("iteration: ", self.left_lane.iteration)
            # print("lines are NOT proper distance from each other, line_diff: ", line_diff)
            return False

        # test slopes
        # left_lane_slope = ((fit['lefty'][-1] - fit['lefty'][0]) / (fit['leftx'][-1] - fit['leftx'][0]))

        # right_lane_slope = ((fit['righty'][-1] - fit['righty'][0]) / (fit['rightx'][-1] - fit['rightx'][0]))

        # slope_diff = left_lane_slope - right_lane_slope

        # if (slope_diff > 5):
        #     print("lines are NOT parallel, slope diff > 5, left_lane_slope, right_lane_slope, diff: ", left_lane_slope, right_lane_slope, slope_diff)
        #     return False

        # TODO: test curvature ???

        return True

    def draw_current_fit(self, fit, img, binary_warped, M_inverse):

        left_fit = fit['left_fit']
        right_fit = fit['right_fit']
        left_curverad = fit['left_curverad']
        right_curverad = fit['right_curverad']
        center_diff = fit['center_diff']

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

        # Print radius of curvature on video
        radius = int(left_curverad + right_curverad) / 2
        cv2.putText(result, 'Radius of Curvature {}(m)'.format(radius), (120,80), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)

        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(center_diff), (100,140), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)

        return result
