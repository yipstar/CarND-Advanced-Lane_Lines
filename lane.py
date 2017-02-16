import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.iteration = 0
        self.n_fits = 4
        self.current_binary_type = None

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # polynomial coefficient values of the last n fits of the line
        self.recent_fit = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

    def line_detected(self, fit, fitx, curverad, center_diff, x, y):

        self.detected = True

        # print("fitx length", len(fitx))
        # print("x length", len(x))
        # print("y length", len(y))

        # print("recent_xfitted length", len(self.recent_xfitted))

        # if self.bestx == None:
        #     self.bestx = fitx
        #     self.best_fit = fit
        # else:
        #     self.bestx = self.bestx + (fitx / self.n_fits)
        #     # print("bestx: ", self.bestx)

        #     ploty = np.linspace(0, 719, 720)
        #     self.best_fit = np.polyfit(self.bestx, ploty, 2)

        self.best_fit = fit

        # print("bestx: ", self.bestx)
        # print("best_fit: ", self.best_fit)

        # if self.iteration >= self.n_fits:
        #     self.recent_xfitted.pop()
        #     self.recent_xfitted.append(fitx)

        #     # print("recent_xfitted length", len(self.recent_xfitted))

        #     # self.bestx = np.mean(self.recent_xfitted, axis=0)
        #     # print("bestx: ", self.bestx)

        #     self.recent_fit.pop()
        #     self.recent_fit.append(fit)

        #     print("recent_fit: ", self.recent_fit)

        #     self.best_fit = np.mean(self.recent_fit, axis=0)
        #     print("best_fit: ", self.best_fit)
        # else:
        #     self.recent_xfitted.append(fitx)
        #     self.recent_fit.append(fit)
        #     self.best_fit = fit
        #     self.best_x = x


        self.current_fit = fit
        self.radius_of_curvature = curverad
        self.line_base_pos = center_diff

        # if (len(self.recent_fit) > 0):
        #     self.diffs = self.recent_fit[-1] - fit

        self.allx = x
        self.ally = y

        self.iteration += 1

    def flatten(l):
        return [item for sublist in l for item in sublist]
