"""
    file: cvvis.py
    brief: file contains the implementation of a visualizer class for 
           visualizing the full-stack status of the car at real time and
           should work over SSH
           Implementation currently supports viewing multiple view-points,
           displaying predicted cone positions, and viewing mid-line splines

    Usage:
    >>> from fsdv.perceptions.lidar.cvvis import cvvis
    >>> cvvis = CVVIs()
    >>> while True:
    >>>     cvvis.update(cones, spline_points)
    >>> cvvis.close() # once done using visualizer
"""

import time

import cv2
import numpy as np
import open3d as o3d

# conversion from meters to pixels
PIXELS_PER_M = 45

# image space
DIMS = np.array([990, 540])
ORIGIN = np.array([720, 270])

# sizes of drawable objects (in meters)
AXIS_LONG_M = 1
AXIS_SHORT_M = 0.1
CONE_LENGTH_M = 0.1
SPLINE_LENGTH_M = 0.075

# sizes of drawable objects (in pixels)
AXIS_LONG_PIXELS = int(AXIS_LONG_M * PIXELS_PER_M)
AXIS_SHORT_PIXELS = int(AXIS_SHORT_M * PIXELS_PER_M)
CONE_LENGTH_PIXELS = int(CONE_LENGTH_M * PIXELS_PER_M)
SPLINE_LENGTH_PIXELS = int(SPLINE_LENGTH_M * PIXELS_PER_M)

ORANGE = np.array([32, 131, 250])


class CVVis:
    def __init__(self, name="cvvis"):
        self.name = name
        self.image = np.ones((DIMS[0], DIMS[1], 3))

    def start(self):
        cv2.imshow(self.name, self.image)
        cv2.waitKey(1)

    def _draw_grid(self):
        """draw meter lines in an image to make a frame of reference"""
        first_horiz_bars = ORIGIN[0] - np.arange(0, ORIGIN[0], PIXELS_PER_M)
        second_horiz_bars = np.arange(ORIGIN[0], DIMS[0], PIXELS_PER_M)
        first_vert_bars = ORIGIN[1] - np.arange(0, ORIGIN[1], PIXELS_PER_M)
        second_vert_bars = np.arange(ORIGIN[1], DIMS[1], PIXELS_PER_M)

        # get indices where bars are supposed to happen
        horiz_bars = np.concatenate([first_horiz_bars, second_horiz_bars])
        vert_bars = np.concatenate([first_vert_bars, second_vert_bars])

        # draw the horizontal meter-lines with a black bar
        self.image[horiz_bars, :, :] = 0
        self.image[:, vert_bars, :] = 0

        return

    def _draw_axes(self):
        """draw an axis on the image from the origin"""
        # draw the x-axis (red)
        rs = ORIGIN[0] - AXIS_SHORT_PIXELS // 2
        re = ORIGIN[0] + AXIS_SHORT_PIXELS // 2
        cs = ORIGIN[1]
        ce = ORIGIN[1] + AXIS_LONG_PIXELS
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [255, 0, 0]

        # draw the y-axis (blue)
        rs = ORIGIN[0] - AXIS_LONG_PIXELS
        re = ORIGIN[0]
        cs = ORIGIN[1] - AXIS_SHORT_PIXELS // 2
        ce = ORIGIN[1] + AXIS_SHORT_PIXELS // 2
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [0, 255, 0]

        # draw the center of the axis (black)
        rs = ORIGIN[0] - AXIS_SHORT_PIXELS // 2
        re = ORIGIN[0] + AXIS_SHORT_PIXELS // 2
        cs = ORIGIN[1] - AXIS_SHORT_PIXELS // 2
        ce = ORIGIN[1] + AXIS_SHORT_PIXELS // 2
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [0, 0, 0]

        return

    def _setup_image(self):
        self.image = np.ones((DIMS[0], DIMS[1], 3)) * 255
        self._draw_grid()
        self._draw_axes()
        return

    def _points_to_pixels(self, points):
        """converts points in x and y dimensions to a central position in the
        image where the pixels should be

        NOTE: function will remove points that are not in the image
        """
        points = np.rint(points * PIXELS_PER_M)
        points[:, 1] *= -1
        pixel_deltas = points[:, [1, 0]]
        pixels = pixel_deltas + ORIGIN
        pixels = pixels.astype(np.int64)

        in_height = np.logical_and(pixels[:, 0] >= 0, pixels[:, 0] <= DIMS[0])
        in_width = np.logical_and(pixels[:, 1] >= 0, pixels[:, 1] <= DIMS[1])
        in_image = np.logical_and(in_height, in_width)

        pixels = pixels[in_image]
        return pixels

    def _draw_squares(self, centers, colors=None, length=20):
        """draws squares in the image for each point"""
        for i in range(centers.shape[0]):
            r, c = centers[i, :]
            rs, re = r - length, r + length
            cs, ce = c - length, c + length

            c = colors[i, :] if colors is not None else ORANGE
            self.image[rs:re, cs:ce, :] = c

        return

    def update(self, cones, spline):
        """points are (x,y) positions and are given in meters

        cones is a np.ndarray of shape (N,3) representing (x,y,c) where c is color
        spline is an np.ndarray of shape (N,2) representing points on the spline
        """
        # refresh the page
        self._setup_image()

        if cones.shape[0] == 0:
            cones = np.zeros((0, 3))
        if spline.shape[0] == 0:
            spline = np.zeros((0, 3))
            print("CVVis Warning: given empty spline points")

        # construct the colors for
        cone_colors = np.zeros((cones.shape[0], 3))
        cone_colors[cones[:, 2] == 0, :] = [237, 61, 7]
        cone_colors[cones[:, 2] == 1, :] = [92, 209, 255]

        spline_colors = np.zeros((spline.shape[0], 3))
        spline_colors[:, :] = [0, 0, 255]

        # draw the points
        cone_pixels = self._points_to_pixels(cones)
        spline_pixels = self._points_to_pixels(spline)
        self._draw_squares(cone_pixels, cone_colors, length=CONE_LENGTH_PIXELS)
        self._draw_squares(spline_pixels, spline_colors, length=SPLINE_LENGTH_PIXELS)

        # display the image
        cv2.imshow(self.name, self.image.astype(np.uint8))
        cv2.waitKey(1)
        pass

    def close(self):
        cv2.destroyWindow(self.name)
        pass


def main():
    v = CVVis()
    v.start()

    while True:
        cones = np.array([[-1, 0, 0], [1, 0, 1], [-0.95, 0.5, 0], [0.98, 0.45, 1]])
        spline = np.array(
            [[0, 0.1], [-0.01, 0.2], [0.03, 0.3], [0, 0.4], [-0.01, 0.5], [0.03, 0.6]]
        )

        spline = np.random.randn(10, 2)

        v.update(cones, spline)
        # time.sleep(1)
        pass


if __name__ == "__main__":
    main()
