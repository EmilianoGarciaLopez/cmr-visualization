# Point Cloud Visualization with Camera Views
This application is a utility to visualize point cloud data and display real-time camera feeds side-by-side.

Use the `visualization_suite_points.py` file, not the `visualization_suite.py` file

#### Requirements:
Python (tested with Python 3.11)
Libraries:
- OpenCV
- matplotlib
- numpy
- VTK
- PyQt6

## Code Overview:

### Functions:

`load_and_transform_points`: Loads the point cloud data from a .npy file and rotates it using a rotation matrix.
map_points_to_colors: Maps the loaded points to colors based on their distance from the origin.
create_point_cloud: Creates a VTK point cloud object from the loaded and transformed points.
Classes:

`PointCloudVisualization`: This class sets up the VTK visualization of the point cloud.
CameraView: A QWidget-based class responsible for capturing and displaying live video from a camera using OpenCV.
MainWindow: The main application window that combines camera views and VTK views.
Execution: If the script is run as the main module:

- Points are loaded from pointcloud.npy.
- Point colors are calculated.
- A VTK point cloud is created.
- The main PyQt6 application loop is started, showing the main window with point cloud visualization and camera views.


### How to Run:
1. Make sure you have the necessary libraries installed.
2. Place a pointcloud.npy file in the same directory as the script (or modify the path in the code).
3. Run the script.