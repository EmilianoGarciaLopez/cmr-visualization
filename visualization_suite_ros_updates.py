import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vtk
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sensor_msgs import Image, PointCloud2
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

IMAGE_LEFT_TOPIC = "/zedsdk_left_color_image"
IMAGE_RIGHT_TOPIC = "/zedsdk_right_color_image"
POINT_CLOUD_TOPIC = "/zedsdk_point_cloud_image"


def load_and_transform_points(file_path):
    """Load and transform the points from the given file."""
    points_array = np.load(file_path)
    points_array = points_array[:, :3]

    rotation_matrix_270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    return np.dot(points_array, rotation_matrix_270)


def map_points_to_colors(points_array):
    """Map points to colors based on their distance from origin."""
    distances = np.linalg.norm(points_array, axis=1)
    normalized_distances = (distances - np.min(distances)) / (
        np.max(distances) - np.min(distances)
    )

    colormap = plt.get_cmap("YlOrBr")
    return (colormap(normalized_distances)[:, :3] * 255).astype(np.uint8)


def create_point_cloud(points_array, colors_array):
    """Create a VTK point cloud object."""
    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    for p, c in zip(points_array, colors_array):
        points.InsertNextPoint(p)
        colors.InsertNextTuple3(c[0], c[1], c[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(colors)
    return polydata


class CameraViewSubscriber(CameraView):
    def __init__(self, topic_name):
        super().__init__(-1)  # We are not using a real camera index here
        self.subscription = self.create_subscription(
            Image, topic_name, self.update_frame_from_topic, BEST_EFFORT_QOS_PROFILE
        )

    def update_frame_from_topic(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.update_frame(frame)


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("point_cloud_visualizer")
        self.latest_polydata = None  # Add this line to store the latest polydata
        self.subscription = self.create_subscription(
            PointCloud2,
            POINT_CLOUD_TOPIC,
            self.point_cloud_callback,
            BEST_EFFORT_QOS_PROFILE,
        )

    def point_cloud_callback(self, msg):
        point_cloud_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC4")
        points_array = np.array(point_cloud_data[:, :3])
        colors_array = map_points_to_colors(points_array)
        self.latest_polydata = create_point_cloud(
            points_array, colors_array
        )  # Update the polydata here


class PointCloudVisualization:
    """A class responsible for setting up the VTK visualization of the point cloud."""

    def __init__(self, polydata):
        self.actor = self._setup_actor(polydata)

    @staticmethod
    def _setup_actor(polydata):
        # Use the vtkVertexGlyphFilter to represent the data points as simple points.
        glyph_filter = vtk.vtkVertexGlyphFilter()
        glyph_filter.SetInputData(polydata)
        glyph_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointData()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Adjust the point size
        actor.GetProperty().SetPointSize(3)
        return actor


class CameraView(QWidget):
    def __init__(self, camera_index):
        super().__init__()
        self.cam = cv2.VideoCapture(camera_index)
        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

        # Set a fixed aspect ratio
        self.aspect_ratio = 16 / 9

    def update_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print(f"Failed to grab frame from camera.")
            return

        aspect_ratio = frame.shape[1] / frame.shape[0]
        qt_img_width = self.label.width()
        qt_img_height = int(qt_img_width / aspect_ratio)

        # If the calculated height exceeds the QLabel height, recalculate
        if qt_img_height > self.label.height():
            qt_img_height = self.label.height()
            qt_img_width = int(qt_img_height * aspect_ratio)

        frame = cv2.resize(
            frame, (qt_img_width, qt_img_height), interpolation=cv2.INTER_NEAREST
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the resized frame

        image = self.convert_to_qimage(frame)
        self.label.setPixmap(QPixmap.fromImage(image))

    def convert_to_qimage(self, frame):
        height, width, _ = frame.shape
        bytesPerLine = 3 * width
        return QImage(
            frame.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        )

    def convert_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        bytesPerLine = 3 * width
        return QImage(
            frame.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        )

    def closeEvent(self, event):
        self.cam.release()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, subscriber: PointCloudSubscriber, parent=None):
        super().__init__(parent)
        self.subscriber = subscriber
        self._setup_ui()
        self.resize(800, 600)
        self.show()

        # Set up a QTimer to refresh the visualization every 50ms
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_vtk_view)
        self.refresh_timer.start(50)

    def _setup_ui(self):
        self.frame = QWidget()
        self.hl = QHBoxLayout()

        # Configure the Renderers
        renderer1 = self._configure_renderer_perspective()
        renderer2 = self._configure_renderer_overhead()

        renderer1.AddActor(visualization.actor)
        renderer2.AddActor(visualization.actor)

        # Left and Right VTK views
        self.vtkWidget1 = self._create_vtk_widget(self.frame, renderer1)
        self.vtkWidget2 = self._create_vtk_widget(self.frame, renderer2)

        # Camera views
        self.cameraView1 = CameraViewSubscriber(IMAGE_LEFT_TOPIC)
        self.cameraView2 = CameraViewSubscriber(IMAGE_RIGHT_TOPIC)

        # Organize the two camera views side-by-side in top_splitter
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.cameraView1)
        top_splitter.addWidget(self.cameraView2)
        top_splitter.setSizes([self.width() // 2, self.width() // 2])
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        # Organize the two VTK views side-by-side in bottom_splitter
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.vtkWidget1)
        bottom_splitter.addWidget(self.vtkWidget2)
        bottom_splitter.setSizes([self.width() // 2, self.width() // 2])
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        # Organize top_splitter and bottom_splitter vertically in main_splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setSizes([self.height() // 2, self.height() // 2])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)

        self.hl.addWidget(main_splitter)

        self.frame.setLayout(self.hl)
        self.setCentralWidget(self.frame)

        # refreshing of the frame
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_vtk_view)
        self.refresh_timer.start(50)  # Refresh every 50ms

    def _create_vtk_widget(self, parent, renderer):
        vtk_widget = QVTKRenderWindowInteractor(parent)
        vtk_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        vtk_widget.GetRenderWindow().Render()
        vtk_widget.Start()
        return vtk_widget

    def refresh_vtk_view(self):
        if not point_cloud_subscriber.latest_polydata:
            return

        # Remove existing actors from the renderers
        self.vtkWidget1.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveAllViewProps()
        self.vtkWidget2.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveAllViewProps()

        # Create and add the new actor
        visualization = PointCloudVisualization(point_cloud_subscriber.latest_polydata)
        self.vtkWidget1.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(
            visualization.actor
        )
        self.vtkWidget2.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(
            visualization.actor
        )

        # Render the updates
        self.vtkWidget1.GetRenderWindow().Render()
        self.vtkWidget2.GetRenderWindow().Render()

    @staticmethod
    def _configure_renderer_perspective():
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
        renderer.GetActiveCamera().SetPosition(0, 0, 200)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        return renderer

    @staticmethod
    def _configure_renderer_overhead():
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
        distance_from_center = 200
        angle = 45  # degrees
        height = distance_from_center * np.tan(np.radians(90 - angle))
        renderer.GetActiveCamera().SetPosition(0, -height, distance_from_center)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 0, 1)
        return renderer


if __name__ == "__main__":
    rclpy.init()
    app = QApplication(sys.argv)

    point_cloud_subscriber = PointCloudSubscriber()  # Initialize point cloud subscriber

    window = MainWindow(point_cloud_subscriber)
    sys.exit(app.exec())

    rclpy.spin(point_cloud_subscriber)
    point_cloud_subscriber.destroy_node()
    rclpy.shutdown()
