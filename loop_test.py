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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


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


class PointCloudVisualization:
    """A class responsible for setting up the VTK visualization of the point cloud."""

    def __init__(self, polydata):
        self.actor = self._setup_actor(polydata)
        self.transform = vtk.vtkTransform()

    @staticmethod
    def _setup_actor(polydata):
        # Use the vtkVertexGlyphFilter to represent the data points as simple points.
        glyph_filter = vtk.vtkVertexGlyphFilter()
        glyph_filter.SetInputDataz(polydata)
        glyph_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointData()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Adjust the point size
        actor.GetProperty().SetPointSize(3)
        return actor

    def shift_point_cloud(self, x_shift=1.0, y_shift=1.0, z_shift=1.0):
        """Shift the point cloud by given amounts in x, y, and z directions."""
        self.transform.Translate(x_shift, y_shift, z_shift)
        self.actor.SetUserTransform(self.transform)


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.resize(800, 600)
        self.show()

        # Store references to the VTK render windows
        self.render_windows = [
            self.vtkWidget1.GetRenderWindow(),
            self.vtkWidget2.GetRenderWindow(),
        ]

        # Set up a QTimer to update the point cloud's position
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_point_cloud)
        self.update_timer.start(100)  # Update every 100ms

    def update_point_cloud(self):
        """Update the position of the point cloud."""
        visualization.shift_point_cloud()

        # Refresh the VTK render windows
        for render_window in self.render_windows:
            render_window.Render()

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
        self.cameraView1 = CameraView(1)
        self.cameraView2 = CameraView(1)

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

    def _create_vtk_widget(self, parent, renderer):
        vtk_widget = QVTKRenderWindowInteractor(parent)
        vtk_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        vtk_widget.GetRenderWindow().Render()
        vtk_widget.Start()
        return vtk_widget

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
    points_array = load_and_transform_points("pointcloud.npy")
    colors_array = map_points_to_colors(points_array)
    polydata = create_point_cloud(points_array, colors_array)

    visualization = PointCloudVisualization(polydata)

    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
