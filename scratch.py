import vtk
from PyQt6.QtWidgets import QApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def create_basic_point_cloud():
    """Create a VTK point cloud object with just a few handpicked points."""
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(10, 0, 0)
    points.InsertNextPoint(0, 10, 0)
    points.InsertNextPoint(0, 0, 10)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    return polydata


def create_basic_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(10)
    actor.GetProperty().SetColor(1, 0, 0)  # Red color
    return actor


if __name__ == "__main__":
    app = QApplication([])
    window = QVTKRenderWindowInteractor()
    window.GetRenderWindow().SetSize(800, 600)

    renderer = vtk.vtkRenderer()
    window.GetRenderWindow().AddRenderer(renderer)

    polydata = create_basic_point_cloud()
    actor = create_basic_actor(polydata)

    renderer.AddActor(actor)
    renderer.ResetCamera()
    renderer.SetBackground(0.5, 0.5, 0.5)  # Setting a gray background

    window.Initialize()
    window.Start()
    app.exec()
import matplotlib.pyplot as plt
import numpy as np
import vtk

# Load your points
points_array = np.load("pointcloud.npy")
points_array = points_array[:, :3]

# Calculate distances from the origin and map to colors
distances = np.linalg.norm(points_array, axis=1)
normalized_distances = (distances - np.min(distances)) / (
    np.max(distances) - np.min(distances)
)

colormap = plt.get_cmap(
    "viridis"
)  # You can change to any available colormap, e.g., "jet", "plasma", etc.
colors_array = (colormap(normalized_distances)[:, :3] * 255).astype(np.uint8)

# Create a point cloud object
points = vtk.vtkPoints()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

for p, c in zip(points_array, colors_array):
    points.InsertNextPoint(p)
    colors.InsertNextTuple3(c[0], c[1], c[2])

# Create a polydata object
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.GetPointData().SetScalars(colors)

# Create a glyph filter to visualize each point
glyphFilter = vtk.vtkVertexGlyphFilter()
glyphFilter.SetInputData(polydata)
glyphFilter.Update()

# Create a visualization mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyphFilter.GetOutputPort())
mapper.SetScalarModeToUsePointData()

# Create an actor for visualization
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window, renderers, and interactor
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(2560, 1440)

# Left Renderer - 3D Perspective View
renderer1 = vtk.vtkRenderer()
renderer1.SetViewport(0, 0, 0.5, 1)  # Left half of the window

# Adjusted camera position and view up vectors
renderer1.GetActiveCamera().SetPosition(
    0, 0, 500
)  # Position the camera on the positive z-axis
renderer1.GetActiveCamera().SetFocalPoint(0, 0, 0)
renderer1.GetActiveCamera().SetViewUp(0, -1, 0)  # Flip the up vector

renderWindow.AddRenderer(renderer1)


# Right Renderer - Overhead 2D View
renderer2 = vtk.vtkRenderer()
renderer2.SetViewport(0.5, 0, 1, 1)  # Right half of the window
renderer2.GetActiveCamera().SetPosition(0, -500, 0)
renderer2.GetActiveCamera().SetFocalPoint(0, 0, 0)
renderer2.GetActiveCamera().SetViewUp(0, 0, 1)
renderer2.GetActiveCamera().ParallelProjectionOn()
renderer2.GetActiveCamera().SetParallelScale(100)
renderWindow.AddRenderer(renderer2)

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the actor to both renderers
renderer1.AddActor(actor)
renderer2.AddActor(actor)

# Start the visualization
renderWindow.Render()
renderWindowInteractor.Start()
