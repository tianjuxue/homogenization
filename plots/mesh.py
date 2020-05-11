import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from matplotlib import pyplot
from matplotlib import collections as mc
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from vtk.util.numpy_support import vtk_to_numpy


if __name__ == '__main__':

    # plot from vtk
    reader = vtk.vtkXMLUnstructuredGridReader()
    # reader.SetFileName("plots/data/tension/homo_gradient.vtu")
    reader.SetFileName("u000000.vtu")
    reader.Update()

    data = reader.GetOutput()

    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('u'))

    # x = x + u

    triangles =  vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size//4  # number of cells
    tri = np.take(triangles, [n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)

    fig = plt.figure(figsize=(8, 8))
    plt.triplot(x[:,0], x[:,1], tri, linewidth=0.5, color='r')    
    # plt.triplot(x[:,0], x[:,1], tri, marker='o', markersize=1, linewidth=0.5, color='r')
    plt.gca().set_aspect('equal')
    plt.axis('off')

    plt.show()
