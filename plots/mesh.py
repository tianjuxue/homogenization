'''这个文件主要是作local rotational feature的图，参加paper的appendix
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from matplotlib import pyplot
from matplotlib import collections as mc
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from vtk.util.numpy_support import vtk_to_numpy
import fenics as fa


def show_solution(path):
    # plot from vtk
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    # reader.SetFileName("u000000.vtu")
    reader.Update()

    data = reader.GetOutput()

    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('u'))

    x_ = x + u

    print(x.shape)

    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4  # number of cells
    tri = np.take(triangles, [n for n in range(
        triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    # fig = plt.figure(figsize=(8, 8))
    plt.triplot(x_[:, 0], x_[:, 1], tri, color='gray', alpha=0.5)

    # colors = u[:, 0]
    # tpc = plt.tripcolor(x_[:, 0], x_[:, 1], tri, colors, cmap='bwr', shading='flat', vmin=None, vmax=None)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    # plt.show()
    return x[:, :2], x_[:, :2]


def find_nearest(array, value):
    array = np.asarray(array)
    abs_dist = np.abs(array - value).sum(axis=1)
    idx = abs_dist.argmin()
    return idx if abs_dist[idx] < 1e-10 else None


def get_inds(bmesh_coos, x, pore_flag):
    inds = []
    for i, bcoos in enumerate(bmesh_coos):
        if i % 1000 == 0:
            print("i", i)
        indx = find_nearest(bcoos, x)
        assert indx is not None, "None value found"
        inds.append(indx)
    inds = np.asarray(inds)
    np.save('plots/new_data/numpy/inds/inds_pore{}.npy'.format(pore_flag), inds)
    return inds


def get_boundary_coos(pore_flag):
    mesh = fa.Mesh(
        'plots/new_data/sol/post_processing/input/DNS_mesh_com_pore' + str(pore_flag) + '.xml')
    bmesh = fa.BoundaryMesh(mesh, "exterior", True)
    bmesh_coos = bmesh.coordinates()
    print(bmesh_coos.shape)
    return bmesh_coos


def get_pair(points):
    max_index = (0, 0)
    tmp = 0
    for i in range(0, len(points)):
        for j in range(i, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > tmp:
                max_index = (i, j)
                tmp = dist
    return max_index


def arrows(x, x_, pore_flag):
    inds = np.load(
        'plots/new_data/numpy/inds/inds_pore{}.npy'.format(pore_flag))
    boundary_points = x[inds]
    boundary_points_ = x_[inds]
    N = 16
    L0 = 0.5
    grouped_points = [[[] for j in range(N)] for i in range(N)]
    for index, bp in enumerate(boundary_points):
        if bp[0] > 1e-10 and bp[0] < N * L0 - 1e-10 and bp[1] > 1e-10 and bp[1] < N * L0 - 1e-10:
            i = int(bp[0] / L0)
            j = int(bp[1] / L0)
            grouped_points[i][j].append(boundary_points_[index])
    grouped_points = np.asarray(grouped_points)

    print(grouped_points.shape)

    paired_points = [[[] for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            ind1, ind2 = get_pair(grouped_points[i, j])
            paired_points[i][j].append(
                [grouped_points[i, j, ind1], grouped_points[i, j, ind2]])
    paired_points = np.squeeze(np.asarray(paired_points))
    print(paired_points.shape)
    for i in range(N):
        for j in range(N):
            plt.plot(paired_points[i, j, :, 0], paired_points[i, j, :, 1], 
                linestyle='-', color='black')
            # right_point =  paired_points[i, j, 0] if paired_points[i, j, 0, 0] > \
            #                paired_points[i, j, 1, 0] else paired_points[i, j, 1]
            # left_point =  paired_points[i, j, 0] if paired_points[i, j, 0, 0] < \
            #                paired_points[i, j, 1, 0] else paired_points[i, j, 1]
            # x = left_point[0]
            # y = left_point[1]
            # dx = right_point[0] - left_point[0]
            # dy = right_point[1] - left_point[1]
            # plt.arrow(x, y, dx, dy, color='black', head_width=0.1)


def run_single(pore_flag):
    fig = plt.figure(pore_flag)
    path = 'plots/new_data/sol/post_processing/output/DNS_sol_com_pore{}000000.vtu'.format(
        pore_flag)
    x, x_ = show_solution(path)
    if False:
        bmesh_coos = get_boundary_coos(pore_flag)
        inds = get_inds(bmesh_coos, x, pore_flag)
    arrows(x, x_, pore_flag)
    fig.savefig('plots/new_data/images/local_feature_pore{}.png'.format(pore_flag), bbox_inches='tight')

def run():
    run_single(0)
    run_single(2)

if __name__ == '__main__':
    run()
    plt.show()
