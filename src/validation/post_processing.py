import numpy as np
import os
import fenics as fa
import sys
import matplotlib.pyplot as plt
from .. import arguments


def vis_single(points_ref, c):
    N = len(points_ref)
    for i in range(N):
        for j in range(N - 1):
            plt.plot([points_ref[i, j, 0], points_ref[i, j + 1, 0]],
                     [points_ref[i, j, 1], points_ref[i, j + 1, 1]],
                     marker='o', markersize=5, linewidth=1, color=c)
    for j in range(N):
        for i in range(N - 1):
            plt.plot([points_ref[i, j, 0], points_ref[i + 1, j, 0]],
                     [points_ref[i, j, 1], points_ref[i + 1, j, 1]],
                     marker='o', markersize=5, linewidth=1, color=c)
    plt.axis('off')
    plt.axis('equal')


def vis_outline(args, points_ref):
    N = len(points_ref)
    plt.plot([0, args.L0 * N * 2], [0, 0],
             linestyle='--', linewidth=1, color='black')
    plt.plot([0, args.L0 * N * 2], [args.L0 * N * 2, args.L0 * N * 2],
             linestyle='--', linewidth=1, color='black')
    plt.plot([0, 0], [0, args.L0 * N * 2],
             linestyle='--', linewidth=1, color='black')
    plt.plot([args.L0 * N * 2, args.L0 * N * 2], [0, args.L0 * N * 2],
             linestyle='--', linewidth=1, color='black')
    plt.axis('off')
    plt.axis('equal')


def create_RVE_centers(args):
    dim = 2
    N = 4
    points_ref = np.empty([N, N, dim])
    for i in range(N):
        for j in range(N):
            points_ref[i, j, 0] = (i * 2 + 1) * args.L0
            points_ref[i, j, 1] = (j * 2 + 1) * args.L0

    return points_ref


def get_disp(args, point, mesh, disp_sol):
    cf = fa.MeshFunction('size_t', mesh, 2)
    region = fa.AutoSubDomain(lambda x, on: x[0] < point[0] + args.L0 and
                              x[0] > point[0] - args.L0 and
                              x[1] < point[1] + args.L0 and
                              x[1] > point[1] - args.L0)
    region.mark(cf, 1)
    dx_sub = fa.Measure('dx', subdomain_data=cf)
    ux = fa.assemble(disp_sol[0] * dx_sub(1)) / \
        fa.assemble(disp_sol[0] / disp_sol[0] * dx_sub(1))
    uy = fa.assemble(disp_sol[1] * dx_sub(1)) / \
        fa.assemble(disp_sol[1] / disp_sol[1] * dx_sub(1))
    return ux, uy


def run_single_python(args, points_ref, name, deform_info, pore_flag):
    N = len(points_ref)
    mesh = fa.Mesh('plots/new_data/sol/post_processing/' + name + '_mesh_' +
                   deform_info + '_pore' + str(pore_flag) + '.xml')
    V = fa.VectorFunctionSpace(mesh, 'P', 1)
    disp_sol = fa.Function(V, 'plots/new_data/sol/post_processing/' + name + '_sol_' +
                           deform_info + '_pore' + str(pore_flag) + '.xml')

    points_def = np.copy(points_ref)
    for i in range(N):
        for j in range(N):
            point = points_ref[i][j]
            ux, uy = get_disp(args, point, mesh, disp_sol)
            points_def[i, j, 0] += ux
            points_def[i, j, 1] += uy
    return points_def


def run_python_plot(args):
    points_ref = create_RVE_centers(args)
    points_def_DNS_pore0 = run_single_python(args, points_ref, 'DNS', 'com', 0)
    points_def_DNS_pore2 = run_single_python(args, points_ref, 'DNS', 'com', 2)
    points_def_NN_pore0 = run_single_python(args, points_ref, 'NN', 'com', 0)
    points_def_NN_pore2 = run_single_python(args, points_ref, 'NN', 'com', 2)

    fig = plt.figure(0)
    vis_single(points_def_DNS_pore0, 'blue')
    vis_single(points_def_NN_pore0, 'red')
    vis_single(points_ref, 'black')
    vis_outline(args, points_ref)
    fig.savefig("cmp_pore0.pdf", bbox_inches='tight', pad_inches=0)

    fig = plt.figure(1)
    vis_single(points_def_DNS_pore2, 'blue')
    vis_single(points_def_NN_pore2, 'red')
    vis_single(points_ref, 'black')
    vis_outline(args, points_ref)
    fig.savefig("cmp_pore1.pdf", bbox_inches='tight', pad_inches=0)

    # plt.show()


def run_single_paraview(args, points_ref, name, deform_info, pore_flag):
    N = len(points_ref)
    mesh = fa.Mesh('plots/new_data/sol/post_processing/' + name + '_mesh_' +
                   deform_info + '_pore' + str(pore_flag) + '.xml')
    V = fa.VectorFunctionSpace(mesh, 'P', 1)
    disp_sol = fa.Function(V, 'plots/new_data/sol/post_processing/' + name + '_sol_' +
                           deform_info + '_pore' + str(pore_flag) + '.xml')
    mesh_avg = fa.RectangleMesh(fa.Point(args.L0, args.L0), fa.Point((2*N-1) * args.L0, (2*N-1) * args.L0), N - 1, N - 1)
    V_avg = fa.VectorFunctionSpace(mesh_avg, 'P', 1)
    u_avg = fa.Function(V_avg)

    dofs =  V_avg.dofmap().dofs()
    dofs_x = V_avg.tabulate_dof_coordinates().reshape((len(dofs), -1))
    for i, dof_x in enumerate(dofs_x):
        ux, uy = get_disp(args, dof_x, mesh, disp_sol)
        if i % 2 == 0:
            u_avg.vector()[i] = ux
        else:
            u_avg.vector()[i] = uy

    file2 = fa.File('u_DNS.pvd')
    file2 << u_avg
    file1 = fa.File('disp_sol.pvd')
    file1 << disp_sol

    # file2 = fa.File('mesh_DNS.pvd')
    # file2 << mesh_DNS

def run_paraview_plot(args):
    points_ref = create_RVE_centers(args)
    run_single_paraview(args, points_ref, 'DNS', 'com', 0)

if __name__ == '__main__':
    args = arguments.args
    run_paraview_plot(args)
