#deploy_nn.py
def manual_gpr(network, C_list):
    result0 = 0
    for i, x in enumerate(X_train[0:900]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result0 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i]

    result1 = 0
    for i, x in enumerate(X_train[900:1800]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result1 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 900]

    result2 = 0
    for i, x in enumerate(X_train[1800:2700]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result2 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 1800]

    result3 = 0
    for i, x in enumerate(X_train[2700:3600]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result3 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 2700]

    result4 = 0
    for i, x in enumerate(X_train[3600:4500]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result4 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 3600]

    result5 = 0
    for i, x in enumerate(X_train[4500:5400]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result5 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 4500]

    result6 = 0
    for i, x in enumerate(X_train[5400:]):
        tmp = (x[0] / l[0] - C_list[0] / l[0])**2 + \
              (x[1] / l[1] - C_list[1] / l[1])**2 + \
              (x[2] / l[2] - C_list[2] / l[2])**2 + \
              (x[3] / l[3] - C_list[3] / l[3])**2
        result6 += sigma_f**2 * ufl.operators.exp(-0.5 * tmp) * v[i + 5400]

    return result0 + result1 + result2 + result3 + result4 + result5 + result6


# Put constraint for large deformation - energy must be very large
def manual_nn(network, x):
    x1 = layer(x, network.fc1.weight, network.fc1.bias, 1)
    x2 = layer(x1, network.fc2.weight, network.fc2.bias, 2)
    bounds = [2, 0.65, 1.6, -1.6, 2, 0.65]
    # bounds = [1.1, 0.9, 1.1, -1.1, 1.1, 0.9]
    energy_bound = 1e10
    result = x2[0]
    result = conditional(x[0] > bounds[0], energy_bound, result)
    result = conditional(x[0] < bounds[1], energy_bound, result)
    result = conditional(x[1] > bounds[2], energy_bound, result)
    result = conditional(x[1] < bounds[3], energy_bound, result)
    result = conditional(x[2] > bounds[4], energy_bound, result)
    result = conditional(x[2] < bounds[5], energy_bound, result)
    return result

if __name__ == '__main__':
    args = arguments.args
    args.n_macro = 8
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 2000

    # GPR related
    params = np.load('plots/new_data/numpy/gpr/para.npz')
    l = params['l']
    sigma_f = params['sigma_f']
    X_train = params['X_train']
    v = params['v']
    energy, force, u = homogenization(args, disp=-0.1, pore_flag=1)

#size_effect.py
for i, af in enumerate(generator.anneal_factors):
    file = fa.File('plots/new_data/sol/instability_tracking/size' + str(size) + '_step' + str(i) + '_.pvd')
    sols[i].rename('u', 'u')
    file << sols[i]

DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore0_size' + str(8)  + '_good.npy')
plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='blue')

DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore0_size' + str(8)  + '_bad.npy')
plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='red')

mesh_name = 'plots/new_data/mesh/' + 'DNS' + '_size' + str(size) + '_pore' + str(2) + '.xml'
if os.path.isfile(mesh_name):
    mesh = fa.Mesh(mesh_name)
    V = fa.VectorFunctionSpace(mesh, 'P', 1)
    disp_sol = fa.Function(V, 'plots/new_data/sol/intermediate/size' + str(size) + '_disp_0.07600.xml')
    print("load solution")
else:
    disp_sol = None


# Remarks:
# size8 pore0
# generator.anneal_factors = np.linspace(0.76, 1, 21)

# size21 pore0
# generator.anneal_factors = np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11)))


# regr = MLPRegressor(hidden_layer_sizes=(256,), activation='logistic', solver='adam', alpha=0,
#                     batch_size=32, learning_rate_init=1e-2, max_iter=2000, random_state=1,
#                     tol=1e-9, verbose=True, n_iter_no_change=1000).fit(self.X_train, self.y_train)
