# inspection.py
try:
    generator._anealing_solver_fluctuation(False)
    print(generator.energy_density)
    print(generator.probe_all)
except Exception as e:
    print(e)
    print(generator.energy_density)
    print(generator.probe_all)

# generator.py
def generate_data_series(self, def_grad=None, void_shape=None):
    if def_grad is not None:
        self.def_grad = def_grad
    if void_shape is not None:
        self.void_shape = void_shape

    self.energy_density_list = []

    for n_cells in self.n_cells_array:
        print("  n_cells is", n_cells)
        self.args.n_cells = n_cells
        energy_density = self._anealing_solver()
        self.energy_density_list.append(energy_density)

    self.energy_density_array = np.array(self.energy_density_list).T
    self.predicted_energy_all = [self._compute_fitted_energy(phi) 
                                for phi in self.energy_density_array]

    self.def_grad_all = []
    self.void_shape_all = []

    for i, factor in enumerate(self.anneal_factors):  
        self.def_grad_all.append(factor*self.def_grad)
        self.void_shape_all.append(self.void_shape) 

    return self.def_grad_all, self.void_shape_all, self.predicted_energy_all, self.energy_density_array

# generator.py
def _compute_fitted_energy(self, phi):
    assert(self.n_cells_array.shape == phi.shape)
    x = 1/self.n_cells_array
    y = phi
    linear_coeff = np.polyfit(x, y, 1)
    return linear_coeff[1]

# generator.py
class GeneratorDummy(Generator):
    def generate_data(self):
        self.def_grad_all = []
        self.void_shape_all = []
        for i, factor in enumerate(self.anneal_factors):  
            self.def_grad_all.append(factor*self.def_grad)
            self.void_shape_all.append(self.void_shape) 
        self.predicted_energy_all = [self._compute_energy(F) for F in self.def_grad_all]
        return self.def_grad_all, self.void_shape_all, self.predicted_energy_all

    def _compute_energy(self, F):
        shear_mod = self.args.young_modulus / (2 * (1 + self.args.poisson_ratio))
        bulk_mod = self.args.young_modulus / (3 * (1 - 2*self.args.poisson_ratio))
        d = 2
        F = np.array([F[0:2], F[2:4]]) + np.eye(2)
        C = np.matmul(F.T, F)
        J = np.linalg.det(F)
        Jinv = J**(-2 / d)
        I1 = np.trace(C)
        energy = (shear_mod / 2) * (Jinv * I1 - d) + (bulk_mod / 2) * (J - 1)**2

        # print((Jinv * I1 - d))
        return energy