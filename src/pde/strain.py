import fenics as fa

# Deformation gradient
def DeformationGradient(u):
    I = fa.Identity(u.geometric_dimension())
    return fa.variable(I + fa.grad(u))

# Determinant of the deformation gradient
def DetDeformationGradient(u):
    F = DeformationGradient(u)
    return fa.variable(fa.det(F))

# Right Cauchy-Green tensor
def RightCauchyGreen(F):
    return fa.variable(F.T * F)

# Invariants of an arbitrary tensor, A
def Invariants(A):
    I1 = fa.tr(A)
    I2 = 0.5 * (fa.tr(A)**2 - fa.tr(A * A))
    I3 = fa.det(A)
    return [I1, I2, I3]

def NeoHookeanEnergy(u, young_mod, poisson_ratio, return_stress=False, fluctuation=False, F_list=None):

    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}"
            .format(poisson_ratio))

    if fluctuation:
        return NeoHookeanEnergyFluctuation(u, young_mod, poisson_ratio, return_stress, F_list)

    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2*poisson_ratio))
    d = u.geometric_dimension()
    F = DeformationGradient(u)
    F = fa.variable(F)
    J = fa.det(F)
    I1 = fa.tr(RightCauchyGreen(F))

    # Plane strain assumption
    Jinv = J**(-2 / 3)
    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2)  

    # Pure 2d assumption 
    # Jinv = J**(-2 / d)
    # energy = ((shear_mod / 2) * (Jinv * I1 - d) +
    #           (bulk_mod / 2) * (J - 1)**2)

    if return_stress:
        FinvT = fa.inv(F).T
        first_pk_stress = (Jinv * shear_mod * (F - (1 / 3) * (I1 + 1) * FinvT) +
                           J * bulk_mod * (J - 1) * FinvT)

        first_pk_stress = fa.diff(energy, F)

        return energy, first_pk_stress

    return energy

# Deformation gradient
def DeformationGradientFluctuation(v, F_list):
    F_I = fa.as_matrix(F_list)
    grad_u = fa.grad(v) + F_I  
    I = fa.Identity(v.geometric_dimension())
    return fa.variable(I + grad_u)

def NeoHookeanEnergyFluctuation(v, young_mod, poisson_ratio, return_stress, F_list):
 
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2*poisson_ratio))
    d = v.geometric_dimension()
    F = DeformationGradientFluctuation(v, F_list)
    F = fa.variable(F)
    J = fa.det(F)
    Jinv = J**(-2 / 3)
    I1 = fa.tr(RightCauchyGreen(F))

    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
 
    if return_stress:
        first_pk_stress = fa.diff(energy, F)
        return energy, first_pk_stress

    return energy




