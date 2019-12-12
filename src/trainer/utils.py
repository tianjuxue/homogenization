'''Some of the old functions. Not sure useful anymore'''

def remove_duplicated(Xin, Xout):
    duplicated_index = []
    for i, xin in enumerate(Xin):
        if np.sum(np.absolute(xin)[0:4]) < 1e-5:
            duplicated_index.append(i)

    xin_zero = Xin[duplicated_index[0]].reshape(1, -1)
    xout_zero = Xout[duplicated_index[0]].reshape(1)


    Xin = np.delete(Xin, duplicated_index, axis=0)
    Xin = np.concatenate((Xin, xin_zero))
    Xout = np.delete(Xout, duplicated_index, axis=0)
    Xout = np.concatenate((Xout, xout_zero))
    return Xin, Xout

def get_labels(Xin, Xout):
    d = 2
    N_samples = Xin.shape[0]
    F = Xin.copy()
    F = F[:, 0:4]
    F[:, 0] = F[:, 0] + 1
    F[:, 3] = F[:, 3] + 1
    F = F.reshape(N_samples, 2, 2)
    F_T = np.transpose(F, (0, 2, 1))
    C = np.matmul(F_T, F)

    J = np.linalg.det(F)
    I1 = np.trace(C, axis1=1, axis2=2)
    I1_b = J**(-2/d) * I1

    shear_mod = args.young_modulus / (2 * (1 + args.poisson_ratio))
    bulk_mod = args.young_modulus / (3 * (1 - 2*args.poisson_ratio))
    
    Energy = Xout - 0.1*(J - 1)**2
    # Xout = (Xout + 1)/((shear_mod / 2) * (I1_b - d) + (bulk_mod / 2) * (J - 1)**2 + 1)
 
    return Energy 