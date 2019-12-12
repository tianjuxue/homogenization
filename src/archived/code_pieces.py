def feature_map(C, C_square, I1_b, J, n=2, m=1):
    d = 2
    N_samples = I1_b.shape[0]
    N_features = (n + 1) + m
    Features = np.zeros((N_samples, N_features))
    count = 0

    for i in range(n + 1):
        Features[:, count] = (I1_b - d)**i 
        count = count + 1

    for i in range(1, m + 1):
        # Features[:, count] = np.log(J)**(2*i)
        Features[:, count] = (J - 1)**(2*i)
        count = count + 1

    return Features

def get_features_general(Xin):
    d = 2
    N_samples = Xin.shape[0]
    F = Xin.copy()
    F = F[:, 0:4]
    F[:, 0] = F[:, 0] + 1
    F[:, 3] = F[:, 3] + 1
    F = F.reshape(N_samples, 2, 2)
    F_T = np.transpose(F, (0, 2, 1))
    C = np.matmul(F_T, F)
    C_square = np.matmul(C, C)
    J = np.linalg.det(F)
    I1 = np.trace(C, axis1=1, axis2=2)
    I1_b = J**(-2/d) * I1
    Features = feature_map(C, C_square, I1, J)
    # Features = feature_map_essential(C)

    return Features

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
    
    Energy = Xout - 10*(J - 1)**2
    # Xout = (Xout + 1)/((shear_mod / 2) * (I1_b - d) + (bulk_mod / 2) * (J - 1)**2 + 1)
 
    return Energy  



# Proof of concept dirty code delete later
class PointBoundary0(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], 0) and fa.near(x[1], 0)

class PointBoundary1(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 1) and fa.near(x[1], 0)

class PointBoundary2(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 2) and fa.near(x[1], 0)

class PointBoundary3(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], 0) and fa.near(x[1], L0 * 1)

class PointBoundary4(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 1) and fa.near(x[1], L0 * 1)

class PointBoundary5(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 2) and fa.near(x[1], L0 * 1)

class PointBoundary6(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], 0) and fa.near(x[1], L0 * 2)

class PointBoundary7(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 1) and fa.near(x[1], L0 * 2)

class PointBoundary8(fa.SubDomain):
    def inside(self, x, on_boundary):
        return fa.near(x[0], L0 * 2) and fa.near(x[1], L0 * 2)


# Proof of concept dirty code delete later
self.corner_dic = [PointBoundary0(), PointBoundary1(), PointBoundary2(), 
                   PointBoundary3(), PointBoundary4(), PointBoundary5(),
                   PointBoundary6(), PointBoundary7(), PointBoundary8()]



# shear
def get_parameters_sobol(sobol_vec):

    parameters = [0.5*(sobol_vec[0] - 0.5),
                  1.6*(sobol_vec[1] - 0.5),
                  1.6*(sobol_vec[2] - 0.5),
                  0.5*(sobol_vec[3] - 0.5),
                  -0.2,
                  0.2]

    return parameters

# shear1
division_array = np.array([4, 5, 5, 4])
# shear2
division_array = np.array([3, 3, 3, 3])


# normal
def get_parameters_sobol(sobol_vec):

    parameters = [0.7*(sobol_vec[0] - 0.5),
                  0.4*(sobol_vec[1] - 0.5),
                  0.4*(sobol_vec[2] - 0.5),
                  0.7*(sobol_vec[3] - 0.5),
                  -0.,
                  0.]

division_array = np.array([5, 3, 3, 5])



mm_t = Net(Xin.shape[1])  
criterion = torch.nn.MSELoss(reduction='mean') 
optimizer = torch.optim.SGD(mm_t.parameters(), lr = 1e-5) 
optimizer = torch.optim.Adam(mm_t.parameters(), lr = 1e-3, weight_decay=args.wd)
Xin = torch.tensor(Xin).float()
Xout = torch.tensor(Xout).float()

for t in range(100):
    y_pred = mm_t(Xin)
    loss = criterion(y_pred, Xout)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 

if step % step_interval == 0:
    plt.figure()
    index = np.linspace(0, 1, len(f))
    plt.plot(index, f.data.numpy(), '-*', label='true')
    plt.plot(index, fhat.data.numpy(), '-*', label='predicted')            
    plt.legend(loc='upper right')
    plt.show()
if step == total_steps - 1:
    # print(mm_t.fc1.weight)
    plt.figure()
    index = np.linspace(0, 1, len(f))
    plt.plot(index, f.data.numpy(), '-*', label='true')
    plt.plot(index, fhat.data.numpy(), '-*', label='predicted')            
    plt.legend(loc='upper right')
    plt.show()
if step % step_interval == 0:

    total_number = step_interval*batchSize
    print('\nepoch: ', step)
    print('MSE of training: ', fAv/total_number, ' and testing: ', fAvTest/total_number)  
    print('MPE of training: ', f_dev/total_number, 'and testing: ', f_dev_test/total_number)
    torch.save(mm_t, MODEL_PATH + '/model_step_'+str(step))
    epoch.append(step//step_interval)
    mse_train.append(fAv/total_number)
    mse_test.append(fAvTest/total_number)   
    mpe_train.append(f_dev/total_number)
    mpe_test.append(f_dev_test/total_number)      
    fAv = fAvTest = f_dev = f_dev_test = 0. 


Xin = Xin_shear
Xout = Xout_shear
Xin, Xout = remove_duplicated(Xin, Xout)
for i, xin in enumerate(Xin):
    if np.absolute(xin[0]) < 1e-3 and \
       np.absolute(xin[2]) < 1e-3 and \
       np.absolute(xin[3]) < 1e-3:
        print(xin)
        print(Xout[i])
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)
Xin, Xout = remove_duplicated(Xin, Xout)


print(clf.score(Xin, Xout))
theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Xin), Xin)), np.transpose(Xin)), Xout)
print(theta)

python3 -m src.collector.driver --metamaterial_mesh_size 10 --relaxation_parameter 0.1 --max_newton_iter 200 --verbose