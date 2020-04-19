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
    I1_b = J**(-2 / d) * I1

    shear_mod = args.young_modulus / (2 * (1 + args.poisson_ratio))
    bulk_mod = args.young_modulus / (3 * (1 - 2 * args.poisson_ratio))

    Energy = Xout - 0.1 * (J - 1)**2
    # Xout = (Xout + 1)/((shear_mod / 2) * (I1_b - d) + (bulk_mod / 2) * (J - 1)**2 + 1)

    return Energy


def run_trainer(args, Xin, Xout, lr, bsize, hidden_dim):
    args.lr = lr
    args.batch_ize = bsize
    args.hidden_dim = hidden_dim

    nSamps = len(Xin)
    nTrain = int((1 - args.val_fraction) * nSamps)
    inds = np.random.permutation(nSamps)
    indsTrain = inds[:nTrain]
    indsTest = inds[nTrain:]

    Xt_in = Xin[indsTest]
    Xt_out = Xout[indsTest]

    Xin = Xin[indsTrain]
    Xout = Xout[indsTrain]

    nTest = len(Xt_in)
    fAv = fAvTest = f_dev = f_dev_test = 0.

    model = Linear(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    epoch_number = 1000
    log_interval = 100
    step_number = nTrain // args.batch_ize

    epoch = []
    mse_train = []
    mse_test = []
    mpe_train = []
    mpe_test = []

    for ep_num in range(epoch_number):
        for step in range(1, step_number):
            optimizer.zero_grad()

            # Testing
            inds_test = np.random.randint(0, nTest, args.batch_ize)
            inBatch_test = torch.tensor(
                Xt_in[inds_test], requires_grad=True).float()
            outBatch_test = torch.tensor(Xt_out[inds_test]).view(-1, 1).float()
            fhat_test = model(inBatch_test)
            f_test = outBatch_test
            f_loss_test = torch.sum((f_test - fhat_test)**2)
            fAvTest += float(f_loss_test.data)
            # mean percentage error (MPE)
            f_dev_test += float(torch.sum(abs(f_test -
                                              fhat_test) / (f_test + 1e-5)).data)

            # Training
            inds = np.random.randint(0, nTrain, args.batch_ize)
            inBatch = torch.tensor(Xin[inds], requires_grad=True).float()
            outBatch = torch.tensor(Xout[inds]).view(-1, 1).float()
            fhat = model(inBatch)
            f = outBatch
            f_loss = torch.sum((f - fhat)**2)
            fAv += float(f_loss.data)
            f_dev += float(torch.sum(abs(f - fhat) / (f + 1e-5)).data)

            total_loss = f_loss
            total_loss.backward()
            optimizer.step()

        if (ep_num + 1) % log_interval == 0:
            print('\nepoch:', ep_num + 1)
            epoch.append(ep_num)
            train_mse = compute_MSE(model, Xin, Xout)
            test_mse = compute_MSE(model, Xt_in, Xt_out)
            print("MSE for training", train_mse, "MSE for testing", test_mse)
            mse_train.append(train_mse)
            mse_test.append(test_mse)
            # torch.save(model, MODEL_PATH + '/model_step_'+str(ep_num))
            fAv = fAvTest = f_dev = f_dev_test = 0.

    print('\n\n')
    print("MSE for training", compute_MSE(model, Xin, Xout))
    print("MSE for testing", compute_MSE(model, Xt_in, Xt_out))

    return compute_MSE(model, Xt_in, Xt_out)
