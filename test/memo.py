
# def iterLearning(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num):
#     # gp and safety game
#     # gpmodels = GP(z_train, y_train, args.noise)
#     # symmodel = Symbolic(args, gpmodels, y_train)

#     gpmodels = GP(z_train, y_train, args.noise)
#     symmodel = Symbolic(args, gpmodels, y_train)
#     # return
#     # Q, Qind = symmodel.safeyGame()
#     # np.save('../data/Q2.npy', Q)
#     # np.save('../data/Qind2.npy', Qind)
#     # return

#     # etmpc
#     etmpc = ETMPC(args, gpmodels, y_train, symmodel.gamma)
#     mpc, simulator, estimator = etmpc.setUp()
#     trigger_status, trigger_values2 = etmpc.triggerValue2(mpc)

#     flag_asm = 0
#     while 1:
#         ze_train = np.zeros((1, 5))
#         ye_train = np.zeros((1, 3))
#         x0 = np.array([np.random.rand(1) + 1,
#                        np.random.rand(1) + 1, 1 * np.random.rand(1)])
#         while 1:
#             etmpc.set_initial(mpc, simulator, estimator, x0)
#             mpc_status, ulist = etmpc.operation(
#                 mpc, simulator, estimator, x0)
#             trigger_status, trigger_values3 = etmpc.triggerValue3(
#                 mpc, trigger_values2)
#             print(trigger_values3)
#             # trigger_status, trigger_values = etmpc.triggerValue(mpc)

#             print('mpc status:', mpc_status)
#             print('trigger status:', trigger_status)
#             if mpc_status and trigger_status == 'optimal':
#                 flag_asm = 1
#                 ulist_pre = ulist
#                 for i in range(args.horizon + 1):
#                     if i == 0:
#                         xe = x0.reshape(-1)
#                         xr = np.array(args.xinit_r).reshape(-1)
#                         traj_data[iter_num] = np.concatenate(
#                             [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
#                     xstar = np.array(
#                         mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

#                     if etmpc.triggerF(xstar, xe, trigger_values3[i, :]):
#                         if i == args.horizon:
#                             return [1, traj_data, trigger_data]
#                         u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
#                         ur = np.array([args.v_r, args.omega_r])

#                         xe_next = vehicle.errRK4(xe, u)
#                         xr_next = vehicle.realRK4(xr, ur)

#                         traj_data[iter_num] = np.concatenate(
#                             [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

#                         ze_train, ye_train = etmpc.learnD(
#                             xe, u, xe_next, ze_train, ye_train)

#                         xe = xe_next
#                         xr = xr_next
#                     else:
#                         trigger_time = i
#                         x0 = xe
#                         trigger_data[iter_num] = np.concatenate(
#                             [trigger_data[iter_num], np.array([trigger_time])])
#                         print('trigger:', trigger_time)
#                         break
#             else:
#                 if flag_asm == 0:
#                     print('assumption is not hold.')
#                     break
#                 elif flag_asm == 1:
#                     for i in range(args.horizon - trigger_time):
#                         if i == 0:
#                             xe = x0.reshape(-1)
#                             traj_data[iter_num] = np.concatenate(
#                                 [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
#                         u = ulist_pre[trigger_time + i]
#                         ur = np.array([args.v_r, args.omega_r])

#                         xe_next = vehicle.errRK4(
#                             xe, u)
#                         xr_next = vehicle.realRK4(xr, ur)

#                         traj_data[iter_num] = np.concatenate(
#                             [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

#                         xe = xe_next
#                     x0 = xe
#                     z_train_sum, y_train_sum = etmpc.dataCat(
#                         ze_train[1:], ye_train[1:])
#                     return [0, z_train_sum, y_train_sum]


# if b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0] < 0:
# return np.sqrt(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])

# def triggerValue(self, mpc):
#     trigger_values = np.array(
#         [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
#     for i in reversed(range(self.horizon)):
#         xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
#         usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
#         zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)

#         _, stdsuc = self.gpmodels.predict(zsuc)

#         psi = cp.Variable(3, pos=True)
#         if i == self.horizon - 1:
#             pg = np.array([self.gamma, self.gamma, self.gamma])
#         c = self.cF(pg)

#         constranits = [cp.quad_form(cp.multiply(self.b * self.y_std, psi) + self.beta *
#                                     self.y_std * stdsuc, np.linalg.inv(self.Lambdax)) <= np.min(c) ** 2]
#         constranits += [psi[j] <= 1.41213 * self.alpha for j in range(3)]

#         trigger_func = cp.geo_mean(psi)
#         prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranits)
#         prob_trigger.solve(solver=cp.CVXOPT)

#         if prob_trigger.status == 'infeasible':
#             return prob_trigger.status, 0
#         pg = psi.value
#         trigger_values = np.concatenate(
#             [trigger_values, psi.value.reshape(1, -1)], axis=0)
#     return prob_trigger.status, np.flip(trigger_values)
# self.y_mean = np.mean(y_train, axis=0)
# self.y_std = np.std(y_train, axis=0)
# xvar_next = vertcat(xvar[0] + self.y_std[0] * mu[0] + self.y_mean[0],
#                     xvar[1] + self.y_std[1] * mu[1] + self.y_mean[1],
#                     xvar[2] + self.y_std[2] * mu[2] + self.y_mean[2])
# print(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])
# return np.sqrt(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])
# y_train_sum = self.y_std * self.gpmodels.gpr.y_train_ + self.y_mean

# alpha0 = np.sqrt(
#             np.exp(gpmodels.gpr[0].kernel_.theta[0]))

#         alpha1 = np.sqrt(
#             np.exp(gpmodels.gpr[1].kernel_.theta[0]))

#         alpha2 = np.sqrt(
#             np.exp(gpmodels.gpr[2].kernel_.theta[0]))

#         Lambda0 = np.diag(np.exp(gpmodels.gpr[0].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64)

#         Lambda1 = np.diag(np.exp(gpmodels.gpr[1].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64)

#         Lambda2 = np.diag(np.exp(gpmodels.gpr[2].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64)

#         print('lambda')

#         print(Lambda0)

#         print(Lambda1)

#         print(Lambda2)

#         gamma0 = self.setGamma(np.sqrt(
#             np.exp(gpmodels.gpr[0].kernel_.theta[0])), np.diag(np.exp(gpmodels.gpr[0].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64),
# self.zlattice)

#         gamma1 = self.setGamma(np.sqrt(
#             np.exp(gpmodels.gpr[1].kernel_.theta[0])), np.diag(np.exp(gpmodels.gpr[1].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64),
# self.zlattice)

#         gamma2 = self.setGamma(np.sqrt(
#             np.exp(gpmodels.gpr[2].kernel_.theta[0])), np.diag(np.exp(gpmodels.gpr[2].kernel_.theta[1:1 + 3]) ** 2).astype(np.float64),
# self.zlattice)

#         print('gamma')
#         print(gamma0)
#         print(gamma1)
#         print(gamma2)

#         cin0 = self.setC(alpha0, gamma0)
#         cin1 = self.setC(alpha1, gamma1)
#         cin2 = self.setC(alpha2, gamma2)

#         ellin0 = np.diag(cin0 * np.sqrt(Lambda0)
#                          ).reshape(-1).astype(np.float64)
#         ellin1 = np.diag(cin1 * np.sqrt(Lambda1)
#                          ).reshape(-1).astype(np.float64)
#         ellin2 = np.diag(cin2 * np.sqrt(Lambda2)
#                          ).reshape(-1).astype(np.float64)

#         print('ellin')
#         print(ellin0)
#         print(ellin1)
#         print(ellin2)

# class GP2:
#     def __init__(self, z_train, y_train, noise):
#         self.z_train = z_train
#         self.y_train = y_train
#         self.noise = noise
#         self.rbfk = [RBF(length_scale=np.ones(
#             z_train.shape[1]), length_scale_bounds=(1e-20, 1e20)) for i in range(3)]
#         self.whtk = [WhiteKernel(noise_level=self.noise, noise_level_bounds=(1e-20, 1e20)) for i in range(3)]
#         self.csk = [ConstantKernel(constant_value_bounds=(1e-20, 1e20)) for i in range(3)]
#         self.gpr = [GaussianProcessRegressor(
#             alpha=1e-4,
#             kernel=self.csk[i] * self.rbfk[i] + self.whtk[i],
#             # normalize_y=True,
#             n_restarts_optimizer=10,
#             random_state=0,
#             optimizer='fmin_l_bfgs_b',
#         ) for i in range(3)]
#         for i in range(3):
#             self.gpr[i].fit(self.z_train, self.y_train[:, i])

#     def predict(self, z_test):
#         means = np.zeros(3)
#         for i in range(3):
#             means = self.gpr.predict(
#                 z_test)
#         return means

# iterdata = iterLearning(args, vehicle, z_train,
#                         y_train, traj_data, trigger_data, iter_num)

# class Symbolic:
#     def __init__(self, args, gpmodels, y_train):
#         self.gpmodels = gpmodels
#         self.y_mean = np.mean(y_train, axis=0).astype(np.float64)
#         self.y_std = np.std(y_train, axis=0).astype(np.float64)
#         self.ZT = self.gpmodels.gpr.X_train_.astype(np.float64)
#         self.Y = self.gpmodels.gpr.y_train_.astype(np.float64)
#         self.covs = self.gpmodels.gpr.L_ @ self.gpmodels.gpr.L_.T.astype(
#             np.float64)
#         self.alpha = np.sqrt(
#             np.exp(gpmodels.gpr.kernel_.theta[0])).astype(np.float64)
#         self.Lambda = np.diag(
#             np.exp(gpmodels.gpr.kernel_.theta[1:1 + 5]) ** 2).astype(np.float64)
#         self.Lambdax = np.diag(
#             np.exp(gpmodels.gpr.kernel_.theta[1:1 + 3]) ** 2).astype(np.float64)
#         self.noises = np.exp(
#             gpmodels.gpr.kernel_.theta[-1]).astype(np.float64)

#         self.b = np.array(args.b).astype(np.float64)
#         self.etax_param = args.etax_param
#         self.etau = args.etau
#         self.Xsafe = np.array(args.Xsafe).astype(np.float64)
#         self.v_max = args.v_max
#         self.omega_max = args.omega_max

#         self.xqparams = np.sqrt(np.diag(self.Lambdax)) / \
#             np.max(np.sqrt(np.diag(self.Lambdax)))
#         self.etax_v = (self.xqparams * self.etax_param).astype(np.float64)
#         self.zlattice = args.zlattice

#         self.Xqlist = self.setXqlist()
#         self.Uq = self.setUq()
#         self.epsilon = self.setEpsilon(
#             self.alpha, self.Lambdax)
#         self.gamma = self.setGamma(self.alpha, self.Lambdax, self.zlattice)
#         self.cout = self.setC(self.alpha, self.epsilon)
#         self.ellout = np.diag(self.cout * np.sqrt(self.Lambdax)).reshape(-1)
#         self.cin = self.setC(self.alpha, self.gamma)
#         self.ellin = np.diag(self.cin * np.sqrt(self.Lambdax)
#                              ).reshape(-1).astype(np.float64)

#     def setEpsilon(self, alpha, Lambdax):
#         return np.sqrt(2 * (alpha**2) * (1 - np.exp(-0.5 * self.etax_v @ np.linalg.inv(Lambdax) @ self.etax_v)))

#     def setGamma(self, alpha, Lambdax, zlattice):
#         return np.sqrt(2 * (alpha**2) * (1 - np.exp(-(2 * (zlattice ** 2) / 3) * self.etax_v @ np.linalg.inv(Lambdax) @ self.etax_v)))

#     def setC(self, alpha, epsilon):
#         return np.sqrt(2 * np.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

#     def setXqlist(self):
#         for i in range(3):
#             self.Xsafe[i, :] = self.Xsafe[i, :] * self.xqparams[i]
#         return [np.arange(self.Xsafe[i, 0],
#                           self.Xsafe[i, 1] + 0.000001, 2 / np.sqrt(3) * self.etax_v[i]).astype(np.float64) for i in range(3)]

#     def setUq(self):
#         Vq = np.arange(0., self.etau + 0.000001, self.etau)
#         Omegaq = np.arange(0, self.omega_max +
#                            self.etau + 0.000001, self.etau)
#         Uq = np.zeros((Vq.shape[0] * Omegaq.shape[0], 2))
#         for i in range(Vq.shape[0]):
#             for j in range(Omegaq.shape[0]):
#                 Uq[i * Omegaq.shape[0] + j, :] = np.array([Vq[i], Omegaq[j]])
#         return Uq

#     def setQind_init(self):
#         Qinit = np.zeros([self.Xqlist[0].shape[0],
#                           self.Xqlist[1].shape[0], self.Xqlist[2].shape[0]]).astype(np.int)
#         Qind_out = np.ceil(self.ellout / (2 / np.sqrt(3) *
#                                           self.etax_v)).astype(np.int)
#         Qinit[Qind_out[0]: -Qind_out[0] + 1,
#               Qind_out[1]: -Qind_out[1] + 1,
#               Qind_out[2]: -Qind_out[2] + 1] = 1
#         Qind_init_list = np.nonzero(Qinit)
#         return Qinit.tolist(), np.concatenate(
#             [Qind_init_list[0].reshape(-1, 1), Qind_init_list[1].reshape(-1, 1), Qind_init_list[2].reshape(-1, 1)], axis=1).astype(np.float64)

#     def safeyGame(self):
#         Qinit, Qind_init = self.setQind_init()
#         sgflag = 1
#         while sgflag == 1:
#             Q = sg.operation(Qinit, Qind_init, self.alpha, self.Lambda, self.Lambdax, self.covs,
#                              self.noises, self.ZT, self.Y, self.b, self.Xqlist,
#                              self.Uq, self.etax_v, self.epsilon, self.gamma,
#                              self.y_mean, self.y_std, self.ellin)
#             Qindlist = np.nonzero(np.array(Q))
#             Qind = np.concatenate([Qindlist[0].reshape(-1, 1), Qindlist[1].reshape(-1, 1),
#                                    Qindlist[2].reshape(-1, 1)], axis=1)
#             if Qind_init.shape[0] == Qind.shape[0]:
#                 if Qind.shape[0] == 0:
#                     print('empty.')
#                     return
#                 sgflag = 0
#                 print('complete.')
#                 return Q, Qind
#             else:
#                 Qinit = Q
#                 Qind_init = Qind.astype(np.float64)
#                 print('continue..')
