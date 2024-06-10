EXP_SYN_NON_INTERP_COMPARE_CONFIGS={}
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['max_epoch'] = 7000
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['kappa'] =  [2000, 1000, 500, 200, 100]
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['runs'] = [0,1]
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['batch_size'] = [-10/9]
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['benchmarks_list'] = ["synthetic_kappa"]
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['losses'] = ["squared_loss"]
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['is_kernelize'] = 0
EXP_SYN_NON_INTERP_COMPARE_CONFIGS['variance'] = [1e-2, 1e-4, 1e-6, 1e-8]
opt_list = []

for c in [0.4]:
    for beta in [True, False]:
        opt_list += [{'name': 'M_ASHB', 'c':c, 'beta_const':beta}]

for C in [2, 'max']:
    opt_list += [{'name': 'M_SHB_PAN', 'C':C}]

opt_list += [{'name': 'Mix_SHB', 'c':0.5}]

EXP_SYN_NON_INTERP_COMPARE_CONFIGS['opt_list'] = opt_list