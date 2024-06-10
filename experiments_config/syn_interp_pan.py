EXP_SYN_INTERP_PAN_CONFIGS={}
EXP_SYN_INTERP_PAN_CONFIGS['max_epoch'] = 4000
EXP_SYN_INTERP_PAN_CONFIGS['kappa'] = [2048, 1024, 512, 256, 128, 64]
EXP_SYN_INTERP_PAN_CONFIGS['runs'] = [0,1,2]
EXP_SYN_INTERP_PAN_CONFIGS['batch_size'] = [-1, -10/9, -10/8, -10/7, -10/6, -10/5, -10/4, -10/3, -10/2, -10/1]
EXP_SYN_INTERP_PAN_CONFIGS['benchmarks_list'] = ["synthetic_kappa"]
EXP_SYN_INTERP_PAN_CONFIGS['losses'] = ["squared_loss"]
EXP_SYN_INTERP_PAN_CONFIGS['is_kernelize'] = 0
EXP_SYN_INTERP_PAN_CONFIGS['variance'] = [0]
opt_list = []

opt_list += [{'name': 'M_SHB_PAN', 'C':2}]

EXP_SYN_INTERP_PAN_CONFIGS['opt_list'] = opt_list