EXP_SYN_NON_INTERP_CONFIGS={}
EXP_SYN_NON_INTERP_CONFIGS['max_epoch'] = 7000
EXP_SYN_NON_INTERP_CONFIGS['kappa'] =  [1000, 500, 200]
EXP_SYN_NON_INTERP_CONFIGS['runs'] = [0,1,2]
EXP_SYN_NON_INTERP_CONFIGS['batch_size'] = [-10/9]
EXP_SYN_NON_INTERP_CONFIGS['benchmarks_list'] = ["synthetic_kappa"]
EXP_SYN_NON_INTERP_CONFIGS['losses'] = ["squared_loss"]
EXP_SYN_NON_INTERP_CONFIGS['is_kernelize'] = 0
EXP_SYN_NON_INTERP_CONFIGS['variance'] = [1e-2, 1e-4, 1e-6]
opt_list = []

opt_list += [{'name': 'EXP_SHB',
        'alpha_t': 'CNST',
        'method': 'WANG21',
        'is_sls': False,
        'mis_spec': 1.0,
        'ada': None,
        'ld': None,
        'ld_sche': None,
        'c':1
        }]

opt_list += [{'name': 'Mix_SHB', 'c':0.5}]

for c in [0.4]:
    for beta_const in [True, False]:
        opt_list += [{'name': 'M_ASHB', 'c':c, 'beta_const':beta_const}]

opt_list += [{'name': 'EXP_SGD',
                      'alpha_t': 'CNST',
                      'is_sls': False,
                      'ada': None
                      }]

opt_list += [{'name': 'EXP_ACC_SGD',
                'alpha_t': "DECR",
                'rho': 1,
                'is_sls': False
                }]

EXP_SYN_NON_INTERP_CONFIGS['opt_list'] = opt_list