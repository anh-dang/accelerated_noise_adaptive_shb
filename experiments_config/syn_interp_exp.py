EXP_SYN_INTERP_CONFIGS={}
EXP_SYN_INTERP_CONFIGS['max_epoch'] = 2000
EXP_SYN_INTERP_CONFIGS['kappa'] = [256, 128, 64]
EXP_SYN_INTERP_CONFIGS['runs'] = [0,1,2]
EXP_SYN_INTERP_CONFIGS['batch_size'] = [-1, -10/9, -10/8, -10/7, -10/6, -10/5, -10/4, -10/3, -10/2, -10/1]
EXP_SYN_INTERP_CONFIGS['benchmarks_list'] = ["synthetic_kappa"]
EXP_SYN_INTERP_CONFIGS['losses'] = ["squared_loss"]
EXP_SYN_INTERP_CONFIGS['is_kernelize'] = 0
opt_list = []

# SHB
for alpha_t in ['CNST']:
    for c in [1]:
        opt_list += [{'name': 'EXP_SHB',
                    'alpha_t': alpha_t,
                    'method': 'WANG21',
                    'is_sls': False,
                    'mis_spec': 1.0,
                    'ada': None,
                    'ld': None,
                    'ld_sche': None,
                    'c':c
                    }]

opt_list += [{'name': 'EXP_SHB',
            'alpha_t': 'CNST',
            'method': 'SEBBOUH',
            'is_sls': False,
            'mis_spec': 1.0,
            'ada': None,
            'ld': None,
            'ld_sche': None,
            'c': 1
            }]

opt_list += [{'name': 'EXP_SGD',
            'alpha_t': "CNST",
            'is_sls': False,
            'ada': None}]

EXP_SYN_INTERP_CONFIGS['opt_list'] = opt_list