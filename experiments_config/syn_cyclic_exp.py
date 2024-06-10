EXP_SYN_CYCLIC_CONFIGS={}
EXP_SYN_CYCLIC_CONFIGS['max_epoch'] = 10000
EXP_SYN_CYCLIC_CONFIGS['kappa'] = [64,32]
EXP_SYN_CYCLIC_CONFIGS['runs'] = [0,1,2]
EXP_SYN_CYCLIC_CONFIGS['batch_size'] = [1, -10/1]
EXP_SYN_CYCLIC_CONFIGS['benchmarks_list'] = ["synthetic_kappa"]
EXP_SYN_CYCLIC_CONFIGS['losses'] = ["squared_loss"]
EXP_SYN_CYCLIC_CONFIGS['is_kernelize'] = 0
EXP_SYN_CYCLIC_CONFIGS['n_samples'] = [10000]
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

opt_list += [{'name': 'EXP_SGD',
            'alpha_t': "CNST",
            'is_sls': False,
            'ada': None}]

EXP_SYN_CYCLIC_CONFIGS['opt_list'] = opt_list

