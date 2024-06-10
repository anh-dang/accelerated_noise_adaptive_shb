EXP_SYN_CHECK_ALPHA_BETA={}
EXP_SYN_CHECK_ALPHA_BETA['max_epoch'] = 5000
EXP_SYN_CHECK_ALPHA_BETA['kappa'] = [32]
EXP_SYN_CHECK_ALPHA_BETA['runs'] = [0,1,2]
EXP_SYN_CHECK_ALPHA_BETA['batch_size'] = [4000]
EXP_SYN_CHECK_ALPHA_BETA['benchmarks_list'] = ["mushrooms","synthetic_kappa"]
EXP_SYN_CHECK_ALPHA_BETA['losses'] = ["squared_loss"]
EXP_SYN_CHECK_ALPHA_BETA['is_kernelize'] = 0
opt_list = []

# SHB

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

opt_list += [{'name': 'EXP_SHB',
            'alpha_t': 'EXP',
            'method': 'SEBBOUH',
            'is_sls': False,
            'mis_spec': 1.0,
            'ada': None,
            'ld': None,
            'ld_sche': None,
            'c': 1
            }]

EXP_SYN_CHECK_ALPHA_BETA['opt_list'] = opt_list