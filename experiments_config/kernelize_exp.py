EXP_KERNEL_CONFIGS={}
EXP_KERNEL_CONFIGS['max_epoch']=2500
EXP_KERNEL_CONFIGS['runs']=[0,1,2]
EXP_KERNEL_CONFIGS['batch_size']=[-2,-4,-8]
EXP_KERNEL_CONFIGS['benchmarks_list']=["mushrooms", "ijcnn", "rcv1"]
EXP_KERNEL_CONFIGS['losses']=["squared_loss"]
EXP_KERNEL_CONFIGS['is_kernelize']=1
EXP_KERNEL_CONFIGS['regularization_factor']=0
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
                    'alpha_t': 'CNST',
                    'is_sls': False,
                    'ada': None,
                    }]

EXP_KERNEL_CONFIGS['opt_list'] = opt_list