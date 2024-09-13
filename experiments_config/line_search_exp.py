EXP_LINE_SEARCH_CONFIGS={}
EXP_LINE_SEARCH_CONFIGS['max_epoch'] = 2000
EXP_LINE_SEARCH_CONFIGS['runs'] = [0,1,2]
EXP_LINE_SEARCH_CONFIGS['batch_size'] = [-10/1]
EXP_LINE_SEARCH_CONFIGS['benchmarks_list'] = ["ijcnn"] #["mushrooms", "rcv1", "ijcnn"]
EXP_LINE_SEARCH_CONFIGS['losses'] = ["squared_loss", "logistic_loss"]
EXP_LINE_SEARCH_CONFIGS['is_kernelize'] = 0
opt_list = []

# grid search
for lr in [0.5, 1e-1, 1e-3]:
    opt_list += [{'name': 'EXP_SGD',
                        'alpha_t': 'CNST',
                        'lr': lr,
                        'is_sls': False,
                        'ada': None
                        }]

opt_list += [{'name': 'EXP_SGD',
                      'alpha_t': 'CNST',
                      'lr': 2,
                      'is_sls': 'sls',
                      'ada': None
                      }]

opt_list += [{'name': 'EXP_SGD',
                      'alpha_t': 'CNST',
                      'lr': 2,
                      'is_sls': 'log_sls',
                      'ada': None
                      }]

# opt_list += [{'name': 'EXP_SGD',
#                       'alpha_t': 'CNST',
#                       'lr': 2,
#                       'is_sls': 'polyak',
#                       'ada': None
#                       }]

EXP_LINE_SEARCH_CONFIGS['opt_list'] = opt_list