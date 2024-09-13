from haven import haven_utils as hu
import itertools
import numpy as np
from experiments_config.kernelize_exp import *
from experiments_config.syn_interp_exp import *
from experiments_config.syn_vary_n_exp import *
from experiments_config.syn_cyclic_exp import *
from experiments_config.syn_check_alpha_beta import *
from experiments_config.syn_non_interp_exp import *
from experiments_config.syn_non_interp_compare import *
from experiments_config.syn_interp_pan import *
from experiments_config.line_search_exp import *


def get_benchmark(benchmark,
                  opt_list,
                  batch_size=[1, 100, -1],
                  runs=[0, 1, 2, 3, 4],
                  max_epoch=[50],
                  losses=["logistic_loss", "squared_loss", "squared_hinge_loss"],
                  kappa=[100],
                  n_samples=[10000],
                  d=[20],
                  variance=None,
                  is_kernelize=None,
                  regularization_factor=0
                  ):
    if benchmark == "mushrooms":
        return {"dataset": ["mushrooms"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": regularization_factor,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "is_kernelize": is_kernelize}

    elif benchmark == "ijcnn":
        return {"dataset": ["ijcnn"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": regularization_factor,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "is_kernelize": is_kernelize}

    elif benchmark == "a1a":
        return {"dataset": ["a1a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "a2a":
        return {"dataset": ["a2a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 2300,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "w8a":
        return {"dataset": ["w8a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 50000,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "covtype":
        return {"dataset": ["covtype"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 500000,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "phishing":
        return {"dataset": ["phishing"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1e-4,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "rcv1":
        return {"dataset": ["rcv1"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": regularization_factor,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "is_subsample": 1,
                "subsampled_n": 10000,
                "is_kernelize": is_kernelize}

    elif benchmark == "synthetic_interpolation":
        return {"dataset": ["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "margin": [0.1],
                "false_ratio": [0, 0.1, 0.2],
                "n_samples": n_samples,
                "d": [200],
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}
    
    elif benchmark == "synthetic_ls":
        return {"dataset": ["synthetic_ls"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "regularization_factor": 0.,
                "n_samples": n_samples,
                "d": d,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "synthetic_reg":
        return {"dataset": ["synthetic_reg"],
                "loss_func": ["logistic_loss"],
                "opt": opt_list,
                "regularization_factor": 1. / 10000,
                "n_samples": n_samples,
                "d": d,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}
    
    elif benchmark == "synthetic_kappa":
        return {"dataset": ["synthetic_kappa"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "regularization_factor": 0.,
                "n_samples": n_samples,
                "d": d,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "kappa":kappa,
                "variance":variance,
                }
    
    elif benchmark == "synthetic_test":
        return {"dataset": ["synthetic_test"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "regularization_factor": 0.,
                "n_samples": n_samples,
                "d": d,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "kappa":kappa,
                "variance":variance,
                }

    else:
        print("Benchmark unknown")
        return


def get_exp_group(opt_list, benchmarks_list = ["mushrooms", "ijcnn", "rcv1", "synthetic_ls", "synthetic_kappa", "synthetic_test"],
                  max_epoch=1000, n_samples=[10000], d=[20],
                  runs=[0,1,2], batch_size=[1, 100, -1], 
                  losses=["logistic_loss", "squared_loss", "squared_hinge_loss"], 
                  kappa=[100], variance=[0], is_kernelize=0, regularization_factor=0):
    exp_groups = {}
    for benchmark in benchmarks_list:
        exp_groups['exp_%s' % benchmark] = hu.cartesian_exp_group(get_benchmark(benchmark, opt_list,
                                                                             batch_size=batch_size,
                                                                             variance=variance,
                                                                             is_kernelize=is_kernelize,
                                                                             max_epoch=[max_epoch],
                                                                             runs=runs, 
                                                                             kappa=kappa,
                                                                             n_samples=n_samples,
                                                                             d=d,
                                                                             losses=losses,
                                                                             regularization_factor=regularization_factor))
    return exp_groups

EXP_GROUPS = get_exp_group(**EXP_LINE_SEARCH_CONFIGS)