from dependencies import *

from objectives import *
from datasets import *
from utils import *

from optimizers.exp_step_sgd import *
from optimizers.exp_step_acc_sgd import *
from optimizers.rit_sgd import *
from optimizers.masg import *
from optimizers.shb import *
from optimizers.adam import *
from optimizers.rit_shb import *
from optimizers.mashb import *
from optimizers.mix_shb import *
from optimizers.mshbpan import *

import argparse
import exp_configs

import tqdm
import pandas as pd
import pprint
import math
import itertools
import os, sys
import pylab as plt
# import exp_configs_fullbatch
import time
import numpy as np
import shutil

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj

VERBOSE = False
Lparam={}
def trainval(exp_dict, savedir_base, reset=False):

	# dataset options
	data_dir = './'

	# get experiment directory
	exp_id = hu.hash_dict(exp_dict)
	opt_dict = exp_dict["opt"]
	savedir = os.path.join(savedir_base, exp_id)

	if reset:
		# delete and backup experiment
		hc.delete_experiment(savedir, backup_flag=True)
	
	# create folder and save the experiment dictionary
	os.makedirs(savedir, exist_ok=True)
	hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
	pprint.pprint(exp_dict)
	print('Experiment saved in %s' % savedir)

	score_list_path = os.path.join(savedir, 'score_list.pkl')

	seed = 42 + exp_dict['runs']
	np.random.seed(seed)
	kappa = None
	variance = None
	# default values	
	if "is_subsample" not in exp_dict.keys():
		is_subsample = 0
	else:
		is_subsample = exp_dict["is_subsample"]

	if "is_kernelize" not in exp_dict.keys():
		is_kernelize = 0
	else:
		is_kernelize = exp_dict["is_kernelize"]

	if "false_ratio" not in exp_dict.keys():		
		false_ratio = 0		
	else:
		false_ratio = exp_dict["false_ratio"]
        
	if "standardize" not in exp_dict.keys():		
		standardize = False		
	else:
		standardize = exp_dict["standardize"]
	if "remove_strong_convexity" not in exp_dict.keys():		
		remove_strong_convexity = False		
	else:
		remove_strong_convexity = exp_dict["remove_strong_convexity"]
	
    # load the dataset
	if exp_dict["dataset"] == "synthetic":
		n, d = exp_dict["n_samples"], exp_dict["d"]
		false_ratio = exp_dict["false_ratio"]	   
		margin = exp_dict["margin"]	    
		X, y, X_test, y_test, L, mu = data_load(data_dir, exp_dict["dataset"],n, d, margin,
										 false_ratio, standardize=standardize, remove_strong_convexity=remove_strong_convexity)

	elif exp_dict["dataset"] == "synthetic_ls" or exp_dict["dataset"] == "synthetic_reg":
		n, d = exp_dict["n_samples"], exp_dict["d"]
		X, y, X_test, y_test, L, mu = data_load(data_dir, exp_dict["dataset"], n, d,
										 standardize=standardize, remove_strong_convexity=remove_strong_convexity)
		n = X.shape[0]
	elif exp_dict["dataset"] == "synthetic_kappa":
		variance = exp_dict["variance"]
		n, d, kappa= exp_dict["n_samples"], exp_dict["d"], exp_dict["kappa"]
		X, y, X_test, y_test, L, mu = data_load(data_dir, exp_dict["dataset"], n, d,
										 standardize=standardize, remove_strong_convexity=remove_strong_convexity, kappa=kappa, variance=variance)
		n = X.shape[0]
		# if "c" in opt_dict.keys():
		# 	opt_dict["c"] = 1/((-exp_dict["batch_size"])*kappa)
	elif exp_dict["dataset"] == "synthetic_test":
		variance = exp_dict["variance"]
		n, d, kappa= exp_dict["n_samples"], exp_dict["d"], exp_dict["kappa"]
		X, y, X_test, y_test, L, mu = data_load(data_dir, exp_dict["dataset"], n, d,
										 standardize=standardize, remove_strong_convexity=remove_strong_convexity, kappa=kappa, variance=variance)
		n = X.shape[0]
	else:
		if is_subsample == 1:
			n = exp_dict["subsampled_n"]
		else:	
			n = 0

		if is_kernelize == 1:
			d = n
		else:
			d = 0
		exp_dict["n_samples"] = None
		X, y, X_test, y_test, L, mu = data_load(data_dir, exp_dict["dataset"] , n, d, false_ratio,
										 is_subsample=is_subsample, is_kernelize=is_kernelize, standardize=standardize,
										 remove_strong_convexity=remove_strong_convexity)
		n = X.shape[0]

	if exp_dict["batch_size"]<0:
		exp_dict["batch_size"]=int(n/(-exp_dict["batch_size"]))
	print('batch_size:', exp_dict["batch_size"])

	rb=int(exp_dict["batch_size"]/n)
	
	regularization_factor = exp_dict["regularization_factor"]
	if exp_dict["loss_func"] == "logistic_loss":
		closure = make_closure(logistic_loss, regularization_factor)
	
	elif exp_dict["loss_func"] == "squared_loss":
		closure = make_closure(squared_loss, regularization_factor)

	elif exp_dict["loss_func"] == "squared_hinge_loss":
		closure = make_closure(squared_hinge_loss, regularization_factor)
	elif exp_dict["loss_func"] == "huber_loss":
		closure = make_closure(huber_loss, regularization_factor)
	else:
		print("Not a valid loss")

	# check if score list exists 
	if os.path.exists(score_list_path):
		# resume experiment
		score_list = hu.load_pkl(score_list_path)
		s_epoch = score_list[-1]['itr'] + 1
	else:
		# restart experiment
		score_list = []
		s_epoch = 0

	print('Starting experiment at epoch %d' % (s_epoch))

	Lmax, Lmin = 1,1

	if opt_dict["name"] == "EXP_SHB":
		# if exp_dict["dataset"] == "synthetic_kappa" and (exp_dict["batch_size"]/n < 0.45 or exp_dict["batch_size"]/n > 0.55) and opt_dict['method']=='SEBBOUH':
		# 	return
		score_list = Exp_SHB(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 D=X, labels=y, method=opt_dict['method'],
						 L=Lmax, mu=Lmin*opt_dict['mis_spec'],
						 is_sls=opt_dict['is_sls'],
						 alpha_t=opt_dict['alpha_t'],
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE,
						 ada=opt_dict['ada'],
						 ld=opt_dict['ld'],
						 ld_sche=opt_dict['ld_sche'],
						 c=opt_dict["c"])

	elif opt_dict["name"] == "EXP_SGD":
		# if exp_dict["dataset"] == "synthetic_kappa" and (exp_dict["batch_size"]/n < 0.45 or exp_dict["batch_size"]/n > 0.55):
		# 	return
		score_list = Exp_SGD(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 gamma=opt_dict['lr'], kappa=Lmax/Lmin,
						 D=X, labels=y,
						 is_sls=opt_dict['is_sls'],
						 alpha_t=opt_dict['alpha_t'],
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE,
						 ada=opt_dict['ada'])
		
	elif opt_dict["name"] == "ADAM":
		score_list = ADAM(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 D=X, labels=y,
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE,
						 )
	elif opt_dict["name"] == "Mix_SHB":
		score_list = Mix_SHB(score_list, closure=closure,batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 D=X, labels=y,
						 L=Lmax, mu=Lmin,
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE,
						 c=opt_dict["c"])
	elif opt_dict["name"] == "EXP_ACC_SGD":
		score_list = Exp_ACC_SGD(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 gamma=1./(opt_dict["rho"]*Lmax),
						 D=X, labels=y,
						 L=Lmax,
						 rho=opt_dict["rho"],
						 mu=Lmin,
						 alpha_t=opt_dict['alpha_t'],
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE
						 )
	
	elif opt_dict["name"] == "M_ASHB":
		score_list = M_ASHB(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 D=X, labels=y,
						 L=Lmax,
						 mu=Lmin,
						 c=opt_dict["c"],
						 beta_const=opt_dict["beta_const"],
						 D_test=X_test, labels_test=y_test, verbose=VERBOSE)
	




	else:
		print(opt_dict["name"])
		print('Method does not exist')
		return 1/0
    
	save_pkl(score_list_path, score_list)  

	return score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_jupyter', default=None)
    parser.add_argument('-j', '--run_jobs', default=None)
    parser.add_argument('-m', '--mu_misspec', default=1, type=int)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(exp_list, 
                       savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            # print(exp_dict)
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset)
    print('Finished!')


# exp_mushrooms exp_ijcnn exp_rcv1