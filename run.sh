#!/bin/bash

python trainval.py -e 'exp_ijcnn' -sb './outputs/output_ijcnn_sgd_shb_cnst' -r 1

python trainval.py -e 'exp_rcv1' -sb './outputs/output_rcv1_sgd_shb_cnst' -r 1