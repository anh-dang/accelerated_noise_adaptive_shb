# Noise adaptive (accelerated) Stochastic Heavy-Ball Momentum
## Experiments

Run the experiments using the command below:

``
python trainval.py -e $exp_{BENCHMARK} -sb ${SAVEDIR_BASE} -r 1
``

with the placeholders defined as follows.

**{BENCHMARK}**

Defines the dataset and regularization constant for the experiments

- `mushrooms`, `ijcnn`, `rcv1` ,`synthetic_kappa`

**{SAVEDIR_BASE}**

Defines the absolute path to where the results will be saved.
