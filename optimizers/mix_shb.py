from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

from optimizers.sls import SLS as SLS

def Mix_SHB(score_list, closure, D, labels,  batch_size=1,max_epoch=100, gamma=None,
    x0=None, mu=1,L=1, is_sls=False, c=0.5, verbose=False, D_test=None, labels_test=None, log_idx=100):
    """
        Mix-SHB for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """

    n = D.shape[0]
    d = D.shape[1]
    if mu>L:
        mu = L
 
    m = int(n/batch_size)

    T=max_epoch
    T0 = int(np.ceil(c * T))
    T1 = T-T0
    

    if x0 is None:
        x = np.zeros(d)
        x0 = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')
    x_prev = x.copy()

    num_grad_evals = 0
    num_func_evals = 0

    loss, full_grad = closure(x, D, labels)

    if verbose:
        output = 'Epoch.: %d, Grad. norm: %.2e' % \
                 (0, np.linalg.norm(full_grad))
        output += ', Func. value: %e' % loss
        # output += ', Step size: %e' % eta
        output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n)
        output += ', Num function evaluations/n: %f' % (num_func_evals / n)
        print(output)

    score_dict = {"itr": 0}
    score_dict["n_func_evals"] = num_func_evals
    score_dict["n_grad_evals"] = num_grad_evals
    score_dict["n_grad_evals_normalized"] = num_grad_evals / n
    score_dict["train_loss"] = loss
    score_dict["grad_norm"] = np.linalg.norm(full_grad)
    score_dict["train_accuracy"] = accuracy(x, D, labels)
    if D_test is not None:
        test_loss = closure(x, D_test, labels_test, backwards=False)
        score_dict["test_loss"] = test_loss
        score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
    

    t=0

    #For stage 1
    eta_0 = 1./(2*L)
    a_k = eta_0
    b_k = (1 - (1/2)*np.sqrt(a_k*mu))**2
    # b_k = (1 - np.sqrt(a_k*mu))**2
    
    #For stage 2
    # alpha=(3*(L/mu)/T1)**(1./T1)
    alpha=(1/T1)**(1./T1)
    gamma = 1./(2*L)
    # gamma = 1e-3
    eta = gamma*alpha
    lrn = gamma*alpha
    lr = lrn
    # lrn=gamma*(alpha**(t))
    
    ldn = ((1.- 2*eta*L)/lrn*mu) * (1 - (1 - lrn*mu)**t)
    ld = ldn

    score_dict["lambda_k"] = ld
    score_dict["alpha_k"] = a_k
    score_dict["beta_k"] = b_k
    score_list += [score_dict]

    for k in range(max_epoch):        
        t_start = time.time()

        if np.linalg.norm(full_grad) <= 1e-12:
            break
        if np.linalg.norm(full_grad) > 1e20:
            break
        if np.isnan(full_grad).any():
            break
        if t >= T:
            break
                   
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
        # for i in range(1):

            # get the minibatch for this iteration
            indices = minibatches[i]
            # indices = minibatches[np.random.randint(m)]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the loss, gradients
            loss, gk = closure(x, Di, labels_i)
            
            if t>T0:
                lr = lrn
                lrn=gamma*(alpha**(t-T0+1))
                ld = ldn
                ldn = ((1.- 2*eta*L)/lrn*mu) * (1 - (1 - lrn*mu)**(t-T0+1))
                a_k = lr/(1 + ldn)
                b_k = ((1 - lr * mu)/(1 + ldn)) * ld
            else:
                a_k = eta_0
                b_k = (1 - np.sqrt(a_k*mu))**2
                
            temp = x.copy()
            x -= a_k * gk - b_k * (x - x_prev)
            x_prev = temp
            num_grad_evals = num_grad_evals + batch_size

            if is_sls:
                gamma,fv=SLS(x,gk,Di,labels_i,gamma,closure,(x_prev,mu,ld,lr,eta,T))
                num_func_evals+=fv
                # num_grad_evals = num_grad_evals + batch_size
                # lr=lr*alpha**(t+1)

            if (num_grad_evals) % log_idx == 0 or (num_grad_evals) % n== 0:
                t_end = time.time()
                # print(ldn)
                loss, full_grad = closure(x, D, labels)

                if verbose:
                    output = 'Epoch.: %d, Grad. norm: %.2e' % \
                             (int(t*batch_size/n), np.linalg .norm(full_grad))
                    output += ', Func. value: %e' % loss
                    output += ', Step size: %e' % eta
                    output += ', L: %e' % L
                    output += ', Num gradient evaluations/%d: %f' % (log_idx,num_grad_evals / log_idx)
                    output += ', Num function evaluations/%d: %f' % (log_idx,num_func_evals / n)
                    print(output)

                score_dict = {"itr": (t+1)}
                score_dict["time"]=t_end-t_start
                score_dict["n_func_evals"] = num_func_evals
                score_dict["n_grad_evals"] = num_grad_evals
                if batch_size==n:
                    score_dict["n_grad_evals_normalized"] = num_grad_evals / n
                else:
                    score_dict["n_grad_evals_normalized"] = num_grad_evals / log_idx

                score_dict["train_loss"] = loss
                score_dict["grad_norm"] = np.linalg.norm(full_grad)
                score_dict["train_accuracy"] = accuracy(x, D, labels)
                score_dict["lambda_k"] = ld
                score_dict["alpha_k"] = a_k
                score_dict["beta_k"] = b_k
                if D_test is not None:
                    test_loss = closure(x, D_test, labels_test, backwards=False)
                    score_dict["test_loss"] = test_loss
                    score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
                score_list += [score_dict]
                if np.linalg.norm(full_grad) <= 1e-12:
                    print(np.linalg.norm(full_grad))
                    break
                if np.linalg.norm(full_grad) > 1e20:
                    print(np.linalg.norm(full_grad))
                    break
                if np.isnan(full_grad).any():
                    print(full_grad)
                    break
                t_start=time.time()
            t += 1
            if t >= T:
                break

    return score_list