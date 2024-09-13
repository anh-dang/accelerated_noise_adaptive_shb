from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

# conservative sls
def SLS(x,g1,Di,labels_i,gamma,closure,x_prev=None,SHB=None):
  gamma_m=gamma
  L = 1/(2*gamma_m)
  j=0
  f1 = closure(x, Di, labels_i, backwards=False)
  g1_normsq = (np.linalg.norm(g1))**2
  func_val=1
  if SHB != None:
    x_prev,mu,ld,lr,eta,T = SHB
  while j<1000:
    if SHB != None:
      lrn = gamma
      ldn = ((1.- 2*eta*L)/lrn*mu) * (1 - (1 - lrn*mu)**T)
      a_k = lr/(1 + ldn)
      b_k = ((1 - lr * mu)/(1 + ldn)) * ld
      new = a_k * g1 - b_k * (x - x_prev)
      f2 = closure(x-new, Di, labels_i, backwards=False)
    else:
      f2 = closure(x-gamma*g1, Di, labels_i, backwards=False)

    #c=0.5

    if gamma <= (f1-f2)/(0.5*g1_normsq+1e-12):
      break
    j+=1
    gamma=0.9*gamma
  func_val += j
  if j==1000:
      gamma=0.9*gamma_m

  return gamma,func_val

def LOG_SLS(x,g1,Di,labels_i,gamma,closure):
  # gamma_m=gamma
  j=0
  f1 = closure(x, Di, labels_i, backwards=False)
  g1_normsq = (np.linalg.norm(g1))**2
  c_g = (0.5/f1) * g1_normsq
  func_val=1
  while j<1000:
    f2 = closure(x-gamma*g1, Di, labels_i, backwards=False)
    #c=0.5
    if gamma <= (np.log(f1)-np.log(f2))/(c_g+1e-12):
      break
    j+=1
    gamma=0.9*gamma
  func_val += j
  if j==1000:
      gamma=0.9*gamma

  return gamma,func_val