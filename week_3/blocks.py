import numpy as np

#######################################################
# put `w2_sigmoid_forward` and `w2_sigmoid_grad_input` here #
#######################################################
def w2_sigmoid_forward(x_input):
    return 1 / (1 + np.exp(-x_input))

def w2_sigmoid_grad_input(x_input, grad_output):
    sig = 1 / (1 + np.exp(-x_input))
    return grad_output * sig * (1 - sig)
#######################################################
# put `w2_nll_forward` and `w2_nll_grad_input` here    #
#######################################################
def w2_nll_forward(target_pred, target_true):
    eps = 1e-15
    p = np.clip(target_pred, eps, 1 - eps)
    loss = -(target_true * np.log(p) + (1 - target_true) * np.log(1 - p))
    return loss.mean()

def w2_nll_grad_input(target_pred, target_true):
    eps = 1e-15
    p = np.clip(target_pred, eps, 1 - eps)
    N = target_true.shape[0]
    return ((1 - target_true) / (1 - p) - target_true / p) / N