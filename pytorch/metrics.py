from poutyne import register_batch_metric
import torch
import math

default_eps = 1e-16


def sum_non_batch(x):
    dims = tuple(range(1, len(x.shape)))
    return x.sum(dims)


def mean_non_batch(x):
    dims = tuple(range(1, len(x.shape)))
    return x.mean(dims)


def reduce_non_batch(x, f: str):
    if f == "sum":
        return sum_non_batch(x)
    elif f == "mean":
        return mean_non_batch(x)
    else:
        raise ValueError(f"Function {f} not supported. Options: 'sum', 'mean'.")


def torch_copysign(input,sign):
    sign = torch.sign(sign)
    input_sign = torch.sign(input)
    return input*input_sign*sign

def safe_divide(x, y, eps: float):
    mag = torch.abs(y)
    test = (mag < eps)*1

    test2 = torch_copysign(test, y)
    denom = test2 * eps  + (1-test) * y
    return x / denom

def replace_zero_eps(x, eps):
    x[x.abs() < eps] = eps * torch.sign(x)


def norm2(x, feature_normalize: bool):
    n = sum_non_batch(x ** 2)
    if feature_normalize: normalize_feature(n, x.shape)
    return n


def norm1(x, feature_normalize: bool):
    n = sum_non_batch(x.abs())
    if feature_normalize: normalize_feature(n, x.shape)
    return n


def normalize_feature(e, shape):
    n_dims = sum(shape[1:])
    e /= n_dims


def squared_error(x, y, feature_normalize: bool):
    return norm2(x - y, feature_normalize)


def absolute_error(x, y, feature_normalize: bool):
    return norm1(x - y, feature_normalize)


# Root Mean Squared Error (RMSE)
@register_batch_metric('mse2')
def mse(y_pred, y_true, feature_normalize=False):
    se = squared_error(y_pred, y_true, feature_normalize)
    return torch.mean(se)


# Root Mean Squared Error (RMSE)
@register_batch_metric('rmse')
def rmse(y_pred, y_true, feature_normalize=False):
    return torch.sqrt(mse(y_pred, y_true, feature_normalize=feature_normalize))

# Mean Normalized Root Mean Squared Error (MNRMSE)
# https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
@register_batch_metric('mnrmse')
def nrmse(y_pred, y_true, feature_normalize=False):
    r = rmse(y_pred,y_true,feature_normalize=feature_normalize)
    rnorm2 = torch.sqrt(norm2(y_true,feature_normalize))
    return r / torch.mean(rnorm2)

# Range Normalized Root Mean Squared Error (RNRMSE)
# https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
@register_batch_metric('rnrmse')
def nrmse(y_pred, y_true, feature_normalize=False):
    r = rmse(y_pred,y_true,feature_normalize=feature_normalize)
    rnorm2=torch.sqrt(norm2(y_true, feature_normalize))
    rnorm2_range = rnorm2.max()-rnorm2.min()
    return r / rnorm2_range


# Mean Absolute Error (MAE)
@register_batch_metric('mae')
def mae(y_pred, y_true, feature_normalize=False):
    ae = absolute_error(y_pred, y_true, feature_normalize)
    return torch.mean(ae)


# https://www.sciencedirect.com/science/article/pii/S0024379501005729
# Relative Squared Error (RSE)
@register_batch_metric('rse')
def rse(y_pred, y_true, feature_normalize=False, eps=default_eps):
    se = mse(y_pred, y_true, feature_normalize=feature_normalize)
    scale = mse(y_true, y_true.mean(0, keepdim=True), feature_normalize=feature_normalize)
    return safe_divide(se, scale, eps)


# https://www.sciencedirect.com/science/article/pii/S0024379501005729
# Relative Absolute Error (RAE)
@register_batch_metric('rae')
def rae(y_pred, y_true, feature_normalize=False, eps=default_eps):
    se = mae(y_pred, y_true, feature_normalize=feature_normalize)
    scale = mae(y_true, y_true.mean(0, keepdim=True), feature_normalize=feature_normalize)
    return safe_divide(se, scale, eps)


# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
# Mean Absolute Percentage Error (MAPE)
@register_batch_metric('mape')
def mape(y_pred, y_true, feature_normalize=False, eps=default_eps):
    ae = absolute_error(y_pred, y_true, feature_normalize)
    scale = y_true.abs()
    replace_zero_eps(scale, eps)
    return torch.mean(ae / scale)


# Armstrong1985
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# Adjusted Mean Absolute Percentage Error (AMAPE)
@register_batch_metric('amape')
def smape(y_pred, y_true, feature_normalize=False, eps=default_eps):
    ae = absolute_error(y_pred, y_true, feature_normalize=feature_normalize)
    scale = norm1(y_pred + y_true, feature_normalize)
    replace_zero_eps(scale, eps)
    return torch.mean(ae / scale)


# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# Symmetric Mean Absolute Percentage Error (SMAPE)
@register_batch_metric('smape')
def smape(y_pred, y_true, feature_normalize=False, eps=default_eps):
    ae = absolute_error(y_pred, y_true, feature_normalize=feature_normalize)
    scale = norm1(y_pred, feature_normalize) + norm1(y_true,feature_normalize)
    return torch.mean(ae/scale)
