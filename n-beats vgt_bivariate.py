import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import torch
from torch import optim
from torch.nn import functional as F

from nbeats_pytorch.model import NBeatsNet
from trainer_pytorch import save

# from darts.utils.statistics import check_seasonality

from tqdm import trange

warnings.filterwarnings(action='ignore', message='Setting attributes')
torch.manual_seed(6165551651)


# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


def load_data(forecast_length, backcast_length): 
    data = pd.read_csv('data/vgt123.csv', index_col=0, parse_dates=True)
    p = data["Price"]
    lp = np.log(p)  # log
    gp = lp.diff()  # difference
    data['GP'] = gp

#    s = data['Vol']
#    ls = np.log(s)
#    gs = math.exp(ls)
#    data['GS'] = s
    
    data = data.dropna()
    z = data[['GP', 'Vol']]
    # print(z.head())
    
    t, n = z.shape

    names = z.columns

    total_shifts = forecast_length + backcast_length
    for i in range(1, total_shifts):
        for v in names:
            s = z[v].shift(i)
            z = pd.concat([z, s.rename(v+str(i))], axis=1)
            # z[v+str(i)] = z[v].shift(i)
    z = z.dropna()
    print(z.head())
    
        
    y = z.iloc[:, :forecast_length*n].values
    x = z.iloc[:, forecast_length*n:].values
    
    y = y[:, 0:1]  # the first column is the target


    # # data backcast/forecast generation.
    # x, y = [], []
    # for epoch in range(backcast_length, len(milk) - forecast_length):
    #     x.append(milk[epoch - backcast_length:epoch])
    #     y.append(milk[epoch:epoch + forecast_length])
    # x = np.array(x)  # shape 
    # y = np.array(y)  # shape 

    # split train/test.
    c = int(len(x) * 0.8)
    x_train, y_train = x[:c], y[:c]
    x_test, y_test = x[c:], y[c:]
    # print(y_train.shape)

    # normalization.
    x_maximum = np.max(x_train, axis=0)
    x_minimum = np.min(x_train, axis=0)
    x_norm_constant = x_maximum - x_minimum
    x_train = (x_train - x_minimum) / x_norm_constant
    x_test = (x_test - x_minimum) / x_norm_constant
    
    y_maximum = np.max(y_train, axis=0)
    y_minimum = np.min(y_train, axis=0)
    y_norm_constant = y_maximum - y_minimum
    y_train = (y_train - y_minimum) / y_norm_constant
    y_test = (y_test - y_minimum) / y_norm_constant
    # x_train, y_train = x_train / norm_constant, y_train / norm_constant
    # x_test, y_test = x_test / norm_constant, y_test / norm_constant
    
    return x_train, y_train, x_test, y_test, y_norm_constant[0]
    
    
def main():

    forecast_length = 1
    backcast_length = 3 * forecast_length
    batch_size = 650  # greater than 4 for viz
    n = 2  # number of inputs

    x_train, y_train, x_test, y_test, norm_constant = load_data(forecast_length, 
                                                                backcast_length)
    

    backcast_length *= n
    
    # model
    net = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        hidden_layer_units=128,
    )
    optimiser = optim.Adam(lr=0.005, params=net.parameters())
    loss_fun = torch.nn.MSELoss()

    nepoch = 60
    grad_step = 0
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, eta_min=1e-4, T_max=1000)
    for epoch in trange(nepoch):
        # train.
        net.train()
        train_loss = []
        for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
            grad_step += 1
            optimiser.zero_grad()
            _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(net.device))
            # print(forecast.shape, y_train_batch.shape)
            loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(net.device))
            train_loss.append(loss.item())
            loss.backward()
            optimiser.step()
            # scheduler1.step()
        train_loss = np.mean(train_loss)  ** 0.5 * norm_constant

        # test.
        net.eval()
        val_loss = []
        val_step = 0
        forecast_list=[]
        for x_test_batch, y_test_batch in data_generator(x_test, y_test, batch_size):
            with torch.no_grad():
                val_step += 1
                _, forecast = net(torch.tensor(x_test_batch, dtype=torch.float))
                test_loss = F.mse_loss(forecast, torch.tensor(y_test_batch, dtype=torch.float)).to(net.device)
                val_loss.append(test_loss.item())
                
                forecast_list.append(forecast)
                
        test_loss = np.mean(val_loss)  ** 0.5 * norm_constant       
        # p = forecast.detach().numpy()
        if epoch % 10 == 0:
            with torch.no_grad():
                save(net, optimiser, grad_step)
            print(f'epoch = {str(epoch).zfill(4)}, '
                  f'grad_step = {str(grad_step).zfill(6)}, '
                  f'tr_loss (RMSE) = {train_loss:.5f}, '
                  f'te_loss (RMSE) = {test_loss:.5f}')
    return forecast_list, norm_constant


if __name__ == '__main__':
    main()