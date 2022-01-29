import torch

def MC_QR_prediction_for_regression(num, x, net, tau='all'):
    ones = torch.ones_like(x).to(x.device)
    zeros = torch.zeros_like(x).to(x.device)
    mask_mat = torch.where(x==0, zeros, ones)

    with torch.no_grad():
        for i in range(num):
            if tau == 'all':
                ax = x.size(0)
                taus = torch.rand(ax, 1, 1, 1).to(device=x.device)
                taus_mat = mask_mat * taus
            else:
                taus_mat = mask_mat * tau
            
            x_taus = torch.cat((x, taus_mat), dim=1)

            outputs = net(x_taus)
            # ax1, ax2 = outputs.shape
            # outputs = outputs.reshape(ax1, 1, ax2)
            if i==0:
                all_prediction = outputs
            else:
                all_prediction = torch.cat((all_prediction, outputs), 1)
    std, mean = torch.std_mean(all_prediction, dim=1)

    return mean, std


def Supervised_MC_QR_prediction_for_regression(num, x, net, tau='all'):
    mask_mat = torch.ones_like(x).to(x.device)

    with torch.no_grad():
        for i in range(num):
            if tau == 'all':
                ax = x.size(0)
                taus = torch.rand(ax, 1, 1, 1).to(device=x.device)
                taus_mat = mask_mat * taus
            else:
                taus_mat = mask_mat * tau
            
            x_taus = torch.cat((x, taus_mat), dim=1)

            outputs = net(x_taus)
            # ax1, ax2 = outputs.shape
            # outputs = outputs.reshape(ax1, 1, ax2)
            if i==0:
                all_prediction = outputs
            else:
                all_prediction = torch.cat((all_prediction, outputs), 1)
    std, mean = torch.std_mean(all_prediction, dim=1)

    return mean, std