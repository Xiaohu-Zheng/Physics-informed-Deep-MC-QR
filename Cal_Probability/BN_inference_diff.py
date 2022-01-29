import torch
import scipy.io as sio

def series_com_prob(prob_T):
    pr_T = torch.prod(prob_T, dim=0)
    return pr_T

def parallel_com_prob(prob_T):
    pr_F = torch.prod((1 - prob_T), dim=0)
    pr_T = 1 - pr_F
               
    return pr_T

def BN_infer(log_rela, pr_com_T, Evidence=0):
    Pr_temp = torch.zeros(len(log_rela), 2)
    pr_com_T_all = torch.cat((pr_com_T, Pr_temp), dim=0)
    for i in range(len(log_rela)):
        ch_number = log_rela[i][0]
        # Evidence inference
        if Evidence != 0 and ch_number == 77:
            pr_com_T_all[Evidence, :] = 0
        prob_T = pr_com_T_all[log_rela[i][2], :]
        if log_rela[i][1] == 'p':
            pr_T = parallel_com_prob(prob_T)
        elif log_rela[i][1] == 's':
            pr_T = series_com_prob(prob_T)
        
        pr_com_T_all[ch_number] = pr_T
    print(pr_com_T_all[74:77, :])
    
    return pr_com_T_all[-1:]

if __name__=="__main__":
    path = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_prob003_mean.mat'
    data = sio.loadmat(path)
    Pf_F, Pf_T = torch.Tensor(data["Pf_F"]), torch.Tensor(data["Pf_T"])

    prob_T = Pf_T[0:57, :]
    # pr_T = series_com_prob(prob_T)
    # print(pr_T)

    # pr_T = parallel_com_prob(prob_T)
    # print(pr_T)
    # log_rela = [[57, 'p', [0,1,2]], [58, 'p', [3,4,5,6,7]], 
    #             [59, 'p', [8,9,10,11]], [60, 'p', [12, 13, 14]],
    #             [61, 's', [15, 18]], [62, 's', [21,25,26,37,56]],
    #             [63, 'p', [27,28, 54]], [64, 's', [19, 39, 53]],
    #             [65, 's', [16, 46, 48]], [66, 'p', [40, 41,42]],
    #             [67, 's', [47, 55]], [68, 's', [20, 29, 38]],
    #             [69, 's', [17, 30]], [70, 'p', [43, 44, 45]],
    #             [71, 'p', [31, 32, 33, 34, 35, 36]],
    #             [72, 'p', [49, 50, 51, 52]], 
    #             [73, 'p', [22, 23, 24]],
    #             [74, 's', [57, 58, 59, 60, 61, 73]],
    #             [75, 'p', [62, 63, 64, 65, 66]],
    #             [76, 'p', [67, 68, 69, 70, 71, 72]],
    #             [77, 's', [74, 75, 76]]
    #            ]
    log_rela = [[57, 'p', [0,1,2]], [58, 'p', [3,4,5,6,7]], 
                [59, 'p', [8,9,10,11]], [60, 'p', [12, 13, 14]],
                [61, 's', [15, 18]], [62, 's', [21,25,26,37,56]],
                [63, 'p', [27,28, 54]], [64, 's', [19, 39, 53]],
                [65, 's', [16, 46, 48]], [66, 'p', [40, 41,42]],
                [67, 's', [47, 55]], [68, 's', [20, 29, 38]],
                [69, 's', [17, 30]], [70, 'p', [43, 44, 45]],
                [71, 'p', [31, 32, 33, 34, 35, 36]],
                [72, 'p', [49, 50, 51, 52]], 
                [73, 's', [22, 23, 24]],
                [74, 's', [57, 58, 59, 60, 61, 73]],
                [75, 's', [62, 63, 64, 65, 66]],
                [76, 's', [67, 68, 69, 70, 71, 72]],
                [77, 'p', [74, 75, 76]]
               ]
    reliability = BN_infer(log_rela, prob_T, Evidence=76)
    print('lower %.8f, upper %.8f'%(reliability[0, 0].item(), reliability[0, 1].item()))
