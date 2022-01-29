import torch
import scipy.io as sio

def series_com_prob(prob_F, prob_T):
    com_num = prob_F.size(0)
    if com_num == 1:
        pr_F = prob_F
        pr_T = prob_T
    elif com_num == 2:
        P2_lo = prob_F[1][0] + prob_T[1][0]
        P2_up = prob_F[1][1] + prob_T[1][1]
        P2 = torch.cat((P2_lo.view(1, 1), P2_up.view(1, 1)), dim=1)
        pr_F = prob_F[0] * P2 + prob_T[0] * prob_F[1]
        pr_T = torch.prod(prob_T, dim=0)
    elif com_num == 3:
        P2_lo = prob_F[1][0] + prob_T[1][0]
        P2_up = prob_F[1][1] + prob_T[1][1]
        P2 = torch.cat((P2_lo.view(1, 1), P2_up.view(1, 1)), dim=1)
        P3_lo = prob_F[2][0] + prob_T[2][0]
        P3_up = prob_F[2][1] + prob_T[2][1]
        P3 = torch.cat((P3_lo.view(1, 1), P3_up.view(1, 1)), dim=1)
        pr_F = prob_F[0] * P3 * P2 + prob_T[0] * prob_F[1] * P3 \
               + prob_T[0] * prob_T[1] * prob_F[2]
        pr_T = torch.prod(prob_T, dim=0)
    else:
        for i in range(com_num-1):
            P_i_lo = prob_F[i+1][0] + prob_T[i+1][0]
            P_i_up = prob_F[i+1][1] + prob_T[i+1][1]
            P_i = torch.cat((P_i_lo.view(1, 1), P_i_up.view(1, 1)), dim=1)
            if i == 0:
                Pk = P_i
            else:
                Pk = torch.cat((Pk, P_i), dim=0)
        prob_temp = 0
        for j in range(com_num-3):
            prob_temp += prob_F[j+2] * torch.prod(Pk[(j+3):(com_num-1), :], dim=0) * \
                         torch.prod(prob_T[1:(j+2), :], dim=0)
        A_se_1 = prob_F[1] * torch.prod(Pk[1:(com_num-1), :], dim=0) + prob_temp + \
                 prob_F[com_num-1] * torch.prod(prob_T[1:(com_num-1)], dim=0)
        pr_F = prob_F[0] * torch.prod(Pk, dim=0) + \
               prob_T[0] * A_se_1
        pr_T = torch.prod(prob_T, dim=0)
    return pr_F, pr_T

def parallel_com_prob(prob_F, prob_T):
    com_num = prob_F.size(0)
    if com_num == 1:
        pr_F = prob_F
        pr_T = prob_T
    elif com_num == 2:
        pr_F = torch.prod(prob_F, dim=0)
        # # P2_lo = prob_F[1][0] + prob_T[1][1]
        # # P2_up = prob_F[1][1] + prob_T[1][0]
        # P2_lo = prob_F[1][0] + prob_T[1][0]
        # P2_up = prob_F[1][1] + prob_T[1][1]
        # P2 = torch.cat((P2_lo.view(1, 1), P2_up.view(1, 1)), dim=1)
        # pr_T = prob_T[0] * P2 + prob_F[0] * prob_T[1]
        pr_T = prob_T[0] + prob_F[0] * prob_T[1]
    elif com_num == 3:
        pr_F = torch.prod(prob_F, dim=0)
        # P2_lo = prob_F[1][0] + prob_T[1][0]
        # P2_up = prob_F[1][1] + prob_T[1][1]
        # P2 = torch.cat((P2_lo.view(1, 1), P2_up.view(1, 1)), dim=1)
        # P3_lo = prob_F[2][0] + prob_T[2][0]
        # P3_up = prob_F[2][1] + prob_T[2][1]
        # P3 = torch.cat((P3_lo.view(1, 1), P3_up.view(1, 1)), dim=1)
        # pr_T = prob_F[0] * prob_F[1] * prob_T[2] + prob_F[0] * prob_T[1] * P3 \
        #        + prob_T[0] * P3 * P2 
        pr_T = prob_F[0] * prob_F[1] * prob_T[2] + prob_F[0] * prob_T[1] + prob_T[0]
    else:
        pr_F = torch.prod(prob_F, dim=0)
        for i in range(com_num-1):
            P_i_lo = prob_F[i+1][0] + prob_T[i+1][0]
            P_i_up = prob_F[i+1][1] + prob_T[i+1][1]
            P_i = torch.cat((P_i_lo.view(1, 1), P_i_up.view(1, 1)), dim=1)
            if i == 0:
                Pk = P_i
            else:
                Pk = torch.cat((Pk, P_i), dim=0)
        prob_temp = 0
        for j in range(com_num-3):
            prob_temp += prob_T[(com_num-1)-(j+1)] * torch.prod(Pk[((com_num-1)-(j+1)):(com_num-1), :], dim=0) * \
                         torch.prod(prob_F[1:((com_num-1)-(j+1)), :], dim=0)
        A_pa_1 = prob_T[com_num-1] * torch.prod(prob_F[1:(com_num-1), :], dim=0) + prob_temp + \
                 prob_T[1] * torch.prod(Pk[1:(com_num-1)], dim=0)
        pr_T = prob_F[0] * A_pa_1 + prob_T[0] * torch.prod(Pk, dim=0)
               
    return pr_F, pr_T

if __name__=="__main__":
    path = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_prob.mat'
    data = sio.loadmat(path)
    Pf_F, Pf_T = torch.Tensor(data["Pf_F"]), torch.Tensor(data["Pf_T"])

    prob_F = Pf_F[0:2, :]
    prob_T = Pf_T[0:2, :]
    pr_F, pr_T = series_com_prob(prob_F, prob_T)
    print(pr_F, pr_T)

    pr_F, pr_T = parallel_com_prob(prob_F, prob_T)
    print(pr_F, pr_T)

    # w = (1 - ) / (pr_T(1)-1)

