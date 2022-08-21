
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from scipy import linalg


class Net(nn.Module):
    def __init__(self, p, P1, P2):
        super(Net, self).__init__()
        self.p = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        self.P1 = torch.tensor(P1, dtype=torch.float32)
        self.P2 = torch.tensor(P2, dtype=torch.float32)
        self.one = torch.tensor([1], dtype=torch.float32)

    def forward(self):
        p = torch.cat([self.p, self.one])
        u1_h = torch.matmul(self.P1, p)
        u2_h = torch.matmul(self.P2, p)
        u1 = u1_h[:2] / u1_h[2]
        u2 = u2_h[:2] / u2_h[2]
        return torch.cat([u1, u2])


# Net for optimizing all landmarks together
# class Net(nn.Module):
#     def __init__(self, p, P1, P2):
#         super(Net, self).__init__()
#         self.p = torch.tensor(p.transpose(), dtype=torch.float32, requires_grad=True)
#         self.P1 = torch.tensor(P1, dtype=torch.float32)
#         self.P2 = torch.tensor(P2, dtype=torch.float32)
#         self.one = torch.tensor(np.ones((1, p.shape[0])), dtype=torch.float32)
#
#     def forward(self):
#         p = torch.cat([self.p, self.one])
#         u1_h = torch.matmul(self.P1, p)
#         u2_h = torch.matmul(self.P2, p)
#         u1 = torch.transpose(u1_h[:2, :] / u1_h[2:3, :], 0, 1)
#         u2 = torch.transpose(u2_h[:2, :] / u2_h[2:3, :], 0, 1)
#         return torch.cat([u1, u2])


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def calc_3d_from_2x2d(landmarks1, landmarks2, cal_mat, geometric=True):
    P1 = cal_mat[:, 0].reshape((3, 4))
    P2 = cal_mat[:, 1].reshape((3, 4))
    out_3d = np.zeros((landmarks1.shape[0], 3))
    out_dlt = np.zeros((landmarks1.shape[0], 3))
    for landmark_idx in range(landmarks1.shape[0]):
        u1, v1 = landmarks1[landmark_idx, :]
        u2, v2 = landmarks2[landmark_idx, :]

        if np.isnan(u1) or np.isnan(u2):
            out_3d[landmark_idx, :] = np.nan
            out_dlt[landmark_idx, :] = np.nan
        else:
            out_3d[landmark_idx, :] = DLT(P1, P2, [u1, v1], [u2, v2])
            out_dlt[landmark_idx, :] = DLT(P1, P2, [u1, v1], [u2, v2])
            if geometric:
                net = Net(out_3d[landmark_idx, :], P1, P2)
                net.train()
                criterion = nn.MSELoss()
                optimizer = optim.SGD([net.p], lr=0.00001, momentum=0.9)
                label = torch.Tensor([[u1, v1, u2, v2]])
                for iter in range(30):
                    optimizer.zero_grad()
                    outputs = net()
                    loss = criterion(outputs, label[0])
                    # print(f'landmark #{landmark_idx}: {iter} - {loss}')
                    loss.backward()
                    optimizer.step()
                out_3d[landmark_idx, :] = net.p.cpu().detach().numpy()

    # optimize all landmarks together
    # out_dlt = out_3d.copy()
    # if geometric:
    #     non_nan_inds = ~np.isnan(landmarks1[:, 0])
    #     u1 = landmarks1[non_nan_inds, :]
    #     u2 = landmarks2[non_nan_inds, :]
    #     _out_3d = out_3d[non_nan_inds, :]
    #     net = Net(_out_3d, P1, P2)
    #     net.train()
    #     criterion = nn.MSELoss()
    #     optimizer = optim.SGD([net.p], lr=0.0001, momentum=0.9)
    #     label = torch.cat([torch.Tensor(u1), torch.Tensor(u2)])
    #     for iter in range(40):
    #         optimizer.zero_grad()
    #         outputs = net()
    #         loss = criterion(outputs, label)
    #         # print(f'landmark #{landmark_idx}: {iter} - {loss}')
    #         loss.backward()
    #         optimizer.step()
    #     out_3d[non_nan_inds, :] = net.p.cpu().detach().numpy().transpose()

    return out_3d, out_dlt
