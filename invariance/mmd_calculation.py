import torch

'''adpated from domainbed.algorithms MMD'''
class MMD():
    def __init__(self, gaussian=True):
        if gaussian:
            self.kernel_type = 'gaussian'
        else:
            self.kernel_type = 'mean_cov'

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm) 
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
            # return cova_diff
    
    def mmd_all_domains(self, x, algo=None):
        mmd_val = 0
        with torch.no_grad():
            for i in range(len(x)):
                for j in range(i+1, len(x)):               
                    features1 = x[i]
                    features2 = x[j]
                    if algo is not None:
                        features1 = algo(features1)
                        features2 = algo(features2)
                    mmd_val += self.mmd(features1, features2)
        
        mmd_val = mmd_val / (len(x) * (len(x)-1)/2)
        return mmd_val.item()