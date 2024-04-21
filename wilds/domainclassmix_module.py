import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainLearner(nn.Module):
    def __init__(self, feature_dim, num_domain):
        super(DomainLearner, self).__init__()
        self.network = nn.Linear(feature_dim, num_domain)
        print('initialise ResNet Domain learner')

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x 


class DomainClassMixAugmentation(nn.Module):
    '''mixup'''
    def __init__(self, batch_size, num_classes, num_domains, hparams):
        super(DomainClassMixAugmentation, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.hparams = hparams
        self.threshold = hparams["threshold"]
        self.threshold_lower_bound = hparams["threshold_lower_bound"]
        self.threshold_change = hparams["value_to_change"]
        self.step_to_change = hparams["step_to_change"]

        self.uniform = torch.distributions.Uniform(0, 1) 
        self.p = hparams['probability_to_discard'] 

        self.call_num = 0

    def update_threshold(self):
        next_threshold = self.threshold - self.threshold_change
        if self.threshold == self.threshold_lower_bound:
            self.threshold = self.hparams["threshold"]
        elif next_threshold < self.threshold_lower_bound:
            self.threshold = self.threshold_lower_bound
        else:
            self.threshold = next_threshold


    def get_threshold(result, quantile):
        return torch.quantile(result, quantile)
    
    def sample_different_class_different_domain(idx, y, domain, y_target, domain_target):
        y_size = y.size(0)
        for trial in range(y_size*4):
            current_idx = random.randrange(y_size)
            if trial <= y_size*2:
                if (y[current_idx] != y_target) and (domain[current_idx] != domain_target):
                    return current_idx
            else:
                if y[current_idx] != y_target:
                    return current_idx
        return idx 
        
    def sample_same_class_different_domain(idx, y, domain, y_target, domain_target):
        y_size = y.size(0)
        for trial in range(y_size*4):
            current_idx = random.randrange(y_size)
            if trial <= y_size*2:
                if (y[current_idx] == y_target) and (domain[current_idx] != domain_target):
                    return current_idx
            else:
                if (y[current_idx] == y_target) and (current_idx != idx):
                    return current_idx
        return idx 

    def get_feature_decomposition(self, class_im, domain_im, feature):
        class_im =  torch.mean(class_im, dim=(1,2), keepdim=True)
        domain_im =  torch.mean(domain_im, dim=(1,2), keepdim=True)
        class_thr = DomainClassMixAugmentation.get_threshold(class_im, 0.5)
        domain_thr = DomainClassMixAugmentation.get_threshold(domain_im, self.threshold)

        cs_idx = class_im > class_thr
        cg_idx = class_im <= class_thr
        ds_idx = domain_im > domain_thr
        di_idx = domain_im <= domain_thr
        csds_mask = cs_idx * ds_idx
        csdi_mask = cs_idx * di_idx
        cgds_mask = cg_idx * ds_idx
        cgdi_mask = cg_idx * di_idx

        return feature*csds_mask, feature*csdi_mask, feature*cgds_mask, feature*cgdi_mask

    def forward(self, x, y, domain, class_gradient, domain_gradient):
        if (self.call_num % self.step_to_change == 0) and self.call_num != 0:
            self.update_threshold()

        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.uniform.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label, domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[diff_y]
            if prob > self.p:
                result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
            else:
                result[b] = new_cgds + csdi[b] + cgdi[b]

        self.call_num += 1
  
        return result

    def clip_get_feature_decomposition(self, class_im, domain_im, feature):
        class_thr = DomainClassMixAugmentation.get_threshold(class_im, 0.5)
        domain_thr = DomainClassMixAugmentation.get_threshold(domain_im, self.threshold)

        cs_idx = class_im > class_thr
        cg_idx = class_im <= class_thr
        ds_idx = domain_im > domain_thr
        di_idx = domain_im <= domain_thr
        csds_mask = cs_idx * ds_idx
        csdi_mask = cs_idx * di_idx
        cgds_mask = cg_idx * ds_idx
        cgdi_mask = cg_idx * di_idx

        return feature*csds_mask, feature*csdi_mask, feature*cgds_mask, feature*cgdi_mask
    
    def clip_forward(self, x, y, domain, class_gradient, domain_gradient):
        if (self.call_num % self.step_to_change == 0) and self.call_num != 0:
            self.update_threshold()

        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b]
            csds_f, csdi_f, cgds_f, cgdi_f = self.clip_get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label, domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[diff_y]
            if prob > self.p:
                result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
            else:
                result[b] = new_cgds + csdi[b] + cgdi[b]

        self.call_num += 1
  
        return result

    def no_discard(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label, domain_label)
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[diff_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
        
        return result
    
    def same_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[same_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
        
        return result
    
    def same_class_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)

            while True:
                same_y2 = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)
                if same_y2 != same_y:
                    break

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[same_y2]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
        
        return result
    

    def same_domain_x(self, x, y, domain, class_gradient, domain_gradient):
        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)
        csds = torch.zeros(x.size()).to(x.device)
        csdi = torch.zeros(x.size()).to(x.device)
        cgds = torch.zeros(x.size()).to(x.device)
        cgdi = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_cg = class_gradient[b].to(x.device)
            current_dg = domain_gradient[b].to(x.device)
            current_feature = x[b, :, :, :]
            csds_f, csdi_f, cgds_f, cgdi_f = self.get_feature_decomposition(current_cg*current_feature, current_dg*current_feature, current_feature)
            
            csds[b] = csds_f
            csdi[b] = csdi_f
            cgds[b] = cgds_f
            cgdi[b] = cgdi_f
        
        mixup_strength = self.beta.sample((B, 2))
        prob = random.random()
        for b in range(B):
            y_label = y[b]
            domain_label = domain[b]
            
            same_y = DomainClassMixAugmentation.sample_same_class_different_domain(b, y, domain, y_label, domain_label)
            diff_y = DomainClassMixAugmentation.sample_different_class_different_domain(b, y, domain, y_label, domain_label)

            new_csds = mixup_strength[b][0] * csds[b] + (1-mixup_strength[b][0]) * csds[same_y]
            new_cgds = mixup_strength[b][1] * cgds[b] + (1-mixup_strength[b][1]) * cgds[diff_y]

            result[b] = new_csds + new_cgds + csdi[b] + cgdi[b]
        
        return result