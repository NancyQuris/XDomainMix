import torch


class FeatureImportance():
    def __init__(self, x, y, gradient, percentage):
        self.x = x
        self.y = y
        self.gradient = gradient
        self.percentage = percentage
        
        self.num_of_x = x.size(0)
        self.x_dimension = x.size(1)
        self.importance_number = int(self.x_dimension * self.percentage)
    
    def get_quantile(self, result):
        return torch.quantile(result, self.percentage)
    
    def random(self):
        result = torch.zeros(self.x.size())
        avg_importance = 0
        for i in range(self.num_of_x):
            selected_dimensions = torch.randperm(self.x_dimension)[:self.importance_number]
            result[i][selected_dimensions] = 1
            avg_importance += torch.count_nonzero(result[i]).item()
        
        result = result.long()
        return result, 1-(avg_importance/(self.num_of_x*self.x.size(1)*self.x.size(2)*self.x.size(3)))

    def gradient_i(self):
        result = torch.ones(self.x.size())
        avg_importance = 0
        for i in range(self.num_of_x):
            current_gradient = self.gradient[i]
            quantile = self.get_quantile(current_gradient)
            important_idx = current_gradient > quantile
            result[i][important_idx] = 0
            avg_importance += torch.count_nonzero(important_idx).item()

        result = result.long()
        return result, avg_importance/(self.num_of_x*self.x.size(1)*self.x.size(2)*self.x.size(3))

    def gradient_norm(self):
        result = torch.ones(self.x.size())
        avg_importance = 0
        for i in range(self.num_of_x):
            current_gradient = torch.abs(self.gradient[i])
            quantile = self.get_quantile(current_gradient)
            important_idx = current_gradient > quantile
            result[i][important_idx] = 0
            avg_importance += torch.count_nonzero(important_idx).item()

        result = result.long()
        return result, avg_importance/(self.num_of_x*self.x.size(1)*self.x.size(2)*self.x.size(3))

    def gradient_and_feature(self):
        result = torch.ones(self.x.size())
        avg_importance = 0
        for i in range(self.num_of_x):
            current_gradient = self.gradient[i] * self.x[i]
            quantile = self.get_quantile(current_gradient)
            important_idx = current_gradient > quantile
            result[i][important_idx] = 0
            avg_importance += torch.count_nonzero(important_idx).item()

        result = result.long()
        return result, avg_importance/(self.num_of_x*self.x.size(1)*self.x.size(2)*self.x.size(3))

    def gradient_and_feature_pool(self):
        result = torch.ones(self.x.size())
        avg_importance = 0
        for i in range(self.num_of_x):
            current_gradient = self.gradient[i] * torch.mean(self.x[i], dim=(1,2), keepdim=True)
            quantile = self.get_quantile(current_gradient)
            important_idx = current_gradient > quantile
            result[i][important_idx] = 0
            avg_importance += torch.count_nonzero(important_idx).item()

        result = result.long()
        return result, avg_importance/(self.num_of_x*self.x.size(1)*self.x.size(2)*self.x.size(3))


