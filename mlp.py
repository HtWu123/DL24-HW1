import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # x --> s1(wx+b) --> a1 --> s2 --> y_pred(y_hat)
        # linear1(wx+b)-->f()active-->linear2(wf+b)-->g()active-->y_pred(y_hat)

        s1 = x @ self.parameters['W1'].t() + self.parameters['b1']

        # Using PyTorch's built-in ReLU and sigmoid functions
        if self.f_function == 'identity':
            a1 = s1
        elif self.f_function == 'relu':
            a1 = s1.clamp(min=0)
        elif self.f_function == 'sigmoid':
            a1 = 1 / (1+torch.exp(-s1))


        s2 = a1 @ self.parameters['W2'].t() + self.parameters['b2']

        if self.g_function == 'identity':
            y_hat = s2
        elif self.g_function == 'relu':
            y_hat = s2.clamp(min=0)
        elif self.g_function == 'sigmoid':
            y_hat = 1 / (1+torch.exp(-s2))
        
        self.cache = {"x" : x, "s1" : s1, "a1" : a1, "s2" : s2, "y_hat" : y_hat}

        return y_hat

        
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function\
        # y_hat --> s2 --> a1 --> s1 --> x
        # y_hat --> g()active-->linear2(wf+b)-->f()active-->linear1(wx+b)-->x

    
        if self.g_function == 'identity':
            dJds_2 = dJdy_hat
        elif self.g_function == 'relu':
            dJds_2 = dJdy_hat * (self.cache['s2'] > 0).float()
        elif self.g_function == 'sigmoid': 
            s2_s = 1 / (1+torch.exp(-(self.cache['s2'])))
            dJds_2 = dJdy_hat * s2_s * (1 - s2_s)


        self.grads['dJdW2'] = dJds_2.t() @ self.cache['a1']
        self.grads['dJdb2']=dJds_2.t() @ torch.ones(dJds_2.shape[0])
        

        if self.f_function == 'identity':
            dJda_1 = dJds_2 @ self.parameters['W2']
        elif self.f_function == 'relu':
            dJda_1 = dJds_2 @ self.parameters['W2'] * (self.cache['s1'] > 0).float()
        elif self.f_function == 'sigmoid':
            s1_s = 1 / (1+torch.exp(-(self.cache['s1'])))
            dJda_1 = dJds_2 @ self.parameters['W2'] * s1_s * (1 - s1_s)
           
    
        self.grads['dJdW1'] = dJda_1.T @ self.cache['x']
        self.grads['dJdb1'] = dJda_1.T @ torch.ones(dJda_1.shape[0])
    
    

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    c = y.shape[0] * y.shape[1]
    mse_loss = torch.mean(torch.square(y_hat - y))
    dJdy_hat = 2 * (y_hat - y) / c
    return mse_loss, dJdy_hat

    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    c = y.shape[0] * y.shape[1]
    bce_loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    dJdy_hat = (-(y / y_hat) + ((1 - y) / (1 - y_hat))) / c

    return bce_loss, dJdy_hat

    # return loss, dJdy_hat
