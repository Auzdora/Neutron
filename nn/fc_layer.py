import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import *

class Linear(Modules):
    """
        Linear layers (Fully connected layers) for neural network.
    """

    def __init__(self, in_features, out_features, bias=True, device=CPU, init_para='xavier_uniform'):
        """
        :param in_features:
        :param out_features:
        :param bias:
        :param init_para: It could be:
                            'normal', 'xavier_normal', 'uniform', 'xavier_uniform'
                          By default, system will use normal.
        """
        self.core_module = True
        self.in_features = in_features
        self.out_features = out_features
        self.init_para = init_para
        self.weight = Tensor((1, out_features, in_features), device, require_grad=True)
        if bias:
            # TODO: 1 could be expanded if coding for batch
            self.bias = Tensor((out_features, 1), require_grad=True)
        else:
            self.bias = None
        self.reset_parameters()
        super(Linear, self).__init__(self.core_module)

    def forward(self, x):
        pass

    def _get_module_info(self):
        """
            Have to rewrite when you define a brand new layers in 'layers'
        directory.
            It could be used to show the model's parameters in defined way.
            It also has to return its parameters value for further function.
        """
        print('In features dim: {}       Out features dim: {}'.format(self.in_features, self.out_features))
        print("Weight:\n{}".format(self.weight.value))
        print("Bias:\n{}".format(self.bias.value))

    def reset_parameters(self):
        """
            Reset network's parameters. To make network easier to learn.
        :return:
        """
        # TODO: Confirm and pack them into a single file
        if self.init_para == 'normal':
            self.weight.value = np.random.normal(loc=0., scale=1., size=(1, self.out_features, self.in_features))
            if self.bias is None:
                pass
            else:
                self.bias.value = np.random.normal(loc=0., scale=1., size=(self.out_features, 1))
        elif self.init_para == 'xavier_normal':
            std = np.sqrt(2./(self.in_features + self.out_features))
            self.weight.value = np.random.normal(loc=0., scale=std, size=(1, self.out_features, self.in_features))
            if self.bias is None:
                pass
            else:
                self.bias.value = np.random.normal(loc=0., scale=std, size=(self.out_features, 1))
        elif self.init_para == 'uniform':
            _bound = np.sqrt(1. / self.in_features)
            self.weight.value = np.random.uniform(low=-_bound, high=_bound,
                                                  size=(1, self.out_features, self.in_features))
            if self.bias is None:
                pass
            else:
                self.bias.value = np.random.uniform(low=-_bound, high=_bound, size=(self.out_features, 1))
        elif self.init_para == 'xavier_uniform':
            _bound = np.sqrt(6. / self.in_features + self.out_features)
            self.weight.value = np.random.uniform(low=-_bound, high=_bound,
                                                  size=(1, self.out_features, self.in_features))
            if self.bias is None:
                pass
            else:
                self.bias.value = np.random.uniform(low=-_bound, high=_bound, size=(self.out_features, 1))