import torch
import torch.nn as nn
from dataclasses import dataclass, field
import typing as ty
import numpy as np


@dataclass
class CNNCVAEConfig:
    latent_size: int
    input_size: ty.Tuple
    num_classes: int
    enc_config: ty.List
    dec_config: ty.List
    conditional: bool

    def __post_init__(self):
        if self.conditional:
            self.enc_config[0][1]['in_channels'] += self.num_classes

        self.enc_config[-1][1]['out_features'] = self.latent_size
        if self.conditional:
            self.dec_config[0][1]['in_features'] = self.latent_size + self.num_classes
        else:
            self.dec_config[0][1]['in_features'] = self.latent_size
        self.dec_config[0][1]['out_features'] = np.prod(self.input_size)



def to_onehot(labels, num_classes, device):

    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot


class CNNCVAE(nn.Module):
    def __init__(self,
                 params):
        super(CNNCVAE, self).__init__()
        self.params = params
        self.encoder = CNNCVAEEncoder(params)
        self.decoder = CNNCVAEDecoder(params)
        self.latent_size = params.latent_size
        self.input_size = params.input_size
        self.num_classes = params.num_classes


    def forward(self, x, c=None):
        cbar = c  # Contains the original cond variable
        if c is not None:
            onehot_targets = to_onehot(c, self.num_classes, x.device)

            cbar = onehot_targets.clone()
            print("size of onehottargest", cbar.shape)
            onehot_targets = onehot_targets.view(-1, self.num_classes, 1, 1)
            c = torch.ones(x.size()[0],
                           self.num_classes,
                           x.size()[2],
                           x.size()[3],
                           dtype=x.dtype).to(x.device)
            c = c * onehot_targets

        x = torch.cat((x, c), dim=1)
        print("size of x1", x.size())

        means, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var)
        batch_size = x.size(0)
        eps = torch.randn([batch_size, self.latent_size]).to(x.device)
        z = eps * std + means
        print("Size of z is 11", z.shape)
        recon_x = self.decoder(z, cbar)

        return recon_x, means, log_var, z

    def inference(self, c=None, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = torch.randn([batch_size, self.latent_size]).to(device)
        recon_x = self.decoder(z)
        return recon_x


def get_output_size(model, input_size, num_classes=1):
    c, w, h = input_size
    if num_classes > 1:
        c += num_classes
    model_device = next(model.parameters()).device
    x = torch.randn(2, c, w, h).to(model_device)
    x =  model(x)
    print("Shape of x is", x.shape)
    return x.size()
    


class CNNCVAEEncoder(nn.Module):
    def __init__(self, params):
        super(CNNCVAEEncoder, self).__init__()

        self.encoder = nn.Sequential()
        for i, (nnType, parameters) in enumerate(params.enc_config[:-1]):
            self.encoder.add_module(name="L%i" % (i),
                                    module=nnType(**parameters))

        final_layer, final_layer_config = params.enc_config[-1]

        if final_layer_config['in_features'] == 0:
            _, C, W, H = get_output_size(self.encoder, params.input_size, params.num_classes)
            print ("c, w, h", C*W*H)
            final_layer_config['in_features'] = C * W * H

        self.linear_means = final_layer(**final_layer_config)
        self.linear_log_var = final_layer(**final_layer_config)

    def forward(self, x):
        # if c is not None:
        #     x = torch.cat([x, c], dim=-1)
        x = self.encoder(x)
        B, C, W, H = x.size()
        x = x.view(B, C * W * H)
        print("Shape of x 11", x.shape)
        print("Shape of x is", x.size())
        means = self.linear_log_var(x)
        logvars = self.linear_log_var(x)
        return means, logvars


class CNNCVAEDecoder(nn.Module):

    def __init__(self, params):
        super(CNNCVAEDecoder, self).__init__()


        self.input_size = params.input_size
        first_layer, first_layer_config = params.dec_config[0]
        self.first_lin_layer = first_layer(**first_layer_config)
        self.decoder = nn.Sequential()
        for i, (nnType, parameters) in enumerate(params.dec_config[1:]):
            self.decoder.add_module(name="L%i" % (i),
                                    module=nnType(**parameters))
            
    def forward(self, z, c):
        if c is not None:
            print("size of c and z", c.shape, z.shape)
            z = torch.cat((z, c), dim=-1)
        z = self.first_lin_layer(z)
        B, _, = z.size()
        z = z.view(B, *self.input_size)
        x = self.decoder(z)
        return x

@dataclass
class CNNCVAEConifg:
    enc_config: ty.List
    dec_config: ty.List
    num_classes: int
    input_size: ty.Tuple
    latent_size: int

    def __post_init__(self):
        self.enc_config[-1][1]['out_features'] = self.latent_size
        self.dec_config[0][1]['out_features'] = self.latent_size


enc_config = [(nn.Conv2d, {'in_channels': 1,
                           'out_channels': 32,
                           'kernel_size': 5,
                           'stride': 1}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 32}),
              (nn.Conv2d, {'in_channels': 32,
                           'out_channels': 32,
                           'kernel_size': 4,
                           'stride': 2}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 32}),
              (nn.Conv2d, {'in_channels': 32,
                           'out_channels': 64,
                           'kernel_size': 3,
                           'stride': 2}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 64}),
              (nn.Conv2d, {'in_channels': 64,
                           'out_channels': 16,
                           'kernel_size': 5,
                           'stride': 1}),
              (nn.BatchNorm2d, {'num_features' : 16}),

              (nn.Linear, {'in_features': 0,
                           'out_features': 0})]



dec_config = [(nn.Linear, {'in_features': 0,
                           'out_features': 0}),

              (nn.Conv2d, {'in_channels': 1,
                           'out_channels': 32,
                           'kernel_size': 3,
                           'stride': 1,
                           'padding': 1}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 32}),

              (nn.Conv2d, {'in_channels': 32,
                           'out_channels': 16,
                           'kernel_size': 5,
                           'stride': 1,
                           'padding': 2}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 16}),

              (nn.Conv2d, {'in_channels': 16,
                           'out_channels': 16,
                           'kernel_size': 5,
                           'stride': 1,
                           'padding': 2}),
              (nn.ELU, {}),
              (nn.BatchNorm2d, {'num_features' : 16}),

              (nn.Conv2d, {'in_channels': 16,
                           'out_channels': 1,
                           'kernel_size': 3,
                           'stride': 1,
                           'padding': 1}),

              (nn.Sigmoid, {})]

dummyCNNCVAEConfig = CNNCVAEConfig(latent_size=8,
                                   input_size=(1, 28, 28),
                                   num_classes=10,
                                   conditional=True,
                                   enc_config=enc_config,
                                   dec_config=dec_config)
