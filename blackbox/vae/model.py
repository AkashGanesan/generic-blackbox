from dataclasses import dataclass, field
import typing as ty
import torch
import torch.nn as nn
from ..prims import Exp


@dataclass
class CVAEParams:
    '''
    input
    =====
    conditional :: bool
    cond_size:: int
    enc_config :: [(in_channel, out_channel, activation)]
    dec_config :: [(in_channel, out_channel, activation)]

    '''
    cond_size: int
    enc_config: ty.List[ty.Tuple[int, int, nn.Module]]
    dec_config: ty.List[ty.Tuple[int, int, nn.Module]]

    def __post__init__(self):
        assert(isinstance(self.cond_size, int))
        assert self.cond_size >= 0, "cond_size can be a non-negative integer"
        for i in [self.enc_config, self.dec_config]:
            for (in_channel, out_channel, module_type) in i:
                assert(isinstance(in_channel, int))
                assert(isinstance(out_channel, int))
                assert(issubclass(module_type, nn.Module))

        enc_out_latent_size = self.enc_config[-1][0]
        dec_in_size = self.dec_config[0][0]
        assert(enc_out_latent_size + self.cond_size == dec_in_size)


class CVAE(nn.Module):

    def __init__(self,
                 params: VAEParams):
        super(CVAE, self).__init__()
        self.params = params
        self.encoder = CVAEEncoder(params.enc_config)
        self.decoder = CVAEDecoder(params.dec_config)
        self.latent_size = params.enc_config[0][0] - params.cond_size

    def forward(self, x, c=None):
        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = to_var(torch.randn([batch_size, self.latent_size]))
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, c=None, batch_size=1):
        z = torch.randn([batch_size, self.latent_size])
        recon_x = self.decoder(z, c=None)
        return recon_x


class CVAEEncoder(nn.Module):
    def __init__(self,
                 params: ty.List[ty.Tuple[int, int, nn.Module]]):
        super(CVAEEncoder, self).__init__()

        self.params = params

        self.MLP = nn.Sequential()
        for i, (in_size, out_size, activation) in enumerate(params):
            self.MLP.add_module(name="L%i" % (i),
                                module=nn.Linear(in_size, out_size))
            if activation is not None:
                self.MLP.add_module(name="A%i" % (i),
                                    module=activation)
        self.linear_means = nn.Linear(*params[-1][0:2])
        self.linear_log_var = nn.Linear(*params[-1][0:2])

    def forward(self, x, c):
        x = torch.cat([x, c], dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        logvars = self.linear_log_var(x)
        return means, logvars


class CVAEDecoder(nn.Module):
    def __init__(self,
                 params: ty.List[ty.Tuple[int, int, nn.Module]]):
        super(CVAEDecoder, self).__init__()

        self.params = params

        self.MLP = nn.Sequential()
        for i, (in_size, out_size, activation) in enumerate(params):
            self.MLP.add_module(name="L%i" % (i),
                                module=nn.Linear(in_size, out_size))
            if activation is not None:
                self.MLP.add_module(name="A%i" % (i),
                                    module=activation)

    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat((z, c), dim=-1)
        # print("Dimension of z is", z.size())
        x = self.MLP(z)
        return x


if __name__ == "__main__":
    cond_size = 10

    enc_config = [(784+cond_size, 400, nn.ReLU()),
                  (400, 50,None),
                  (50, 20,nn.ReLU()) ]

    dec_config = [(20 + cond_size, 200, None),
                  (200, 500, nn.ReLU()),
                  (200, 200, nn.Softmax())]

    vae_config = CVAEParams(cond_size=cond_size,
                           enc_config=enc_config,
                           dec_config=dec_config)
    print(vae_config)
    vae_model = CVAE(vae_config)
