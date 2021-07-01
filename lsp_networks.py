import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Normalizing Flows dependencies
## Conditioned flows following LSP SAC
# https://github.com/haarnoja/sac/blob/master/sac/policies/latent_space_policy.py as a hint for the sampling process and requirements
# https://github.com/haarnoja/sac/blob/master/sac/distributions/real_nvp_bijector.py for the structures of the Conditioned Real NVP Bijector.
class SAC_LSP_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64,64], output_activation=nn.Identity()):
        super().__init__()
        layers = []
        for h0,h1 in zip([input_dim] + hidden_sizes[:-1], hidden_sizes):
            layers.append(nn.Linear(h0,h1))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        layers.append(output_activation)
        self.model = nn.Sequential(*layers)
  
    def forward(self,x):
        return self.model(x)

# Note: this still has some small numerical error when computing x - ( x -> z -> x), order of e-7
class ConditionalAffineTransform(nn.Module):
    def __init__(self, dim, conditional_dim, parity, hidden_sizes=[64,64], device="cpu"):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.conditional_dim = conditional_dim
        self.device = device
        # TODO: cleaner .to(device) scheme ?
        self.log_s_mlp = SAC_LSP_MLP(input_dim=dim+conditional_dim, output_dim=dim, hidden_sizes=hidden_sizes).to(device)
        self.t_mlp = SAC_LSP_MLP(input_dim=dim+conditional_dim, output_dim=dim, hidden_sizes=hidden_sizes).to(device)
        self.scale = nn.Parameter(th.zeros(1), requires_grad=True).to(device)
        self.scale_shift = nn.Parameter(th.zeros(1), requires_grad=True).to(device)
        
        # Masking allows for easier handling of even action dimension.
        self.mask = th.ones([dim]).to(device)
        if not parity == 0:
            self.mask[:(dim//2)] = 0
        else:
            self.mask[(dim//2):] = 0
                
    def forward(self, x, condition, reverse=False):
        masked_x = x * self.mask# Need to use .repeat() ? or expand dims takes care of it already ...
        x_cond = th.cat([masked_x, condition], axis=1)
        log_s, t = self.log_s_mlp(x_cond), self.t_mlp(x_cond)
        # sligktly stabler computation (check: flow from x -> z, then from z -> x, and compute difference)
        log_s = self.scale * th.tanh(log_s) + self.scale_shift
        log_s *= (1. - self.mask)
        t *= (1. - self.mask)
    
        if not reverse:
            x = x * log_s.exp() + t
            logdet = log_s
        else:
            x = (x - t) * log_s.exp().reciprocal() # log_s.neg().exp() more stabler to div by zero like erros
            logdet = (-1 * log_s)
        
        return x, logdet

class ConditionalRealNVP(nn.Module):
    def __init__(self, act_shape, prior=None, n_coupling_layers=4, hidden_sizes=[64,64], device="cpu"):
        super().__init__()
        self.prior = prior
        self.device = device
        if self.prior is None:
            self.prior = Normal(th.zeros([act_shape], device=device), th.ones(act_shape,device=device))
        
        self.flow = nn.ModuleList()
        for n in range(n_coupling_layers):
            # Note: regarding the conditional dim, the original implementation sets it to the double of the action dimension
            self.flow.append(ConditionalAffineTransform(dim=act_shape,
                conditional_dim=act_shape*2, parity=n%2, hidden_sizes=hidden_sizes, device=device))
    
    def forward(self, x, condition):
        z, log_det = x, th.zeros(x.shape).to(self.device)
        for aff_trans in self.flow:
            z, delta_ldet = aff_trans(z, condition)
            log_det += delta_ldet
        prior_logprob = self.prior.log_prob(z)
        return z, prior_logprob, log_det
    
    def backward(self, z, condition):
        x, log_det = z, th.zeros(z.shape).to(self.device)
        for aff_trans in self.flow[::-1]:
            x, delta_ldet = aff_trans(x, condition, reverse=True)
            log_det += delta_ldet
        return x, log_det

    def sample(self, n_samples, condition):
        z = self.prior.sample([n_samples,])
        x, logdet = self.backward(z, condition)
        prior_logprob = self.prior.log_prob(z)
        
        logprobs = prior_logprob + logdet
        return x, logprobs

# Standard SAC LSP and Q Networks
class LSP( nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_sizes=[256,256],
        layer_init = None, n_coupling_layers=2, nmap_action_candidates=100, squashing=True,
        device="cpu", *args, **kwargs):
        super().__init__()
        self.squashing = squashing
        network = []
        for h0,h1 in zip([obs_shape, *hidden_sizes], [*hidden_sizes, act_shape * 2]):
            network.extend([
                nn.Linear(h0,h1), nn.ReLU()
            ])
        network.pop()
        self.network = nn.Sequential(*network)

        self.nmap_action_candidates = nmap_action_candidates # Related to deterministic action sampling

        # NOTE: Original implementation varies the number of MLP layes in the scale function depending on the cmplexity of the environment: 
        # Ref: https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/examples/variants.py#L20
        # Here, we just use [64,64] as default
        self.rnvp = ConditionalRealNVP(act_shape=act_shape,
            n_coupling_layers=n_coupling_layers, device=device) # weird use of device. to fix.
        
        if layer_init is not None:
            self.apply(layer_init)
        
        self.to(device)
    
    def to(self,device):
        super().to(device)
        self.device = device
        return self
    
    def forward(self, x):
        if not isinstance(x, th.Tensor):
            x = th.Tensor(x).to(self.device)
        return self.network(x)
    
    # TODO: Refactor with get_action
    def get_actions( self, observations, latents=None):
        conditions = self.forward(observations)
        if latents is None:
            actions, action_logprobs = self.rnvp.sample(conditions.shape[0], conditions)
        else:
            if not isinstance(latents, th.Tensor):
                latents = th.Tensor(latents).to(self.device)
            assert latents.shape[0] == conditions.shape[0], "latent and observatioon batch_size doese not match"
            actions, log_det = self.rnvp.backward(latents, conditions)
            prior_logprobs = self.rnvp.prior.log_prob(latents)
            action_logprobs = prior_logprobs + log_det # Obtain the logprobs of x, according to the change of variable formula
        
        # Squeeze the actions betweein -1. and 1., correct the logprobs accordingly
        if self.squashing:
            actions = th.tanh(actions)
            action_logprobs -= th.log( 1. - actions.pow(2) + 1e-8)

        return actions, action_logprobs.sum(1)

    def get_action( self, obs, latents=None, deterministic=False):
        with th.no_grad():
            if deterministic:
                raise NotImplementedError
            else:
                if latents is None:
                    actions, _ = self.get_actions([obs])
                else:
                    actions, _ = self.get_actions([obs], [latents])

                return actions.cpu().numpy()[0]

class QFunction( nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_sizes=[256,256],
        layer_init = None, device="cpu", *args, **kwargs):
        super().__init__()
        network = []        
        for h0,h1 in zip([obs_shape + act_shape, *hidden_sizes], [*hidden_sizes, 1]):
            network.extend([
                nn.Linear(h0,h1), nn.ReLU()
            ])
        network.pop()
        self.network = nn.Sequential(*network)

        if layer_init is not None:
            self.apply(layer_init)
        
        self.to(device)

    def to(self,device):
        super().to(device)
        self.device = device
        return self

    def forward( self, x, a):
        if not isinstance(x, th.Tensor):
            x = th.Tensor(x).to(self.device)
        if not isinstance( a, th.Tensor):
            a = th.Tensor(a).to(self.device)
        return self.network(th.cat([x,a],1))

