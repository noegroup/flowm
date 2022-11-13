import numpy
import torch
import torch.nn as nn
import abc
from .utils import FCLayer, ZscoreLayer, calc_lipschitz_norm

__all__ = ["EnergyModel", "GeneralCGnet", "LegacyCGnet", "NewCGnet"]

class EnergyModel(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, k_BT=1.0):
        super(EnergyModel, self).__init__()
        self._k_BT = k_BT
    
    def set_physical_unit(self, k_BT):
        self._k_BT = k_BT
    
    def get_physical_unit(self):
        return self._k_BT

    @abc.abstractmethod
    def forward(self, pos, **kwargs):
        pass

    def energy_force(self, pos, unit="kBT", **kwargs):
        """`unit`='k_BT': (default) uses k_BT as energy unit. 
        `unit`='physical': scale the energy and forces with the physical conversion factor (e.g., k_BT in kcal/mol). 
        Be careful about the length unit of the `pos`: once chosen, it should remain the same during training and application."""
        if unit not in ("kBT", "physical"):
            raise ValueError("`unit` should be either kBT or physical")
        with torch.set_grad_enabled(True):
            pos.requires_grad_()
            energy = self(pos, **kwargs)
            if unit == "physical":
                energy *= self._k_BT
            forces = -torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
            )[0]
        return energy, forces
    
    def force(self, pos, unit="kBT", **kwargs):
        """`unit`='k_BT': (default) uses k_BT as energy unit. 
        `unit`='physical': scale the energy and forces with the physical conversion factor (e.g., k_BT in kcal/mol). 
        Be careful about the length unit of the `pos`: once chosen, it should remain the same during training and application."""
        return self.energy_force(pos, unit=unit)[1]
    
    def energy(self, pos, unit="kBT", **kwargs):
        """`unit`='k_BT': (default) uses k_BT as energy unit. 
        `unit`='physical': scale the energy and forces with the physical conversion factor (e.g., k_BT in kcal/mol). 
        Be careful about the length unit of the `pos`: once chosen, it should remain the same during training and application."""
        return self.energy_force(pos, unit=unit)[0]

class GeneralCGnet(EnergyModel):
    """Internally we always use k_BT as energy unit. Therefore we will need a temperature property for unit transformation when applying the model in physical world."""
    def __init__(self, featurizer, energy_net,
                 k_BT=1.,
                 prior_energy_model=None):
        super(GeneralCGnet, self).__init__(k_BT)
        self._feat = featurizer # CGnetFeaturize(mdtraj_top)
        self._net = energy_net
        self._prior_energy = prior_energy_model

    def featurize(self, coords):
        return self._feat(coords)
    
    def forward(self, pos, **kwargs):
        """Evaluate energy in unit k_BT."""
        feat = self._feat(pos)
        output = self._net(feat)
        if self._prior_energy is not None:
            output += self._prior_energy(feat)
        return output

class LegacyCGnet(GeneralCGnet):
    """Legacy CGnet with vanilla dense layers.
    Internally we always use k_BT as energy unit. Therefore we will need a temperature property for unit transformation when applying the model in physical world."""
    def __init__(self, featurizer, zscores=None,
                 n_layers=5, width=160, activation=nn.Tanh(),
                 k_BT=1.,
                 prior_energy_model=None):
        net_list = []
        if zscores is not None:
            zscores = torch.as_tensor(zscores)
            net_list.append(ZscoreLayer(zscores))
        net_list += [nn.Linear(featurizer.dim, width), activation]
        for i in range(1, n_layers):
            net_list.append(nn.Linear(width, width))
            net_list.append(activation)
        net_list.append(nn.Linear(width, 1, bias=False))
        super(LegacyCGnet, self).__init__(featurizer, k_BT=k_BT,
                                          energy_net=nn.Sequential(*net_list),
                                          prior_energy_model=prior_energy_model)

    def lipschitz_projection(self, lip_strength=10.0):
        """Check the L2 Lipschitz norm of each linear layer and scale the weight matrix
        when the spectral norm exceeds the given lipschitz strength."""
        for m in self._net.children():
            if isinstance(m, nn.Linear):
                for name, par in m.named_parameters():
                    if name == "weight":
                        norm = calc_lipschitz_norm(par.data)
                        par.data /= max(1., norm / lip_strength)

class NewCGnet(GeneralCGnet):
    """New CGnet with vanilla dense layers and some fancy stuff.
    Internally we always use k_BT as energy unit. Therefore we will need a temperature property for unit transformation when applying the model in physical world."""
    def __init__(self, featurizer, zscores=None,
                 batch_norm=False, layer_norm=True, skip_conn=True,
                 n_layers=5, width=160, activation=nn.Tanh(),
                 k_BT=1.,
                 prior_energy_model=None):
        net_list = []
        if zscores is not None:
            zscores = torch.as_tensor(zscores)
            net_list.append(ZscoreLayer(zscores))
        net_list.append(FCLayer(featurizer.dim, activation,
                                out_width=width, bias=True,
                                batch_norm=batch_norm and zscores is None,
                                layer_norm=layer_norm and zscores is None,
                                skip=False))
        for i in range(1, n_layers):
            net_list.append(FCLayer(width, activation,
                                    bias=True,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    skip=skip_conn))
        net_list.append(nn.Linear(width, 1, bias=False))
        super(NewCGnet, self).__init__(featurizer, k_BT=k_BT,
                                       energy_net=nn.Sequential(*net_list),
                                       prior_energy_model=prior_energy_model)

    def lipschitz_projection(self, lip_strength=10.0):
        """Check the L2 Lipschitz norm of each linear layer and scale the weight matrix
        when the spectral norm exceeds the given lipschitz strength."""
        for m in self._net.children():
            if isinstance(m, nn.Linear):
                for name, par in m.named_parameters():
                    if name == "weight":
                        norm = calc_lipschitz_norm(par.data)
                        par.data /= max(1., norm / lip_strength)
            elif isinstance(m, FCLayer):
                m.lipschitz_projection(lip_strength)

