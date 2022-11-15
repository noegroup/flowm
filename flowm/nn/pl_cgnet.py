import numpy as np
import torch
import mdtraj as md
import pytorch_lightning as pl
from .cgnet_utils import *
from ..utils import units

__all__ = ["get_cg_top", "get_feat", "get_CGnet_prior", "get_cgnet_stats", "CGnet"]

def _avail_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_cg_top(cg_top_or_pdb_path):
    if isinstance(cg_top_or_pdb_path, str):
        cg_top = md.load_topology(cg_top_or_pdb_path)
    else:
        cg_top = cg_top_or_pdb_path
    return cg_top

def get_feat(cg_top):
    cg_feat = LinearMolFeaturizer(cg_top)
    return cg_feat

def get_cgnet_stats(
        cg_top,
        ref_coords,   # for fitting the prior parameters
        weights=None, # weights for fitting prior):
    ):
    # fit prior params & zscores over all training data
    cg_feat = get_feat(cg_top).to(_avail_device())
    harmonic_stat = cg_feat.get_harmonic_stat_from_data(ref_coords, weights=weights)
    zscores = cg_feat.get_zscores_from_data(ref_coords, weights=weights)

    return zscores, harmonic_stat

def get_CGnet_prior(cg_top, cg_feat, prior_type, harmonic_stat=None, embed_feat=False):
    # setup prior, when `harmonic_stat` is kept None, then return a repulsion-only prior
    if prior_type == "GLY_SPECIAL_REPUL":
        gly_special = True
        prior_type = CGnetPrior.BOOSTED_AWSEM_REPUL
    else:
        gly_special = False
        try:
            prior_type = getattr(CGnetPrior, prior_type)
        except:
            raise ValueError(f"Desired prior type `{prior_type}` not recognized!")
    if harmonic_stat is not None:
        prior = CGnetPrior(cg_feat, harmonic_stat, prior_type, embed_feat=embed_feat)
    else:
        prior = CGnetRepulPrior(cg_feat, prior_type, embed_feat=embed_feat)
    if gly_special:
        prior.setup_special_gly_repul(cg_top)
    return prior

class CGnet(pl.LightningModule):
    def __init__(
        self,
        cg_top,
        zscores,
        harmonic_stat,
        prior_type="NO_REPUL",
        loss_coeff=1.,
        activation="tanh",
        num_layers=5,
        width=160,
        lipschitz_strength=10.,
        temp=300.,
        lr=1e-3,
        target_lr=1e-5,
        max_epochs=50,
        lr_decay_freq=5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # setup featurizer
        cg_top = get_cg_top(cg_top)
        cg_feat = get_feat(cg_top)

        # setup prior
        prior = get_CGnet_prior(cg_top, cg_feat, prior_type, harmonic_stat=harmonic_stat)

        k_BT = units.k_BT_in_kcal_per_mol(temp)
        cgnet = NewCGnet(cg_feat, zscores, layer_norm=False, batch_norm=False,
                         skip_conn=True, n_layers=num_layers, width=width,
                         activation=torch.nn.SiLU() if activation=="silu" else torch.nn.Tanh(), 
                         k_BT=k_BT,
                         prior_energy_model=prior)
        self.lip_const = lipschitz_strength
        self.model = cgnet
        self.criterion = WeightedFMLoss(loss_coeff)
        self.lr = lr
        self.target_lr = target_lr
        self.num_epochs = max_epochs
        self.decay_freq = lr_decay_freq
        if max_epochs < 2 * lr_decay_freq or lr_decay_freq != 0 and max_epochs % lr_decay_freq != 0:
            raise ValueError("Input `max_epochs` should be at least two times `lr_decay_freq`: please check.")
        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CGnet")
        parser.add_argument("--prior-type", default="NO_REPUL", choices=["NO_REPUL", "DEFAULT_REPUL", "GLY_SPECIAL_REPUL"])
        parser.add_argument("--activation", default="tanh", choices=["tanh", "silu"])
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--width", type=int, default=160)
        parser.add_argument("--lipschitz_strength", type=float, default=10.)
        parser.add_argument("--temp", type=float, default=300.)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--target-lr", type=float, default=1e-5)
        parser.add_argument("--max-epochs", type=int, default=50)
        parser.add_argument("--lr-decay-freq", type=int, default=5)
        return parent_parser        
    
    @staticmethod
    def get_FM_loss(cgnet, batch, criterion):
        coords = batch[0]#.requires_grad_()
        pred_Fs = cgnet.force(coords)
        forces = batch[1]
        if len(batch) > 2:
            weights = batch[2]
        else:
            weights = None
        loss = criterion(pred_Fs, forces, weights)
        return loss
    
    def training_step(self, batch, batch_idx):
        if self.lip_const is not None:
            self.model.lipschitz_projection(self.lip_const) # lip proj
        loss = self.get_FM_loss(self.model, batch, self.criterion)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.get_FM_loss(self.model, batch, self.criterion)
        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        rate_decay = (self.target_lr / self.lr) ** (self.decay_freq / (self.num_epochs - self.decay_freq))
        if self.decay_freq == 0:
            milestones = []
        else:
            milestones = np.arange(self.num_epochs)[self.decay_freq::self.decay_freq]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         milestones=milestones,
                                                         gamma=rate_decay)
        return [optim, ], [scheduler, ]

    
