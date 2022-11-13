import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
import bgflow as bg
from bgflow.utils import as_numpy
import bgmol
from bgmol.systems import FAST_FOLDER_NAMES

__all__ = ["get_flow_marginals", "CGFlow"]


def _get_coords_transf(n_particles):
    # make coordinate transform for a linear CG molecule with n_particles
    topology = bgmol.zmatrix.build_fake_topology(n_particles)
    if isinstance(topology, tuple):
        topology = topology[0]
    assert topology.n_bonds == n_particles - 1
    
    # make coordinate transform
    zmatrix, _ = bgmol.zmatrix.ZMatrixFactory(topology).build_naive()
    coordinate_transform = bg.GlobalInternalCoordinateTransformation(zmatrix)
    return coordinate_transform

def get_flow_marginals(ref_coords, weights=None):
    """Get the marginal distribution range for the whitening layer in CGFlow."""
    n_particles = ref_coords.shape[1]
    coordinate_transform = _get_coords_transf(n_particles)
    # IC marginals
    data = torch.as_tensor(ref_coords).reshape(-1, n_particles*3)
    bonds, angles, torsions, *_ = coordinate_transform.forward(data)
    bonds = as_numpy(bonds)
    angles = as_numpy(angles)
    def weighted_avg_and_std(values, weights):
        """
        Quote: https://stackoverflow.com/a/2415343
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, axis=0, weights=weights, keepdims=True)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, axis=0, weights=weights)
        return average[0], np.sqrt(variance)
    bond_mu, bond_sigma = weighted_avg_and_std(bonds, weights)
    angle_mu, angle_sigma = weighted_avg_and_std(angles, weights)
    marginals = dict(
        bond_lower=bonds.min(axis=0),
        bond_upper=bonds.max(axis=0),
        bond_mu=bond_mu,
        bond_sigma=bond_sigma,
        angle_lower=angles.min(axis=0),
        angle_upper=angles.max(axis=0),
        angle_mu=angle_mu,
        angle_sigma=angle_sigma,
    )
    
    return n_particles, marginals

def weighted_nanmean(xs, ws):
    """Similar to torch.nanmean but with weights. Assuming input are 1D 
    vectors with the same shape and on same device.
    """
    factor = (torch.isnan(xs.detach()).logical_not_() * ws).sum()
    return torch.nansum(xs * ws) / factor

class CGFlow(pl.LightningModule):
    def __init__(
        self,
        n_particles,
        marginals,
        transform="spline",
        hidden=(128, 1024, 128),
        n_torsion_blocks=4,
        lr=1e-3,
        lr_decay=1.0,
        n_splits=4,
        augmented_transform="same",  # or "affine"
        n_bond_bins=2,
        **kwargs
    ):
        """build the flows; requires the pl data object to extract the marginals for defining the whitening layer."""
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.lr_decay = lr_decay

        #self.model = torch.nn.ModuleList()
        self.transform = transform
        
        # make coordinate transform
        coordinate_transform = _get_coords_transf(n_particles)
        shapes = bg.ShapeDictionary.from_coordinate_transform(coordinate_transform, dim_augmented=2)
        ctx = {"device": torch.device("cuda:0") if torch.cuda.is_available else "cpu", "dtype": torch.float32}
        marginals = {key: torch.tensor(value, **ctx) for key, value in marginals.items()}
        
        # create builder
        builder = bg.BoltzmannGeneratorBuilder(shapes, **ctx)
        builder.default_transformer_type = {
            "spline": bg.ConditionalSplineTransformer,
            "smooth": bg.MixtureCDFTransformer
        }[transform]
        builder.default_conditioner_kwargs = {
            "hidden": hidden,
        }
        num_bins = lambda n: {"num_bins": n} if transform == "spline" else {"num_components": n}
        if transform == "smooth":
            builder.default_transformer_kwargs = {"inverse": True}

        if augmented_transform == "affine":
            builder.transformer_type[bg.AUGMENTED] = bg.AffineTransformer
            builder.prior_type[bg.AUGMENTED] = bg.NormalDistribution
        elif augmented_transform == "same":
            pass
        else:
            raise ValueError("augmented_transform has to be 'affine' or 'same'")

        # torsions
        n_torsions = shapes[bg.TORSIONS][0]
        T1, T2 = builder.add_split(bg.TORSIONS, ["T1", "T2"], [n_torsions//2, n_torsions - n_torsions//2] )
        for i in range(n_torsion_blocks):
            builder.add_condition(bg.AUGMENTED, on=[T1, T2], **num_bins(10))
            builder.add_condition(T1, [T2, bg.AUGMENTED], **num_bins(10))
            builder.add_condition(T2, [T1, bg.AUGMENTED], **num_bins(10))
        builder.add_merge([T1, T2], bg.TORSIONS)

        # angles
        n_angles = shapes[bg.ANGLES][0]
        A1, A2 = builder.add_split(bg.ANGLES, ["A1", "A2"], [n_angles//2, n_angles - n_angles//2])
        for i in range(2):
            builder.add_condition(A1, [A2, bg.TORSIONS], **num_bins(5))
            builder.add_condition(A2, [A1, bg.TORSIONS], **num_bins(5))
        builder.add_merge([A1, A2], bg.ANGLES)

        # bonds
        if n_bond_bins == 1:
            builder.add_condition(bg.BONDS, [bg.TORSIONS, bg.ANGLES], num_bins=1, transformer_type=bg.ConditionalSplineTransformer)
        else:
            builder.add_condition(bg.BONDS, [bg.TORSIONS, bg.ANGLES], **num_bins(n_bond_bins))

        # marginal whitening
        ic_marginals = bg.InternalCoordinateMarginals(
            builder.current_dims, ctx,
            **marginals,
            augmented=None if augmented_transform == "affine" else bg.AUGMENTED
        )

        builder.add_map_to_ic_domains(ic_marginals)
        builder.add_map_to_cartesian(coordinate_transform)

        flow = builder.build_generator()
        self.model = flow
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CGFlow")
        parser.add_argument("--transform", default="spline", choices=["spline", "smooth"])
        parser.add_argument("--hidden", type=int, default=(128, 1024, 128), nargs='*')
        parser.add_argument("--n-torsion-blocks", type=int, default=2)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--lr-decay", type=float, default=1.0)
        parser.add_argument("--augmented-transform", default="same", choices=["same", "affine"])
        parser.add_argument("--n-bond-bins", type=int, default=2)
        return parent_parser
    
    @staticmethod
    def eval_nll(model, batch):
        if len(batch) == 2:
            xyz, aug = batch
            nll = model.energy(xyz, aug).nanmean()
        elif len(batch) == 3:
            xyz, aug, weights = batch
            nll = weighted_nanmean(model.energy(xyz, aug).flatten(), weights)
        return nll
        
    
    def training_step(self, batch, batch_idx):
        nll = self.eval_nll(self.model, batch)
        self.log("train_loss", nll)
        return nll
    
    def validation_step(self, batch, batch_idx):
        nll = self.eval_nll(self.model, batch)
        self.log("val_loss", nll)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=self.lr_decay)
        return [optim, ], [scheduler, ]

    def sample(self, n_samples):
        with torch.no_grad():
            xyz, aug = self.model.sample(n_samples)
        energy = self.model.energy(xyz, aug).detach().flatten()
        force = self.model.force(xyz, aug).detach()
        return xyz, energy, force
    
    def sample_to_cpu(self, n_samples, batch_size=256):
        all_xyz = []
        all_forces = []
        all_energies = []
        for i in tqdm(range(n_samples // batch_size)):
            xyz, energy, force = self.sample(batch_size)
            all_xyz.append(xyz.detach().cpu())
            all_energies.append(energy.detach().cpu())
            all_forces.append(force.detach().cpu())
        remaining = n_samples % batch_size
        if remaining > 0:
            xyz, energy, force = self.sample(remaining)
            all_xyz.append(xyz.detach().cpu())
            all_energies.append(energy.detach().cpu())
            all_forces.append(force.detach().cpu())
        all_xyz = torch.cat(all_xyz, dim=0)
        all_energies = torch.cat(all_energies, dim=0)
        all_forces = torch.cat(all_forces, dim=0)
        assert len(all_xyz) == n_samples
        assert len(all_energies) == n_samples
        assert len(all_forces) == n_samples
        return all_xyz, all_energies, all_forces
    
