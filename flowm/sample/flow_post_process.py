# (optional, when the exact path is not specified) find the flow sample npz file
# load flow samples
# filter the flow samples wrt. given force magnitudes
# (optional, when --reweight-repul arg is not NO_REPUL) when specified, assign weights according to CGnet repulsion setting


from argparse import ArgumentParser
import torch
import numpy as np
import os
from glob import glob

import pytorch_lightning as pl

from ..datasets import FlowMatchingData
from ..nn import CGFlow, get_cg_top, get_feat, get_CGnet_prior
from ..utils import units

def look_up_chkpt_file(possible_path):
    if os.path.isfile(possible_path):
        chkpt_file = possible_path
    else:
        versions = glob(os.path.join(possible_path, "version_*"))
        if len(versions) > 0:
            last_version = max([int(p.split("_")[-1]) for p in versions])
            possible_path = os.path.join(possible_path, f"version_{last_version}")
        if os.path.isdir(os.path.join(possible_path, "checkpoints")):
            possible_path = os.path.join(possible_path, "checkpoints")
        ckpts = glob(os.path.join(possible_path, "epoch*.ckpt.npz"))
        assert len(ckpts) >= 1, f"No sample found at path: `{possible_path}`. Please check input and consider using the absolute path to the npz-format sample file."
        assert len(ckpts) <= 1, f"Multiple sample files found at path: `{possible_path}`. Please check input and consider using the absolute path to the npz-format sample file."
        chkpt_file = ckpts[0]
    return chkpt_file

from tqdm import tqdm

def filter_reweight_ca_chain_dataset(flow_dataset, repul_term, max_force_magintude=None):
    # filter the dataset according to the given threshold
    magnitude = np.square(flow_dataset["forces"]).mean(axis=-1)
    if max_force_magintude is None:
        max_force_magintude = np.max(magnitude)
    mask = magnitude <= max_force_magintude
    print("Samples before/after filtering: %d/%d" % (len(mask),np.sum(mask)))
    output = {k:flow_dataset[k][mask] for k in flow_dataset.keys()}

    if repul_term is not None:
        # reweight the dataset according to given repulsion definition
        all_repul_es = []
        all_repul_fs = []
        num_batches = int(np.ceil(len(flow_dataset["samples"]) / 1024))
        n_atoms = flow_dataset["samples"].shape[-1] // 3
        for i in tqdm(range(num_batches)):
            coords = torch.tensor(flow_dataset["samples"][i*1024:(i+1)*1024].reshape((-1, n_atoms, 3))).cuda()
            es, fs = repul_term.energy_force(coords)
            all_repul_es.append(es.detach().cpu().numpy())
            all_repul_fs.append(fs.detach().cpu().numpy())
        all_repul_es = np.concatenate(all_repul_es)[:, 0]
        all_repul_fs = np.concatenate(all_repul_fs).reshape([-1, n_atoms * 3])
        output["forces"] += all_repul_fs[mask]
        w_0 = np.exp(-all_repul_es) # require es unit: k_BT
        reweight_factor = mask.sum() / w_0[mask].sum()
        w = w_0 * reweight_factor
        output["weights"] = w[mask]
        n_eff = np.square(output["weights"].sum()) / np.square(output["weights"]).sum()
        output["n_eff"] = n_eff
        print(f"Effective sample size after filtering and reweighting: {int(n_eff)}")
    assert output["samples"].shape[-1] % 3 == 0, f"Dataset `samples`' shape {output['samples'].shape} does not correspond to 3D coordinates."
    n_atoms = output["samples"].shape[-1] // 3
    output["samples"] = output["samples"].reshape([-1, n_atoms, 3])
    output["forces"] = output["forces"].reshape([-1, n_atoms, 3])
    return output

def main():
    parser = ArgumentParser()
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser.add_argument("--sample-file-path", type=str, default=None)
    parser.add_argument("--name", type=str, default="prot")
    parser.add_argument("--pdb", type=str)
    parser.add_argument("--max-force-magnitude", type=float, default=8e4)
    parser.add_argument("--reweight-repul", type=str, default="NO_REPUL")
    args = parser.parse_args()
    #print(args)
    
    chkpt_path = args.sample_file_path
    if chkpt_path is None:
        # try to locate the checkpoint folder from the dataset options
        data = FlowMatchingData(**vars(args))
        # only the train_size and cv_fold_describe are required, no need to load the dataset in real
        # data.prepare_data()
        # data.setup()        
        chkpt_path = f"./output/cgflow_{args.name}_{data.train_size}_{data.cv_fold_describe}"

    chkpt_file = look_up_chkpt_file(chkpt_path)
    ori_samples = np.load(chkpt_file)
    if args.reweight_repul != "NO_REPUL":
        cg_top = get_cg_top(args.pdb)
        feat = get_feat(cg_top)
        repul_term = get_CGnet_prior(cg_top, feat, args.reweight_repul, harmonic_stat=None, embed_feat=True).cuda() # a repul-only prior
    else:
        repul_term = None
    output_samples = filter_reweight_ca_chain_dataset(ori_samples, repul_term, args.max_force_magnitude)
    np.savez(chkpt_file[:-4] + "_processed.npz", **output_samples)

    print("All OK!")

if __name__ == "__main__":
    main()


