# find CGnet model
# load CGnet model
# sample with loaded model via Langevin or parallel tempering dynamics


from argparse import ArgumentParser
import torch
import numpy as np
import os
from glob import glob

import mdtraj as md
import pytorch_lightning as pl

from ..datasets import FlowMatchingData
from ..nn import CGnet
from .sim_utils import sample_initial_coords, Simulation, PTSimulation
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
        ckpts = glob(os.path.join(possible_path, "epoch*.ckpt"))
        assert len(ckpts) >= 1, f"No checkpoint found at path: `{chkpt_path}`. Please check input and consider using the absolute path to the checkpoint file."
        assert len(ckpts) <= 1, f"Multiple checkpoints found at path: `{chkpt_path}`. Please check input and consider using the absolute path to the checkpoint file."
        chkpt_file = ckpts[0]
    return chkpt_file

def main():
    parser = ArgumentParser()
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser.add_argument("--cgnet-chkpt-path", type=str, default=None)
    parser.add_argument("--trj-save-path", type=str, default=None)
    parser.add_argument("--name", type=str, default="prot")
    parser.add_argument("--pdb", type=str)
    parser.add_argument("--initial-coords-path", type=str, default=None)
    parser.add_argument("--temp-in-K", type=float, nargs='+', default=[300.])
    parser.add_argument("--n-indepedent-sims", type=int, default=100)
    parser.add_argument("--n-time-steps", type=int, default=25000)
    parser.add_argument("--time-step-in-ps", type=float, default=2e-3)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--friction-in-inv-ps", type=float, default=1.)
    parser.add_argument("--use-pt", action="store_true")
    parser.add_argument("--pt-exchange-interval", type=int, default=1000)
    parser.add_argument("--pt-output-all-temps", action="store_true")
    args = parser.parse_args()
    #print(args)
    
    chkpt_path = args.cgnet_chkpt_path

    if args.initial_coords_path is not None:
        train_coords = np.load(args.initial_coords_path)
        if args.initial_coords_path.endwith(".npz"):
            train_coords = train_coords["coords"]
        print(f"Using initial coordinates from file `{args.initial_coords_path}`")
    elif args.data_path is not None:
        # try to locate the checkpoint folder from the dataset options
        data = FlowMatchingData(**vars(args))
        data.prepare_data()
        data.setup()
        train_coords = data.cv_dataset.get_train_set(fold_index=data.cv_fold,
                                                     train_size=data.train_size)[0]
        # infer chkpt path
        if chkpt_path is None:
            possible_paths = [f"./output/cgnet_{args.name}_{data.train_size}_{data.cv_fold_describe}",
                              f"./output/flow_cgnet_{args.name}_{data.train_size}_{data.cv_fold_describe}"]
            possible_chkpt_paths = []
            for p in possible_paths:
                possible_chkpt_path = None
                try:
                    possible_chkpt_path = look_up_chkpt_file(p)
                except:
                    print(f"No checkpoints found at path {p}")
                if possible_chkpt_path is not None:
                    possible_chkpt_paths.append(possible_chkpt_paths)
            assert len(possible_chkpt_paths) > 0, "No checkpoint found at possible paths."
            assert len(possible_chkpt_paths) == 1, f"Multiple checkpoints found at possible paths: {possible_chkpt_paths}. \nPlease specify via --chkpt-path."
            chkpt_file = possible_chkpt_paths[0]
    else:
        raise ValueError("Need to specify either --initial-coords-path or --data-path for initial coordinates.")

    chkpt_file = look_up_chkpt_file(chkpt_path)
    cgnet = CGnet.load_from_checkpoint(chkpt_file).model
    print(f"CGnet model has been loaded from `{chkpt_file}`.")
    # print(f"{next(cgnet.parameters()).device}")
    device = torch.device("cuda")
    core = os.path.basename(chkpt_file)[:-5]
    output_name = args.name + "_" + core + ("_pt_sim" if args.use_pt else "_sim") + f"_{args.temp_in_K[0]:.1f}K.npy" 
    trj_save_path = args.trj_save_path
    if trj_save_path is None:
        dirname = os.path.dirname(chkpt_file)
        trj_save_path = os.path.join(dirname, output_name)
        print(f"Trajectory output path not specified, will be set to `{trj_save_path}`")
    if os.path.isdir(trj_save_path):
        trj_save_path = os.path.join(trj_save_path, output_name)
        print(f"Trajectory output path is a directory. Will output to path `{trj_save_path}`.")
    if os.path.isfile(trj_save_path):
        print(f"File exists at desired output path `{trj_save_path}`. Will overwrite.")
    
    beta = [units.inv_temp_mol_per_kcal(t) for t in args.temp_in_K]
    if args.use_pt:
        assert len(beta) > 1, "With --use-pt more than one temperature need to be specified in --temp-in-K"
        is_sorted = all(beta[i] >= beta[i+1] for i in range(len(beta) - 1))
        assert is_sorted, "Temperatures specified in --temp-in-K should be ascending."
    else:
        assert len(beta) == 1, "Without --use-pt only one temperature should be given"
    initial_coords = sample_initial_coords(train_coords, args.n_indepedent_sims, random=True)
    initial_coords = torch.tensor(initial_coords, requires_grad=True).to(device)
    cg_top = md.load_topology(args.pdb)
    mass_conv = {"N": 14.0, "C": 12.0, "O": 16.0}
    masses = [mass_conv[at.element.symbol] for at in cg_top.atoms]
    cgnet = cgnet.to(device)
    cgnet.eval()
    if args.use_pt:
        sim_obj = PTSimulation(cgnet, initial_coords, betas=beta, length=args.n_time_steps,
                               save_interval=args.save_interval, exchange_interval=args.pt_exchange_interval,
                               friction=args.friction_in_inv_ps, masses=np.array(masses),
                               dt=args.time_step_in_ps, device=device,
                               log_interval=100 * args.save_interval, log_type='print')
    else:
        sim_obj = Simulation(cgnet, initial_coords, dt=args.time_step_in_ps, beta=beta[0],
                             friction=args.friction_in_inv_ps, masses=np.array(masses),
                             length=args.n_time_steps, save_interval=args.save_interval,
                             log_interval=100*args.save_interval, log_type="print",
                             device=device)
    traj = sim_obj.simulate()
    if args.use_pt and not args.pt_output_all_temps:
        print(f"Outputing replicas simulated at temperature {args.temp_in_K[0]} K.")
        traj = traj[:args.n_indepedent_sims]
    np.save(trj_save_path, traj)

    print("All OK!")

if __name__ == "__main__":
    main()


