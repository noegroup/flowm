# Modified by Yaoyi Chen from the cgnet package
# Adding support for parallel tempering
# Loosely based on the code from noegroup/reform project
# Revision: fixed bugs in the velocity scaling (Apr 12, 2021)
# --- original copyright ---
# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm, Andreas Kraemer

import torch
import numpy as np

import os
import time
import warnings

# from cgnet.feature import SchnetFeature

__all__ = ["sample_initial_coords", "Simulation", "PTSimulation"]

### for cg molecular simulation with an energy-based model (can be a cgnet or a flow)
def sample_initial_coords(coords, n=1000, random=True):
    # sampling n initial structures from original_dataset
    if random:
        selected = np.random.choice(len(coords), n, replace=False)
    else:
        selected = np.arange(n)
    initial_coords = coords[selected]
    print("Produced {} initial coordinates.".format(len(initial_coords)))
    return initial_coords


class Simulation():
    """Simulate an artificial trajectory from a CGnet.

    If friction and masses are provided, Langevin dynamics are used (see also
    Notes). The scheme chosen is BAOA(F)B, where
        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
        F = force calculation (i.e., from the cgnet)

    Where we have chosen the following implementation so as to only calculate
    forces once per timestep:
        F = - grad( U(X_t) )
        [BB] V_(t+1) = V_t + dt * F/m
        [A] X_(t+1/2) = X_t + V * dt/2
        [O] V_(t+1) = V_(t+1) * vscale + dW_t * noisescale
        [A] X_(t+1) = X_(t+1/2) + V * dt/2

    Where:
        vscale = exp(-friction * dt)
        noisecale = sqrt(1 - vscale * vscale)

    The diffusion constant D can be back-calculated using the Einstein relation
        D = 1 / (beta * friction)

    Initial velocities are set to zero with Gaussian noise.

    If friction is None, this indicates Langevin dynamics with *infinite*
    friction, and the system evolves according to overdamped Langevin
    dynamics (i.e., Brownian dynamics) according to the following stochastic
    differential equation:

        dX_t = - grad( U( X_t ) ) * D * dt + sqrt( 2 * D * dt / beta ) * dW_t

    for coordinates X_t at time t, potential energy U, diffusion D,
    thermodynamic inverse temperature beta, time step dt, and stochastic Weiner
    process W. The choice of Langevin dynamics is made because CG systems
    possess no explicit solvent, and so Brownian-like collisions must be
    modeled indirectly using a stochastic term.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        Trained model used to generate simulation data
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Units are determined
        by the frame striding of the original training data simulation
    beta : float (default=1.0)
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
    friction : float (default=None)
        If None, overdamped Langevin dynamics are used (this is equivalent to
        "infinite" friction). If a float is given, Langevin dynamics are
        utilized with this (finite) friction value (sometimes referred to as
        gamma)
    masses : list of floats (default=None)
        Only relevant if friction is not None and (therefore) Langevin dynamics
        are used. In that case, masses must be a list of floats where the float
        at mass index i corresponds to the ith CG bead.
    diffusion : float (default=1.0)
        The constant diffusion parameter D for overdamped Langevin dynamics
        *only*. By default, the diffusion is set to unity and is absorbed into
        the dt argument. However, users may specify separate diffusion and dt
        parameters in the case that they have some estimate of the CG
        diffusion
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.

    Notes
    -----
    Long simulation lengths may take a significant amount of time.

    Langevin dynamics simulation velocities are currently initialized from
    zero. You should probably remove the beginning part of the simulation.

    Any output files will not be overwritten; the presence of existing files
    will cause an error to be raised.

    Langevin dynamics code based on:
    https://github.com/choderalab/openmmtools/blob/master/openmmtools/integrators.py

    Checks are broken into two methods: one (_input_model_checks()) that deals
    with checking components of the input models and their architectures and another
    (_input_option_checks()) that deals with checking options pertaining to the
    simulation, such as logging or saving/exporting options. This is done for the
    following reasons:

		1) If cgnet.network.Simluation is subclassed for multiple or specific
        model types, the _input_model_checks() method can be overridden without
        repeating a lot of code. As an example, see
        cgnet.network.MultiModelSimulation, which uses all of the same
        simulation options as cgnet.network.Simulation, but overides
        _input_model_checks() in order to perform the same model checks as
        cgnet.network.Simulation but for more than one input model.

		2) If cgnet.network.Simulation is subclassed for different simulation
        schemes with possibly different/additional simulation
        parameters/options, the _input_option_checks() method can be overriden
        without repeating code related to model checks. For example, one might
        need a simulation scheme that includes an external force that is
        decoupled from the forces predicted by the model.

    """

    def __init__(self, model, initial_coordinates, dt=5e-4,
                 beta=1.0, friction=None, masses=None, diffusion=1.0,
                 save_forces=False, save_potential=False, length=100,
                 save_interval=10, random_seed=None,
                 device=torch.device('cpu'),
                 export_interval=None, log_interval=None,
                 log_type='write', filename=None):

        self.initial_coordinates = initial_coordinates
        # self.embeddings = embeddings

        # Here, we check the model mode ('train' vs 'eval')
        self._input_model_checks(model)
        self.model = model

        self.friction = friction
        # for unit consistency: when using kcal/mol as energy unit
        # and g/mol as mass unit, there will be this factor.
        self.masses = masses / 4.184

        self.n_sims = self.initial_coordinates.shape[0]
        self.n_beads = self.initial_coordinates.shape[1]
        self.n_dims = self.initial_coordinates.shape[2]

        self.save_forces = save_forces
        self.save_potential = save_potential
        self.length = length
        self.save_interval = save_interval

        self.dt = dt
        self.diffusion = diffusion
        self.beta = beta

        self.device = device
        self.export_interval = export_interval
        self.log_interval = log_interval

        if log_type not in ['print', 'write']:
            raise ValueError(
                "log_type can be either 'print' or 'write'"
            )
        self.log_type = log_type
        self.filename = filename

        # Here, we check to make sure input options for the simulation 
        # are acceptable. Note that these checks are separated from
        # the input model checks in _input_model_checks() for ease in
        # subclassing. See class notes for more information.
        self._input_option_checks()

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed

        self._simulated = False


    def _input_model_checks(self, model, idx=None):
        """Method to  perform the following checks:
        - warn if the input model is in 'train' mode.
          This does not prevent the simulation from running.
        - Checks to see if model has a SchnetFeature if if no
          embeddings are specified

        Notes
        -----
        This method is meant to check options only related to model settings
        (such as being in 'train' or 'eval' mode) and/or model architectures
        (such as the presence or absence of certain layers). For checking
        options pertaining to simulation details, saving/output settings,
        and/or log settings, see the _input_option_checks() method.
        """

        # This condition accounts for MultiModelSimulation iterating through
        # _input_model_checks instead of running it directly. It's placed
        # here to avoid repeating code, but if the simulation utilities
        # are expanded it might make sense to remove this condition
        # and rewrite _input_model_checks for each class
        if type(model) is list:
            pass

        else:
            idx = '' # just for ease of printing
            if model.training:
                warnings.warn('model {} is in training mode, and certain '
                              'PyTorch layers, such as BatchNorm1d, behave '
                              'differently in training mode in ways that can '
                              'negatively bias simulations. We recommend that '
                              'you put the model into inference mode by '
                              'calling `model.eval()`.'.format(idx))


    def _input_option_checks(self):
        """Method to catch any problems before starting a simulation:
        - Make sure the save_interval evenly divides the simulation length
        - Checks shapes of starting coordinates and embeddings
        - Ensures masses are provided if friction is not None
        - Warns if diffusion is specified but won't be used
        - Checks compatibility of arguments to save and log
        - Sets up saving parameters for numpy and log files, if relevant

        Notes
        -----
        This method is meant to check the acceptability/compatibility of
        options pertaining to simulation details, saving/output settings,
        and/or log settings. For checks related to model structures/architectures
        and input compatibilities (such as using embeddings in models
        with SchnetFeatures), see the _input_model_checks() method.

        """

        # make sure save interval is a factor of total length
        if self.length % self.save_interval != 0:
            raise ValueError(
                'The save_interval must be a factor of the simulation length'
            )


        # make sure initial coordinates are in the proper format
        if len(self.initial_coordinates.shape) != 3:
            raise ValueError(
                'initial_coordinates shape must be [frames, beads, dimensions]'
            )

        # set up initial coordinates
        if type(self.initial_coordinates) is not torch.Tensor:
            initial_coordinates = torch.tensor(self.initial_coordinates)

        self._initial_x = self.initial_coordinates.detach().requires_grad_(
            True).to(self.device)

        # set up simulation parameters
        if self.friction is not None:  # langevin
            if self.masses is None:
                raise RuntimeError(
                    'if friction is not None, masses must be given'
                )
            if len(self.masses) != self.initial_coordinates.shape[1]:
                raise ValueError(
                    'mass list length must be number of CG beads'
                )
            self.masses = torch.tensor(self.masses, dtype=torch.float32
                                       ).to(self.device)

            self.vscale = np.exp(-self.dt * self.friction)
            self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

            self.kinetic_energies = []

            if self.diffusion != 1:
                warnings.warn(
                    "Diffusion other than 1. was provided, but since friction "
                    "and masses were given, Langevin dynamics will be used "
                    "which do not incorporate this diffusion parameter"
                )

        else:  # Brownian dynamics
            self._dtau = self.diffusion * self.dt

            self.kinetic_energies = None

            if self.masses is not None:
                warnings.warn(
                    "Masses were provided, but will not be used since "
                    "friction is None (i.e., infinte)."
                )

        # everything below has to do with saving logs/numpys

        # check whether a directory is specified if any saving is done
        if self.export_interval is not None and self.filename is None:
            raise RuntimeError(
                "Must specify filename if export_interval isn't None"
            )
        if self.log_interval is not None:
            if self.log_type == 'write' and self.filename is None:
                raise RuntimeError(
                    "Must specify filename if log_interval isn't None and log_type=='write'"
                )

        # saving numpys
        if self.export_interval is not None:
            if self.length // self.export_interval >= 1000:
                raise ValueError(
                    "Simulation saving is not implemented if more than 1000 files will be generated"
                )

            if os.path.isfile("{}_coords_000.npy".format(self.filename)):
                raise ValueError(
                    "{} already exists; choose a different filename.".format(
                        "{}_coords_000.npy".format(self.filename))
                )

            if self.export_interval is not None:
                if self.export_interval % self.save_interval != 0:
                    raise ValueError(
                        "Numpy saving must occur at a multiple of save_interval"
                    )
                self._npy_file_index = 0
                self._npy_starting_index = 0

        # logging
        if self.log_interval is not None:
            if self.log_interval % self.save_interval != 0:
                raise ValueError(
                    "Logging must occur at a multiple of save_interval"
                )

            if self.log_type == 'write':
                self._log_file = self.filename + '_log.txt'

                if os.path.isfile(self._log_file):
                    raise ValueError(
                        "{} already exists; choose a different filename.".format(
                            self._log_file)
                    )

    def _set_up_simulation(self, overwrite):
        """Method to initialize helpful objects for simulation later
        """
        if self._simulated and not overwrite:
            raise RuntimeError('Simulation results are already populated. '
                               'To rerun, set overwrite=True.')

        self._save_size = int(self.length/self.save_interval)

        self.simulated_coords = torch.zeros((self._save_size, self.n_sims, self.n_beads,
                                             self.n_dims))
        if self.save_forces:
            self.simulated_forces = torch.zeros((self._save_size, self.n_sims,
                                                 self.n_beads, self.n_dims))
        else:
            self.simulated_forces = None

        # the if saved, the simulated potential shape is identified in the first
        # simulation time point in self._save_timepoint
        self.simulated_potential = None

        if self.friction is not None:
            self.kinetic_energies = torch.zeros((self._save_size, self.n_sims))

    def _timestep(self, x_old, v_old, forces):
        """Shell method for routing to either Langevin or overdamped Langevin
        dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None or torch.Tensor
            None if overdamped Langevin; velocities before propagation
            otherwise
        forces: torch.Tensor
            forces at x_old
        """
        if self.friction is None:
            assert v_old is None
            return self._overdamped_timestep(x_old, v_old, forces)
        else:
            return self._langevin_timestep(x_old, v_old, forces)

    def _langevin_timestep(self, x_old, v_old, forces):
        """Heavy lifter for Langevin dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : torch.Tensor
            velocities before propagation
        forces: torch.Tensor
            forces at x_old
        """

        # BB (velocity update); uses whole timestep
        v_new = v_old + self.dt * forces / self.masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt / 2.

        # O (noise)
        noise = torch.sqrt(1. / self.beta / self.masses[:, None])
        noise = noise * torch.randn(size=x_new.size(),
                                    generator=self.rng).to(self.device)
        v_new = v_new * self.vscale
        v_new = v_new + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt / 2.

        return x_new, v_new

    def _overdamped_timestep(self, x_old, v_old, forces):
        """Heavy lifter for overdamped Langevin (Brownian) dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None
            Placeholder
        forces: torch.Tensor
            forces at x_old
        """
        noise = torch.randn(size=x_old.size(),
                            generator=self.rng).to(self.device)
        x_new = (x_old.detach() + forces*self._dtau +
                 np.sqrt(2*self._dtau/self.beta)*noise)
        return x_new, None

    def _save_timepoint(self, x_new, v_new, forces, potential, t):
        """Utilities to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy

        Parameters
        ----------
        x_new : torch.Tensor
            current coordinates
        v_new : None or torch.Tensor
            current velocities, if Langevin dynamics are used
        forces: torch.Tensor
            current forces
        potential : torch.Tensor
            current potential
        t : int
            Timestep iteration index
        """
        save_ind = t // self.save_interval

        self.simulated_coords[save_ind, :, :] = x_new
        if self.save_forces:
            self.simulated_forces[save_ind, :, :] = forces

        if self.save_potential:
            # The potential will look different for different network
            # structures, so determine its dimensionality at the first
            # timepoint (as opposed to in self._set_up_simulation)
            if self.simulated_potential is None:
                assert potential.shape[0] == self.n_sims
                potential_dims = ([self._save_size, self.n_sims] +
                                  [potential.shape[j]
                                   for j in range(1,
                                                  len(potential.shape))])
                self.simulated_potential = torch.zeros((potential_dims))

            self.simulated_potential[t//self.save_interval] = potential

        if v_new is not None:
            kes = 0.5 * torch.sum(torch.sum(self.masses[:, None]*v_new**2,
                                            dim=2), dim=1)
            self.kinetic_energies[save_ind, :] = kes

    def _log_progress(self, iter_):
        """Utility to print log statement or write it to an text file"""
        printstring = '{}/{} time points saved ({})'.format(
            iter_, self.length // self.save_interval, time.asctime())

        if self.log_type == 'print':
            print(printstring)

        elif self.log_type == 'write':
            printstring += '\n'
            file = open(self._log_file, 'a')
            file.write(printstring)
            file.close()

    def _get_numpy_count(self):
        """Returns a string 000-999 for appending to numpy file outputs"""
        if self._npy_file_index < 10:
            return '00{}'.format(self._npy_file_index)
        elif self._npy_file_index < 100:
            return '0{}'.format(self._npy_file_index)
        else:
            return '{}'.format(self._npy_file_index)

    def _save_numpy(self, iter_):
        """Utility to save numpy arrays"""
        key = self._get_numpy_count()

        coords_to_export = self.simulated_coords[self._npy_starting_index:iter_]
        coords_to_export = self._swap_and_export(coords_to_export)
        np.save("{}_coords_{}.npy".format(
            self.filename, key), coords_to_export)

        if self.save_forces:
            forces_to_export = self.simulated_forces[self._npy_starting_index:iter_]
            forces_to_export = self._swap_and_export(forces_to_export)
            np.save("{}_forces_{}.npy".format(
                self.filename, key), forces_to_export)

        if self.save_potential:
            potentials_to_export = self.simulated_potential[self._npy_starting_index:iter_]
            potentials_to_export = self._swap_and_export(potentials_to_export)
            np.save("{}_potential_{}.npy".format(
                self.filename, key), potentials_to_export)

        if self.friction is not None:
            kinetic_energies_to_export = self.kinetic_energies[self._npy_starting_index:iter_]
            kinetic_energies_to_export = self._swap_and_export(
                kinetic_energies_to_export)
            np.save("{}_kineticenergy_{}.npy".format(self.filename, key),
                    kinetic_energies_to_export)

        self._npy_starting_index = iter_
        self._npy_file_index += 1

    def _swap_and_export(self, data, axis1=0, axis2=1):
        """Helper method to exchange the zeroth and first axes of tensors that
        will be output or exported as numpy arrays

        Parameters
        ----------
        data : torch.Tensor
            Tensor to perform the axis swtich upon. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        axis1 : int (default=0)
            Zero-based index of the first axis to swap
        axis2 : int (default=1)
            Zero-based index of the second axis to swap

        Returns
        -------
        swapped_data : torch.Tensor
            Axes-swapped tensor. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        """
        axes = list(range(len(data.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = data.permute(*axes)
        return swapped_data.cpu().detach().numpy()


    def calculate_potential_and_forces(self, x_old):
        """Method to calculated predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates from the previous timestep

        Returns
        -------
        potential : torch.Tensor
            scalar potential predicted by the model
        forces : torch.Tensor
            vector forces predicted by the model

        Notes
        -----
        This method has been isolated from the main simulation update scheme
        for ease in overriding in a subclass that may preprocess or modify the
        forces or potential returned from a model. For example, see
        cgnet.network.MultiModelSimulation, where this method is overridden
        in order to average forces and potentials over more than one model.
        """
        # potential, forces = self.model(x_old, self.embeddings)
        potential, forces = self.model.energy_force(x_old, unit="physical")
        return potential, forces


    def simulate(self, overwrite=False):
        """Generates independent simulations.

        Parameters
        ----------
        overwrite : Bool (default=False)
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords : np.ndarray
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
            calculated for each frame in the simulation at the
            save_interval
        simulated_potential : np.ndarray or None
            Shape [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation at the save_interval
        kinetic_energies : np.ndarray or None
            If friction is not None, stores the kinetic energy calculated
            for each frame in the simulation at the save_interval

        """
        self._set_up_simulation(overwrite)

        if self.log_interval is not None:
            printstring = "Generating {} simulations of length {} saved at {}-step intervals ({})".format(
                self.n_sims, self.length, self.save_interval, time.asctime())
            if self.log_type == 'print':
                print(printstring)

            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        x_old = self._initial_x

        # for each simulation step
        if self.friction is None:
            v_old = None
        else:
            # initialize velocities at zero
            v_old = torch.tensor(np.zeros(x_old.shape),
                                 dtype=torch.float32).to(self.device)
            # v_old = v_old + torch.randn(size=v_old.size(),
            #                             generator=self.rng).to(self.device)

        for t in range(self.length):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(x_old)
            potential = potential.detach()
            forces = forces.detach()

            # step forward in time
            x_new, v_new = self._timestep(x_old, v_old, forces)

            # save to arrays if relevant
            if (t+1) % self.save_interval == 0:
                self._save_timepoint(x_new, v_new, forces, potential, t)

                # save numpys if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self._save_numpy((t+1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self._log_progress((t+1) // self.save_interval)

            # prepare for next timestep
            x_old = x_new.detach().requires_grad_(True).to(self.device)
            v_old = v_new

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t+1) % self.export_interval > 0:
                self._save_numpy(t+1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            printstring = 'Done simulating ({})'.format(time.asctime())
            if self.log_type == 'print':
                print(printstring)
            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(
            self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(
                self.simulated_forces)

        if self.save_potential:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential)

        if self.friction is not None:
            self.kinetic_energies = self._swap_and_export(
                self.kinetic_energies)

        self._simulated = True

        return self.simulated_coords


class PTSimulation(Simulation):
    """Simulate an artificial trajectory from a CGnet with parallel tempering.
    For thoeretical details on (overdamped) Langevin integration schemes,
    see help(cgnet.network.Simulation).
    For thoeretical details on replica exchange/parallel tempering, see 
    https://github.com/noegroup/reform.
    Note that currently we only implement parallel tempering for Langevin 
    dynamics, so please make sure you provide correct parameters for that.
    Be aware that the output will contain information (e.g., coordinates) 
    for all replicas. 

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        Trained model used to generate simulation data
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Unit should be ps.
    betas : list of floats (default=[1.0])
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
        This parameter will determine how many replicas will be simulated and
        at which temeperatures.
    friction : float (default=None)
        Please provide a float friction value for Langevin dynamics.
    masses : list of floats (default=None)
        Must be a list of floats where the float at mass index i corresponds 
        to the ith CG bead. (Note: please divide it by 418.4 if you are using
        force unit kcal/mol/A and time .)
    diffusion : float (default=1.0)
        Not used for Langevin dynamics
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    exchange_interval : int (default=100)
        The interval at which we attempt to exchange the replicas at different 
        temperatures. Must be a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.

    Notes
    -----
    For notes from the original code, see help(cgnet.network.Simulation).

    """

    def __init__(self, model, initial_coordinates, betas=[1.0], 
                 exchange_interval=100, **kwargs):
        # the basic idea is to reuse all routines in the original code to
        # perform (n_replicas * n_sims) independent simulations of the same
        # system and to attempt replica exchange among corresponding replicas
        # at the given time interval.
        # Example:
        # input betas: [1.0, 0.68, 0.5]
        # input initial_coordinates: [conf_0, conf_1] (each conf_i has shape
        #                                              [n_beads, n_dims])
        # actual internal n_sims = 3 * 2 = 6
        # betas for simulation: [1.0, 1.0, 0.68, 0.68, 0.5, 0.5]
        # actual internal initial_coordinates = [conf_0, conf_1, conf_0,
        #                                        conf_1, conf_0, conf_1]

        # make sure the simulation will be in Langevin scheme
        if kwargs.get("friction") is None:  # overdamped langevin
            raise ValueError('Please provide a valid friction value, '
                             'since we current only support Langevin '
                             'dynamics.')

        # checking customized inputs
        betas = np.array(betas)
        if len(betas.shape) != 1 or betas.shape[0] <= 1:
            raise ValueError('betas must have shape (n_replicas,), where '
                             'n_replicas > 1.')
        self._betas = betas
        if type(exchange_interval) is not int or exchange_interval < 0:
            raise ValueError('exchange_interval must be a positive integer.')
        self.exchange_interval = exchange_interval

        # identify number of replicas
        self.n_replicas = len(self._betas)

        # preparing initial coordinates for each replica
        if type(initial_coordinates) is torch.Tensor:
            initial_coordinates = initial_coordinates.detach().cpu().numpy()
        new_initial_coordinates = np.concatenate([initial_coordinates] * 
                                                 self.n_replicas)

        # original initialization (note that we don't use self.beta any more)
        super(PTSimulation, self).__init__(model,
                                           torch.tensor(new_initial_coordinates),
                                           **kwargs)

        # set up betas for simulation
        self._n_indep = len(initial_coordinates)
        self._betas_x = np.repeat(self._betas, self._n_indep)
        self._betas_for_simulation = torch.tensor(self._betas_x[:, None, None],
                                                  dtype=torch.float32
                                                 ).to(self.device)
        # for replica exchange pair proposing
        self._propose_even_pairs = True
        even_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[:-1:2]]
        odd_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        pair_a = []
        pair_b = []
        for pair in even_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._even_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]
        pair_a = []
        pair_b = []
        for pair in odd_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._odd_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]


    def get_replica_info(self, replica_num=0):
        if type(replica_num) is not int or replica_num < 0 or \
           replica_num >= self.n_replicas:
            raise ValueError('Please provide a valid replica number.')
        indices = np.arange(replica_num * self._n_indep,
                            (replica_num + 1) * self._n_indep)
        return {"beta": self._betas[replica_num], 
                "indices_in_the_output": indices}


    def _langevin_timestep(self, x_old, v_old, forces):
        """Heavy lifter for Langevin dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : torch.Tensor
            velocities before propagation
        forces: torch.Tensor
            forces at x_old
        """

        # BB (velocity update); uses whole timestep
        v_new = v_old + self.dt * forces / self.masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt / 2.

        # O (noise)
        noise = torch.sqrt(1. / self._betas_for_simulation / self.masses[:, None])
        noise = noise * torch.randn(size=x_new.size(),
                                    generator=self.rng).to(self.device)
        v_new = v_new * self.vscale
        v_new = v_new + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt / 2.

        return x_new, v_new


    def _overdamped_timestep(self, x_old, v_old, forces):
        """Heavy lifter for overdamped Langevin (Brownian) dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None
            Placeholder
        forces: torch.Tensor
            forces at x_old
        """
        raise NotImplementedError()


    def _get_proposed_pairs(self):
        """Proposes the even and odd pairs alternatively."""
        if self._propose_even_pairs:
            self._propose_even_pairs = False
            return self._even_pairs
        else:
            self._propose_even_pairs = True
            return self._odd_pairs


    def _detect_exchange(self, potentials):
        """Proposes and checks pairs to be exchanged for parallel tempering.
        Modified from `reform`."""
        pair_a, pair_b = self._get_proposed_pairs()
        
        u_a, u_b = potentials[pair_a], potentials[pair_b]
        betas_a, betas_b = self._betas_x[pair_a], self._betas_x[pair_b]
        p_pair = np.exp((u_a - u_b) * (betas_a - betas_b))
        approved = np.random.rand(len(p_pair)) < p_pair
        #print("Exchange rate: %.2f%%" % (approved.sum() / len(pair_a) * 100.))
        self._replica_exchange_attempts += len(pair_a)
        self._replica_exchange_approved += approved.sum()
        pairs_for_exchange = {"a": pair_a[approved], "b": pair_b[approved]}
        return pairs_for_exchange

    def _get_velo_scaling_factors(self, indices_old, indices_new):
        """Velocity scaling factor for Langevin simulation: 
        \sqrt(t_new/t_old)
        """
        return torch.sqrt(self._betas_for_simulation[indices_old] /
                          self._betas_for_simulation[indices_new])

    def _perform_exchange(self, pairs_for_exchange, xs, vs):
        """Exchanges the coordinates and  given pairs"""
        pair_a, pair_b = pairs_for_exchange["a"], pairs_for_exchange["b"]
        # exchange the coordinates
        x_changed = xs.detach().clone()
        x_changed[pair_a, :, :] = xs[pair_b, :, :]
        x_changed[pair_b, :, :] = xs[pair_a, :, :]
        # scale and exchange the velocities
        v_changed = vs.detach().clone()
        v_changed[pair_a, :, :] = vs[pair_b, :, :] * \
                               self._get_velo_scaling_factors(pair_b, pair_a)
        v_changed[pair_b, :, :] = vs[pair_a, :, :] * \
                               self._get_velo_scaling_factors(pair_a, pair_b)
        return x_changed, v_changed

    def simulate(self, overwrite=False):
        """Generates independent simulations.

        Parameters
        ----------
        overwrite : Bool (default=False)
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords : np.ndarray
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
            calculated for each frame in the simulation at the
            save_interval
        simulated_potential : np.ndarray or None
            Shape [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation at the save_interval
        kinetic_energies : np.ndarray or None
            If friction is not None, stores the kinetic energy calculated
            for each frame in the simulation at the save_interval

        """
        self._set_up_simulation(overwrite)

        # counters for replica exchange
        self._replica_exchange_attempts = 0
        self._replica_exchange_approved = 0

        if self.log_interval is not None:
            printstring = ("Generating {} sets of independent parallel-"
                           "tempering simulations at {} different temperatures"
                           " of length {} saved at {}-step intervals ({})"
                          ).format(self._n_indep, self.n_replicas, 
                                   self.length, self.save_interval,
                                   time.asctime())
            printstring += ("\nThere will be {} = {} * {} trajectories "
                            "recorded.").format(self.n_sims, self._n_indep, 
                                                self.n_replicas)
            if self.log_type == 'print':
                print(printstring)

            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        x_old = self._initial_x

        # for each simulation step
        if self.friction is None:
            v_old = None
        else:
            # initialize velocities at zero
            v_old = torch.tensor(np.zeros(x_old.shape),
                                 dtype=torch.float32).to(self.device)
            # v_old = v_old + torch.randn(size=v_old.size(),
            #                             generator=self.rng).to(self.device)

        for t in range(self.length):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(x_old)
            potential = potential.detach()
            forces = forces.detach()

            # step forward in time
            with torch.no_grad():
                x_new, v_new = self._timestep(x_old, v_old, forces)

            # save to arrays if relevant
            if (t+1) % self.save_interval == 0:
                self._save_timepoint(x_new, v_new, forces, potential, t)

                # save numpys if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self._save_numpy((t+1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self._log_progress((t+1) // self.save_interval)

            # !!! attempt to exchange !!!
            if (t+1) % self.exchange_interval == 0:
                # get potentials
                x_new = x_new.detach().requires_grad_(True).to(self.device)
                potential_new, _ = self.calculate_potential_and_forces(x_new)
                potential_new = potential_new.detach().cpu().numpy()[:, 0]
                pairs_for_exchange = self._detect_exchange(potential_new)
                x_new, v_new = self._perform_exchange(pairs_for_exchange,
                                                      x_new, v_new)
                
            # prepare for next timestep
            x_old = x_new.detach().requires_grad_(True).to(self.device)
            v_old = v_new

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t+1) % self.export_interval > 0:
                self._save_numpy(t+1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            attempted = self._replica_exchange_attempts
            exchanged = self._replica_exchange_approved
            printstring = 'Done simulating ({})'.format(time.asctime())
            printstring += "\nReplica-exchange rate: %.2f%% (%d/%d)" % (
                            exchanged / attempted * 100., exchanged,
                            attempted)
            printstring += ("\nNote that you can call .get_replica_info"
                            "(#replica) to query the inverse temperature"
                            " and trajectory indices for a given replica.")
            if self.log_type == 'print':
                print(printstring)
            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(
            self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(
                self.simulated_forces)

        if self.save_potential:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential)

        if self.friction is not None:
            self.kinetic_energies = self._swap_and_export(
                self.kinetic_energies)

        self._simulated = True

        return self.simulated_coords

