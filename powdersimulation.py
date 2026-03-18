from ase.cluster import Icosahedron, Decahedron,Octahedron
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.io import write
from ase.io.trajectory import Trajectory
import numpy as np
import os
from pdfextraction import *
from copy import deepcopy

class PowderSimulation:
    def __init__(
        self,
        element="Au",
        morphology="Icosahedron",
        params=None,
        N=20,
        box_length=80.0,
        min_gap=3,
        optimize = False):
        """
        Create a powder of nanoparticles in a box (x*x*x/2 ).

        Args:
            element (str): Element of the cluster atoms.
            morphology (str): 'Icosahedron', 'Decahedron', 'Octahedron'.
            params (list or tuple): parameters for ASE cluster constructors.
            N (int): number of nanoparticles in the powder.
            box_length (float): cubic box length (Å).
            min_gap (float): minimal interatomic distance allowed between clusters (Å).
            optimize (bool): geometrical optimiaztion of individual cluster
        """
        self.element = element
        self.morphology = morphology
        self.params = params if params is not None else [2]
        self.N = N
        self.box_length = box_length
        self.min_gap = min_gap
        params_str = "_".join(map(str, params))
        self.path = os.getcwd()+f'/powder_{morphology}_{params_str}_{element}_N={self.N}_box={box_length}'
        os.makedirs(self.path,exist_ok=True)

        # Build prototype cluster
        if morphology == "Icosahedron":
            self.prototype = Icosahedron(element, *self.params)
        elif morphology == "Decahedron":
            self.prototype = Decahedron(element, *self.params)
        elif morphology == "Octahedron":
            self.prototype = Octahedron(element, *self.params)
        else:
            raise ValueError(f"Unsupported morphology: {morphology}")
        if optimize:
            self.optimize_cluster()
        # Center at origin
        com = self.prototype.get_center_of_mass()
        self.prototype.translate(-com)

        self.packed = None


    def optimize_cluster(self):        
        """
        Perform geometric optimization based on ASE FIRE algorithm
        Overwrite the initial structure file
        """
        self.atoms = self.prototype
        self.atoms.calc = EMT()
        basename='cluster_optimisation'
        opt = FIRE(self.atoms, trajectory=self.path+'/'+basename+'_FIRE.traj')
        opt.run(fmax=0.01)
        
        traj=Trajectory(self.path+'/'+basename+'_FIRE.traj')        
        self.atoms=traj[-1]
        self.prototype = self.atoms

    def _random_rotation_matrix(self):
        """Return a random 3D rotation matrix (uniform over SO(3))."""
        u1, u2, u3 = np.random.random(3)
        q = np.array([
            np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
            np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
            np.sqrt(u1)     * np.sin(2*np.pi*u3),
            np.sqrt(u1)     * np.cos(2*np.pi*u3),
        ])
        q0, q1, q2, q3 = q
        return np.array([
            [1-2*(q2*q2+q3*q3),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3),     1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2),     2*(q2*q3+q0*q1),   1-2*(q1*q1+q2*q2)]
        ])

    def _minimal_pair_distance(self, atomsA, atomsB):
        """Minimal interatomic distance between two Atoms objects."""
        posA = atomsA.get_positions()
        posB = atomsB.get_positions()
        d2 = np.sum((posA[:, None, :] - posB[None, :, :])**2, axis=2)
        return np.sqrt(d2.min())


    def generate(self, substratefile=None, max_tries_per_particle=2000, verbose=True, z_gap=3.0, z_fraction=0.5):
        """
        Generate a packed powder configuration.

        Args:
            substratefile (str or None): path to substrate xyz file. If provided, deposit powder on top.
            max_tries_per_particle (int): max attempts to place each cluster.
            verbose (bool): show progress.
            z_gap (float): vertical gap between powder and substrate (Å).
            z_fraction (float): fraction of box_length used for powder height (default 0.5).
        """
        placed = []

        from tqdm import tqdm
        iterator = range(self.N)
        if verbose:
            iterator = tqdm(iterator, desc="Packing clusters")

        # ----------------- Process substrate -----------------
        if substratefile is not None:
            from ase.io import read
            substrate = read(substratefile)
            pos = substrate.get_positions()
            x_min, x_max = pos[:,0].min(), pos[:,0].max()
            y_min, y_max = pos[:,1].min(), pos[:,1].max()
            substrate_Lx = x_max - x_min
            substrate_Ly = y_max - y_min

            nx = int(np.ceil(self.box_length / substrate_Lx))
            ny = int(np.ceil(self.box_length / substrate_Ly))

            tiled_substrate = substrate.copy()
            tiled_substrate.set_positions(pos - np.array([x_min, y_min, 0]))

            supercell_atoms = tiled_substrate.copy()
            for i in range(nx):
                for j in range(ny):
                    if i == 0 and j == 0:
                        continue
                    copy_sub = substrate.copy()
                    copy_sub.set_positions(copy_sub.get_positions() - np.array([x_min, y_min, 0]))
                    copy_sub.translate([i * substrate_Lx, j * substrate_Ly, 0])
                    supercell_atoms += copy_sub

            # Découpe XY
            positions = supercell_atoms.get_positions()
            mask = (
                (positions[:,0] >= 0) & (positions[:,0] < self.box_length) &
                (positions[:,1] >= 0) & (positions[:,1] < self.box_length)
            )
            substrate = supercell_atoms[mask]

            # Cellule Z minimale pour la tessellation
            z_min_substrate = positions[:,2].min()
            z_max_substrate = positions[:,2].max()
            substrate.set_cell([self.box_length, self.box_length, z_max_substrate - z_min_substrate + 10])
            substrate.set_pbc([True, True, True])

        # ----------------- Define Z limits for powder -----------------
        z_min = 0
        z_max = self.box_length * z_fraction
        if substratefile is not None:
            z_min = 0  # poudre en bas
            z_max = self.box_length * z_fraction

        # ----------------- Place N clusters -----------------
        # ----------------- Place N clusters -----------------
        for i in iterator:
            success = False
            for attempt in range(max_tries_per_particle):
                R = self._random_rotation_matrix()
                new = deepcopy(self.prototype)
                pos = new.get_positions().dot(R.T)
                new.set_positions(pos)

                # Random XY within box
                center_x = np.random.uniform(0, self.box_length)
                center_y = np.random.uniform(0, self.box_length)
                # Random Z within limits
                center_z = np.random.uniform(z_min, z_max)
                new.translate([center_x, center_y, center_z])

                # Wrap only in X and Y
                new.set_cell([self.box_length, self.box_length, z_max])  # Z cell large enough
                new.set_pbc([True, True, False])  # no PBC in Z
                pos_new = new.get_positions()
                pos_new[:,0:2] = pos_new[:,0:2] % self.box_length  # wrap XY only
                new.set_positions(pos_new)

                # Overlap check
                bad = False
                for other in placed:
                    if self._minimal_pair_distance(new, other) < self.min_gap:
                        bad = True
                        break
                if not bad:
                    placed.append(new)
                    success = True
                    break
            if not success:
                raise RuntimeError(f"Failed to place particle {i+1}")

        # ----------------- Combine powder -----------------
        powder = placed[0].copy()
        for p in placed[1:]:
            powder += p
        powder.set_cell([self.box_length]*3)
        powder.set_pbc([True, True, False])

        # ----------------- Place substrate above powder -----------------
        if substratefile is not None:
            z_max_substrate = substrate.positions[:, 2].max()
            self.z_substrate = z_max_substrate
            z_min_powder = powder.positions[:, 2].min()
            shift = z_max_substrate - z_min_powder + z_gap
            powder.translate([0,0,shift])
            combined = powder + substrate
            combined.set_cell([self.box_length, self.box_length, max(combined.positions[:,2]) + 5])
            combined.set_pbc([True, True, False])
            self.packed = combined
        else:
            self.packed = powder

        return self.packed

    def visualize(self, repeat=(1, 1, 1)):
        """Visualize the packed structure with ASE viewer."""
        if self.packed is None:
            raise RuntimeError("Call .generate() first")
        os.system('ase gui powder.xyz')

    def save(self):
        """Save packed structure to file."""
        powder_filename = f'{self.path}/powder_{self.morphology}_{self.params}_N={self.N}.xyz'
        if self.packed is None:
            raise RuntimeError("Call .generate() first")
        write(powder_filename, self.packed)
        return powder_filename