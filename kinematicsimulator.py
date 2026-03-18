from ase.cluster import Icosahedron, Decahedron,Octahedron
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from pdfextraction import *
from debyecalculator import DebyeCalculator
import torch


class KinematicScatteringSimulator:
    def __init__(
        self,
        path,
        element="Au",
        morphology="Icosahedron",
        params=None,
        optimize = False):
        """
        Create a calculator for Kinematic Scattering based on Debye Scattering Equation.

        Args:
            path: directory where results are stored
            morphology (str): 'Icosahedron', 'Decahedron', 'Octahedron'.
            params (list or tuple): parameters for ASE cluster constructors.
            optimize (bool): geometrical optimiaztion of individual cluster
        """
        self.path = path
        self.element = element
        self.morphology = morphology
        if params is not None:
            self.params = params
        else:
            raise ValueError(f"Please provide parameters for morphology: {morphology}")

        self.params_str = "_".join(map(str, params))
        

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

    def compute_kinematic_PDF(
        self,
        qmin: float = 0,
        qmax: float = 25,
        qstep = 0.01,
        biso = 0.01,
        qmaxinst: float = None,
        rmin: float = 0,
        rmax: float = 50,
        rstep: float = 0.01,
        rpoly: float = 0.9,
        Qdamp: float = 0.09,
        Lorch = True,
        plot = True,
        save = True
        ):
        # save self.prototype as xyz
        self.writexyz(self.prototype)
        composition = self.composition_from_xyz('./structure.xyz')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = DebyeCalculator(qmin,qmax,qstep, biso,device=device)
        q, Ikin = calc.iq(structure_source = './structure.xyz')
        
    
        r,G = compute_PDF(
            q,
            Ikin,
            composition=composition,
            qmin = qmin,
            qmax = qmax,
            qmaxinst = qmaxinst,
            rmin = rmin,
            rmax = rmax,
            rstep = rstep,
            rpoly = rpoly,
            Qdamp = Qdamp,
            Lorch = Lorch, 
            xray=True) # we set x-ray =True as DebyeCalculator uses X-ray scattering form factors to compute I(q)
        
        if plot:
            plt.figure()
            plt.plot(r,G,label='Kinematic PDF')
            plt.xlabel('r ($\AA$)')
            plt.ylabel('G(r)')
        os.remove('./structure.xyz')
        if save:
            outfile = self.path + f'/Kinematic_PDF.gr'
            np.savetxt(outfile, np.column_stack([r,G]))
        return r, G
    
    def compute_kinematic_Iq(
            self,
            element='Au',
            qmin=0,
            qmax=25,
            qstep=0.01,
            biso=0.01):
        
        # compute I(q) using DebyeCalculator
        self.writexyz(self.prototype)
        composition = self.composition_from_xyz('./structure.xyz')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = DebyeCalculator(qmin,qmax,qstep, biso,device=device)
        qkin, I_ase_xray = calc.iq(structure_source = './structure.xyz')
        q_favg_e , favg_e = compute_avg_scattering_factor(element, x_max= qmax,x_step=qstep, qvalues=True, plot = False, xray=False)
        q_favg_xray , favg_xray = compute_avg_scattering_factor(element, x_max= qmax,x_step=qstep, qvalues=True, plot = False, xray=True)

        # interpolation on ase q grid
        favg_e=np.interp(qkin,q_favg_e,favg_e)
        favg_xray=np.interp(qkin,q_favg_xray,favg_xray)

        # COmpute I(q) for electrons
        # Divide by <f_xray>²
        Inorm=I_ase_xray/(favg_xray**2)

        # Multiply by <f_e>²
        I_ase_e=Inorm * (favg_e)**2

        return qkin, I_ase_e

        
        
        


    def writexyz(self,atoms):
        '''
        simple xyz writing, with atomic symbols/x/y/z and no other information sometimes misunderstood by some utilities, such as DebyeCalculator
        '''
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
            
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open('./structure.xyz','w') as file:
            file.write(line2write)
    
    def composition_from_xyz(self, xyz_file):
        """
        Extract chemical composition from a .xyz file.
        
        Parameters
        ----------
        xyz_file : str
            Path to .xyz file
        
        Returns
        -------
        str
            Chemical formula string (e.g. "Ag3Au2")
        """
        with open(xyz_file, "r") as f:
            lines = f.readlines()

        # skip first two lines of xyz format (atom count + comment)
        atom_lines = lines[2:]

        # extract element symbols
        elements = [line.split()[0] for line in atom_lines if line.strip()]

        # count elements
        counts = Counter(elements)

        # build formula string, sorted alphabetically
        formula = "".join(f"{el}{int(count)}" if count > 1 else el
                        for el, count in sorted(counts.items()))
        return formula