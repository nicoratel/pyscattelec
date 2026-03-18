from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.io import read,write
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np
import abtem
import os
import json
import glob
import cv2
from collections import Counter
from pdfextraction import *
import shutil
import re

class DynamicScatteringSimulator:
    def __init__(
        self,
        xyz_file):
        self.path = os.path.dirname(xyz_file)
        self.xyz_file = xyz_file
        self.PDF_ready = False # tag to indicate whether 1D scattering has been computed
        self.twoD = False # tag to indicate whether 2D scattering has been computed
        if xyz_file is None:
            print ('Please provide a structure file (xyz format)')
        else:
            self.atoms = read(xyz_file)
            self.composition = self.composition_from_xyz(xyz_file)
            self.filename = os.path.basename(xyz_file).split('/')[-1].split('.')[0] # filename without extension

            self.atoms.center(vacuum=0)

    def view_structure(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # invert z axis for beam view
        structure = self.atoms.copy()
        structure.positions[:,2] *=-1
        abtem.show_atoms(structure, ax=ax1, title="Beam view")
        print('z axis inverted for beam view, structure is unchanged')
        abtem.show_atoms(self.atoms, ax=ax2, plane="xz", title="Side view", linewidth=0)
        plt.savefig(f'{self.path}/{self.filename}_structure.png')
        #plt.show()

    def optimize_structure(self,overwrite = True):
        """
        Perform geometric optimization based on ASE FIRE algorithm
        Overwrite the initial structure file
        """
        
        self.atoms.calc = EMT()
        basename=os.path.basename(self.xyz_file).split('/')[-1].split('.')[0]
        opt = FIRE(self.atoms, trajectory=self.path+'/'+basename+'_FIRE.traj')
        opt.run(fmax=0.01)
        
        traj=Trajectory(self.path+'/'+basename+'_FIRE.traj')
        if overwrite:
            self.atoms=traj[-1]
            print("Structure optimized. Initial file has been overwritten.")
            write(self.xyz_file,self.atoms)
        else:
            stru_opt = traj[-1]
            filename = self.xyz_file.split('.')[0]+'_optimized.xyz'
            write(filename,stru_opt)
            print("Structure optimized.")
            print(f'New structure saved in {filename}')
            print('KinematicScatteringSimulator is using non optimized structure')

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

    @staticmethod
    def extract_t(fname):
        m = re.search(r'z=(-?\d+)', fname)
        if not m:
            raise ValueError("Aucun 'z=' suivi d'un entier trouvé")
        print(f'l.508 {fname} included in video')
        return int(m.group(1))

    @staticmethod
    def delete_dircontent(path):
        for file in os.listdir(path):
            fullpath = os.path.join(path, file)
            if os.path.isfile(fullpath) or os.path.islink(fullpath):
                os.remove(fullpath)   # supprime fichiers et liens
            elif os.path.isdir(fullpath):
                shutil.rmtree(fullpath)  # supprime sous-dossiers

    def make_movie (self,path,videofilename,z_substrate = 49.41):
        # Pattern pour trouver les images (adapter si besoin)
        images = sorted(glob.glob(os.path.join(path,'*.png')),key=self.extract_t)
        images = [img for img in images if self.extract_t(img) > z_substrate]
        
        # Vérifier qu'on a des images
        print(f"{len(images)} images found")
        
        if not images:
            raise ValueError(f"No image found in {path}.")

        # Lire la première image pour connaître la taille
        frame = cv2.imread(images[0])

        if frame is None:
            raise ValueError(f"Impossible to read image {images[0]}.")

        height, width, layers = frame.shape
        size = (width, height)

        # Nom du fichier vidéo de sortie
        out = cv2.VideoWriter(videofilename,
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            2,  # nombre d'images par seconde
                            size)

        for filename in images:
            img = cv2.imread(filename)
            if img is None:
                print(f"Impossible to read {filename}, image ignored.")
                continue
            img_resized = cv2.resize(img, size)
            out.write(img_resized)
            #out.write(img)
            print(f'{filename} included in video')

        out.release()
        print(" Vidéo créée avec succès : video_diffraction.avi")
    
    def compute2Dscattering(
        self,
        phonon_num_configs = 8, 
        phonons_sigmas = 0.1, 
        phonons_seed = None,
        potential_sampling = 0.1,
        slice_thickness = 1,
        exit_planes = 4,
        voltage = 3e5,
        savetag = True,
        videotag = True,
        z_substrate = 0        
        ):
        """
        Compute 2D diffraction patterns with abTEM using frozen phonon simulations.

        Parameters
        ----------
        phonon_num_configs : int, default=8
            Number of frozen phonon configurations.
        phonons_sigmas : float, default=0.1
            Standard deviation for phonon displacements.
        phonons_seed : int or None, default=None
            Random seed for reproducibility of phonon configurations.
        potential_sampling : float, default=0.1
            Real-space sampling for potential calculation (smaller -> larger reciprocal space).
        slice_thickness : float, default=1
            Thickness of each slice in Å.
        exit_planes : int, default=4
            Number of exit planes for multislice.
        voltage : float, default=3e5
            Accelerating voltage of the incident electron beam (V).
        savetag : bool, default=True
            Whether to save individual diffraction patterns as PNG images.
        videotag : bool, default=True
            Whether to generate a video from the saved diffraction patterns.
        z_substrate: float, default = 49.41
            Z value corresponding to top of carbon grid

        Notes
        -----
        - The direct beam is blocked from the diffraction patterns.
        - Saved images are power-scaled (pattern**0.1) for visualization.

        Raises
        ------
        ValueError
            If diffraction images cannot be saved.
        """
        # Account for thermal vibrations and phonon scattering
        print(f'l.206 exit_planes: {exit_planes}')
        frozen_phonons = abtem.FrozenPhonons(
            self.atoms, 
            num_configs = phonon_num_configs, 
            sigmas=phonons_sigmas, 
            seed = phonons_seed
        )
                
        # Generate potential with smaller sampling for larger angular range
        potential_series = abtem.Potential(
            frozen_phonons, 
            sampling=potential_sampling,     # Reduced sampling = larger reciprocal space
            slice_thickness= slice_thickness,
            exit_planes=exit_planes
        )
        self.dx = potential_series.sampling  # in Angstrom
        # Create incident planewave  
        plane_wave = abtem.PlaneWave(energy=voltage)
        
        # Create and compute exit wave
        exit_wave_series = plane_wave.multislice(potential_series)
        print(f'phonons, exit_planes, ny, nx {exit_wave_series}')
        print('Computing exit_waves')
        exit_wave_series.compute()
        print('Exit waves computed')
        # Get diffraction patterns 
        self.diffraction_patterns_series = exit_wave_series.diffraction_patterns()
        print('Diffraction patterns computed')
        # Average over phonons configs (number given by arg phonons_nums_configs)
        measurement = self.diffraction_patterns_series.mean(0)   # mean over phonons configs
        self.measurement = measurement.compute()                 # trigger compute (turn lazy -> concrete)
        print('Diffraction patterns averaged over phonons')
        # block the direct beam on the averaged measurement
        self.measurement_blocked = self.measurement.block_direct()
        print('Direct beam blocked')
        # get the numpy array of shape (n_exit_planes, ny, nx)
        self.patterns = self.measurement_blocked.array
        print('Patterns array extracted')
        # Retrieve array of thicknesses
        self.thicknesses = potential_series.exit_planes
        self.twoD = True
        # Save images
        if savetag:
            outputdir = self.path+'/2D_Diffraction_Patterns/'
            os.makedirs(outputdir, exist_ok = True)
            self.delete_dircontent(outputdir)
            power = 0.1
            for i, (pattern, t) in enumerate(zip(self.patterns, self.thicknesses)):
                if t>0:
                    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
                    pattern[pattern < 0] = 0
                    plt.figure(figsize=(10,10), dpi=300)
                    plt.imshow(pattern**power, cmap='jet', origin='lower')
                    plt.title(f'{self.filename} — Thickness = {t:.2f} Å')
                    plt.colorbar()
                    fname = f'{outputdir}/{self.filename}_z={int(t)}Å.png'
                    plt.savefig(fname, bbox_inches='tight')
                    plt.close()
                    print(f'File saved: {fname}')
        if videotag:
            videofilename = f'{outputdir}/{self.filename}_2D-DiffractionPatterns.avi'
            self.make_movie(outputdir,videofilename,z_substrate)

        print('2D diffraction patterns computation completed (Intensity vs.pixels).')


    
    def compute1Dscattering(
        self,
        phonon_num_configs=8, 
        phonons_sigmas=0.1, 
        phonons_seed=None,
        potential_sampling=0.1,
        slice_thickness=1,
        exit_planes=4,
        voltage=3e5,
        savetag=True,
        videotag=True,
        z_substrate=0
    ):
        """
        Compute 1D diffraction data with abTEM using frozen phonon simulations.

        This method first computes 2D diffraction patterns (if not already available),
        then performs azimuthal averaging to obtain 1D scattering profiles I(q) at 
        different sample thicknesses.

        Parameters
        ----------
        phonon_num_configs : int, default=8
            Number of frozen phonon configurations.
        phonons_sigmas : float, default=0.1
            Standard deviation for phonon displacements.
        phonons_seed : int or None, default=None
            Random seed for reproducibility of phonon configurations.
        potential_sampling : float, default=0.1
            Real-space sampling for potential calculation (smaller → larger reciprocal space).
        slice_thickness : float, default=1
            Thickness of each slice in Å.
        exit_planes : int, default=4
            Number of exit planes for multislice.
        voltage : float, default=3e5
            Accelerating voltage of the incident electron beam (V).
        savetag : bool, default=True
            Whether to save I(q) profiles as `.iq` files.
        videotag : bool, default=True
            Whether to save I(q) plots as `.png` and compile into a video.
        z_substrate : float, default=49.41
            Z value corresponding to the top of the carbon support (used for movie generation).

        Outputs
        -------
        - `.iq` files containing tabulated (q, I) data for each thickness (if savetag=True).
        - `.png` figures of I(q) vs q for each thickness (if videotag=True).
        - `.avi` video of I(q) evolution with thickness (if videotag=True).
        - JSON file storing all I(q) profiles in a structured dictionary format.

        Attributes updated
        ------------------
        self.oneD_data : np.ndarray
            Array of shape (n_thickness, n_q) containing I(q) profiles.
        self.profiles : dict
            Dictionary mapping thickness values to corresponding I(q) data.
        self.PDF_ready : bool
            Flag indicating that 1D scattering data is ready.
        """

        self.PDF_ready = True
        print(f'l.334 exit_planes: {exit_planes}')
        # Ensure 2D scattering patterns exist
        if not getattr(self, "twoD", False):
            self.compute2Dscattering(
                phonon_num_configs=phonon_num_configs, 
                phonons_sigmas=phonons_sigmas, 
                phonons_seed=phonons_seed,
                potential_sampling=potential_sampling,
                slice_thickness=slice_thickness,
                exit_planes=exit_planes,
                voltage=voltage,
                savetag=False,
                videotag=False
            )

        # Azimuthal averaging
        lineprofiles = self.diffraction_patterns_series.block_direct().azimuthal_average()
        lineprofiles_avg = lineprofiles.reduce_ensemble().compute()  # mean over phonons
        self.oneD_data = lineprofiles_avg.array

        # Build q array
        coeff = 2 * np.pi
        dq = coeff * lineprofiles.sampling  
        qmax = coeff * lineprofiles.extent
        Nq = self.oneD_data.shape[1]
        qmin = qmax - (Nq - 1) * dq
        q = qmin + np.arange(Nq) * dq
        q -= dq  # reposition q

        # Prepare output directory
        self.profiles = {}
        outputdir = os.path.join(self.path, "1D_Diffraction_Data")
        os.makedirs(outputdir, exist_ok=True)
        self.delete_dircontent(outputdir)

        # Save per-thickness profiles
        for i, t in enumerate(self.thicknesses):
            Iexp = self.oneD_data[i]
            self.profiles[str(t)] = np.column_stack([q, Iexp]).tolist()

            if savetag:
                fname = f"{outputdir}/{self.filename}_z={int(t)}Å.iq"
                np.savetxt(fname, np.column_stack([q, Iexp]), delimiter="\t")
                print(f"✅ File saved: {fname}")

            if videotag:
                self.figdir = os.path.join(outputdir, "1DFigures")
                os.makedirs(self.figdir, exist_ok=True)
                plt.figure(dpi=200)
                plt.plot(q, Iexp, label=f"z={int(t)}$\\AA$")
                plt.xlabel("Q ($\\AA^{-1}$)")
                plt.ylabel("Intensity")
                plt.xlim(1.5,np.max(q))
                plt.ylim(0,np.max(Iexp[q>1.5]))
                plt.legend()
                figname = f"{self.figdir}/{self.filename}_z={int(t)}Å.png"
                plt.savefig(figname, bbox_inches="tight")
                plt.close()
                print(f" Figure saved: {figname}")

        # Make video of 1D data
        if videotag:
            videofilename = f"{self.figdir}/{self.filename}_1D-DiffractionData_I(q).avi"
            self.make_movie(self.figdir, videofilename, z_substrate)

        # Save JSON with all profiles
        json_path = f"{outputdir}/{self.filename}_z-I(q).json"
        with open(json_path, "w") as f:
            json.dump(self.profiles, f)
        print(f" JSON saved: {json_path}")



    def plot_1Dsimulations(self):
        try:
            plt.figure(dpi=200)
            for t, data in self.profiles.items():
                data = np.array(data)
                q=data[:,0]; Iexp = data[:,1]
                if int(t)<0:
                    t=0
                if int(t)>=0:
                    plt.plot(q,Iexp,label=f'z = {t} $\AA$')
                    plt.xlabel('Q ($\AA^{-1}$)')
            plt.xlim(1,np.max(q))
            plt.ylim(0, np.max(Iexp[(q>1)]))
            plt.legend()
            figname = f'{self.figdir}/{self.filename}_1D_allslices.png'
            plt.savefig(figname)
        except:
            print('Perform compute1Dscattering first')

    
    def compute_camera_scattering(
            self,
            phonon_num_configs = 8, 
            phonons_sigmas = 0.1, 
            phonons_seed = None,
            potential_sampling = 0.1,
            slice_thickness = 1,
            exit_planes = 4,
            voltage = 3e5,
            savetag = True,
            videotag = True,
            z_substrate = 0        
        ):
        """
        Compute 2D diffraction patterns with abTEM using frozen phonon simulations.

        Parameters
        ----------
        phonon_num_configs : int, default=8
            Number of frozen phonon configurations.
        phonons_sigmas : float, default=0.1
            Standard deviation for phonon displacements.
        phonons_seed : int or None, default=None
            Random seed for reproducibility of phonon configurations.
        potential_sampling : float, default=0.1
            Real-space sampling for potential calculation (smaller -> larger reciprocal space).
        slice_thickness : float, default=1
            Thickness of each slice in Å.
        exit_planes : int, default=4
            Number of exit planes for multislice.
        voltage : float, default=3e5
            Accelerating voltage of the incident electron beam (V).
        savetag : bool, default=True
            Whether to save individual diffraction patterns as PNG images.
        videotag : bool, default=True
            Whether to generate a video from the saved diffraction patterns.
        z_substrate: float, default = 49.41
            Z value corresponding to top of carbon gridpotential_sampling=0.03):
    
        Returns the scattering obtained on the camera (last simulation)
        """
        print(f'l.468 exit_planes: {exit_planes}')
        self.compute1Dscattering(
            phonon_num_configs = phonon_num_configs, 
            phonons_sigmas = phonons_sigmas, 
            phonons_seed = phonons_seed,
            potential_sampling = potential_sampling,
            slice_thickness = slice_thickness,
            exit_planes = exit_planes,
            voltage = voltage,
            savetag = savetag,
            videotag = videotag,
            z_substrate = z_substrate        
        )
        #self.compute1Dscattering(videotag=videotag)
        t = self.thicknesses[-1]
        data = np.array(self.profiles[str(t)])

        q = data[:, 0]
        I = data[:, 1]
        return q, I
    
    def compute_kinematic_scattering(
            self,
            phonon_num_configs = 8, 
            phonons_sigmas = 0.1, 
            phonons_seed = None,
            potential_sampling = 0.1,
            slice_thickness = 1,
            exit_planes = 4,
            voltage = 3e5,
            savetag = True,
            videotag = True,
            z_substrate = 0        
        ):
        """
        Compute scattering from first exit plane (close to kinematic approximation)."""
        self.compute1Dscattering(
            phonon_num_configs = phonon_num_configs, 
            phonons_sigmas = phonons_sigmas, 
            phonons_seed = phonons_seed,
            potential_sampling = potential_sampling,
            slice_thickness = slice_thickness,
            exit_planes = exit_planes,
            voltage = voltage,
            savetag = savetag,
            videotag = videotag,
            z_substrate = z_substrate        
        )
        #self.compute1Dscattering(videotag=videotag)
        t = self.thicknesses[1]  # first exit plane after 0
        data = np.array(self.profiles[str(t)])

        q = data[:, 0]
        I = data[:, 1]
        return q, I
 
    
    
    
    def compute_dynamic_PDF(
        self,
        qmin: float = 0,
        qmax: float = None,
        qmaxinst: float = None,
        rmin: float = 0,
        rmax: float = 50,
        rstep: float = 0.01,
        rpoly: float = 0.9,
        Qdamp: float = 0.09,
        Lorch: bool = True,
        savetag=True,
        videotag=True,
        normalizeplot = False,
        rkin = None,
        Gkin = None,
        z_substrate = 0,
        plot_individual_PDF = False      
    ):
        """
        Compute Pair Distribution Functions (PDF) from 1D diffraction data.

        This method uses previously computed 1D scattering profiles I(q) to
        calculate the PDF G(r) for different thicknesses. It saves the results
        as text files, figures, a video, and a JSON dictionary.

        Parameters
        ----------
        qmin, qmax, qmaxinst, rmin, rmax, rstep, rpoly, Qdamp, Lorch: parameters used for PDF extraction
        savetag : bool, default=True
            Whether to save G(r) profiles as `.gr` files.
        videotag : bool, default=True
            Whether to save G(r) plots as `.png` and compile into a video.
        outputdir : str, default="./PDF_Data"
            Path to output directory.
        normalizeplot : bool, default = False
            Normalize plot
        rkin, Gkin: arrays
            PDF data for kinematical PDF computed using DebyeCalculator

        Outputs
        -------
        - `.gr` files containing tabulated (r, G(r)) data for each thickness (if savetag=True).
        - `.png` figures of G(r) vs r for each thickness (if videotag=True).
        - `.avi` video of G(r) evolution with thickness (if videotag=True).
        - JSON file storing all G(r) profiles in a structured dictionary format.

        Attributes updated
        ------------------
        self.PDF_dict : dict
            Dictionary mapping thickness values to corresponding G(r) data.

        Returns
        ------------------
        r, G : PDF data for last slice (= dynamic PDF)
        """

        if not getattr(self, "PDF_ready", False):
            print(
                """1D scattering computed with default simulation parameters
    Parameters
    ----------
    phonon_num_configs = 8
    phonons_sigmas = 0.1
    phonons_seed = None
    potential_sampling = 0.1
    slice_thickness = 1
    exit_planes = 4
    voltage = 3e5"""
            )
            self.compute1Dscattering()
              # PDF will be computed after 1D scattering

        # Ensure output directory exists
        outputdir=self.path+"/PDF_Data"
        os.makedirs(outputdir, exist_ok=True)
        self.delete_dircontent(outputdir) # for fresh start in case of multiple

        # Store results
        self.PDF_dict = {}
        self.residues=np.zeros(len(self.thicknesses))
        for k,t in enumerate(self.thicknesses):
            if float(t) <= 0:
                continue

            # Get corresponding I(q)
            data = np.array(self.profiles[str(t)])
            q = data[:, 0]
            Iexp = data[:, 1]

            
            r, G = compute_PDF(
                q,
                Iexp,
                composition=self.composition,
                qmin = qmin,
                qmax = qmax,
                qmaxinst = qmaxinst,
                rmin = rmin,
                xray=False,
                rmax = rmax,
                rstep = rstep,
                rpoly = rpoly,
                Qdamp = Qdamp,
                Lorch = Lorch,
                plot= plot_individual_PDF
               )
            self.PDF_dict[str(t)] = np.column_stack([r, G]).tolist()
            if Gkin is not None:
                residual = G[r>2]/np.max(G[r>2])-Gkin[rkin>2]/np.max(Gkin[rkin>2])
                self.residues[k]=np.mean(residual) # save mean value of residual to plot against thickness
            # Save data
            if savetag:
                fname = f"{outputdir}/{self.filename}_z={float(t):.2f}Å.gr"
                np.savetxt(fname, np.column_stack([r, G]), delimiter="\t")
                print(f" File saved: {fname}")

            # Save figure
            if videotag:
                self.figdir = os.path.join(outputdir, "PDF_Figures")
                os.makedirs(self.figdir, exist_ok=True)
                plt.figure(dpi=200)
                if normalizeplot:
                    plt.plot(r,G/np.max(G),label=f'z={t:.2f}$\AA$')
                    if Gkin is not None:
                        plt.plot(rkin,Gkin/np.max(Gkin),label = 'kinematic PDF')
                        plt.axhline(-0.5)
                        plt.plot(r[r>2], -0.5+residual)
                else:
                    plt.plot(r, G, label=f"z={t:.2f}$\\AA$")
                    if Gkin is not None:
                        plt.plot(rkin,Gkin,label = 'kinematic PDF')
                plt.xlabel("r ($\\AA$)")
                plt.ylabel("G(r)")
                plt.legend()
                figname = f"{self.figdir}/{self.filename}_z={int(t)}Å.png"
                plt.savefig(figname, bbox_inches="tight")
                plt.close()
                print(f" Figure saved: {figname}")

        # Save video
        if videotag:
            videofilename = f"{self.figdir}/{self.filename}_PDF_G(r).avi"
            self.make_movie(self.figdir, videofilename)
            print(f" Video saved: {videofilename}")
        

        # Save JSON
        json_path = f"{outputdir}/{self.filename}_z-G(r).json"
        with open(json_path, "w") as f:
            json.dump(self.PDF_dict, f)
        print(f" JSON saved: {json_path}")

        # plot evoultion of mean residual with t
        if Gkin is not None:
            plt.figure(dpi=200)
            plt.plot(self.thicknesses, self.residues)
            plt.plot(self.thicknesses,np.zeros(len(self.thicknesses)),'--')
            plt.axvline(z_substrate,color='black',linestyle='--',linewidth=1)
            plt.xlabel('z value of slice ($\AA$)')
            plt.ylabel('Mean Residuals')
            figname = f"{self.figdir}/{self.filename}_residuals_vs_slicethickness.png"
            plt.savefig(figname)
            plt.close()


#*********************" fonction proposée par Claude"**********************************
# correction de données dynamiques pour extraire les données cinématiques

    def two_beam_correction(q, I_exp, thickness, voltage=300e3):
        """
        Correction two-beam pour des réflexions individuelles.
        
        Basée sur : I_kin ≈ I_dyn / |cos(π·t·s_g)|²
        où s_g est l'excitation error.
        
        Parameters
        ----------
        q : array
            Vecteurs de diffusion
        I_exp : array
            Intensité expérimentale (dynamique)
        thickness : float
            Épaisseur de l'échantillon (Å)
        voltage : float
            Tension d'accélération (V)
        
        Returns
        -------
        I_corrected : array
            Intensité corrigée (approximation cinématique)
        
        Notes
        -----
        Cette méthode fonctionne surtout pour les pics de Bragg isolés,
        moins bien pour le signal diffus.
        """
        # Longueur d'onde électronique relativiste
        m0 = 9.10938e-31  # kg
        e = 1.60218e-19   # C
        c = 2.99792e8     # m/s
        h = 6.62607e-34   # J·s
        
        # Correction relativiste
        lam = h / np.sqrt(2 * m0 * e * voltage * (1 + e * voltage / (2 * m0 * c**2)))
        lam *= 1e10  # conversion en Å
        
        # Excitation error approximatif : s ≈ q²/(2k₀)
        k0 = 2 * np.pi / lam
        s_g = q**2 / (2 * k0)
        
        # Correction two-beam
        # I_kin ≈ I_dyn / |pendellösung|²
        pendellosung = np.cos(np.pi * thickness * s_g)**2 + 1e-6  # éviter division par 0
        I_corrected = I_exp / pendellosung
        
        return I_corrected
    
    def correct_experimental_data(
        self,
        q_exp, 
        I_exp, 
        thickness_estimate,
        method='hybrid',
        voltage=300e3
    ):
        """
        Corrige des données expérimentales pour extraire la composante cinématique.
        
        Parameters
        ----------
        q_exp : array
            Vecteurs de diffusion expérimentaux
        I_exp : array
            Intensité expérimentale (dynamique)
        thickness_estimate : float
            Épaisseur estimée de l'échantillon (Å)
        method : str
            'hybrid' : simulation matching (recommandé)
            'empirical' : fonction de damping empirique
            'two_beam' : correction two-beam
        voltage : float
            Tension d'accélération (V)
        
        Returns
        -------
        q_exp : array
            Vecteurs de diffusion
        I_corrected : array
            Intensité corrigée (approximation cinématique)
        correction_info : dict
            Informations sur la correction appliquée
        """
        
        if method == 'hybrid':
            # 1. Simuler avec abTEM à l'épaisseur estimée
            print(f"Simulating dynamic scattering at t={thickness_estimate:.1f} Å...")
            
            # Utiliser exit_planes pour avoir exactement l'épaisseur voulue
            self.compute2Dscattering(
                potential_sampling=0.03,
                slice_thickness=1,
                exit_planes=[thickness_estimate],
                voltage=voltage,
                savetag=False,
                videotag=False
            )
            self.compute1Dscattering(savetag=False, videotag=False)
            
            # 2. Récupérer I(q) simulé dynamique
            data_sim = np.array(self.profiles[str(thickness_estimate)])
            q_sim_dyn = data_sim[:, 0]
            I_sim_dyn = data_sim[:, 1]
            
            # 3. Calculer I(q) cinématique avec DebyeCalculator
            from debyecalculator import DebyeCalculator
            calc = DebyeCalculator(qmin=q_exp.min(), qmax=q_exp.max(), qstep=0.01)
            I_sim_kin = calc.iq(self.atoms)[1]
            q_sim_kin = calc.iq(self.atoms)[0]
            
            # 4. Calculer le facteur de correction
            I_sim_dyn_interp = np.interp(q_exp, q_sim_dyn, I_sim_dyn)
            I_sim_kin_interp = np.interp(q_exp, q_sim_kin, I_sim_kin)
            
            correction_factor = I_sim_dyn_interp / (I_sim_kin_interp + 1e-10)
            
            # 5. Appliquer aux données expérimentales
            I_corrected = I_exp / correction_factor
            
            correction_info = {
                'method': 'hybrid_simulation_matching',
                'thickness': thickness_estimate,
                'correction_factor': correction_factor,
                'q_sim': q_sim_dyn,
                'I_sim_dynamic': I_sim_dyn,
                'I_sim_kinematic': I_sim_kin_interp
            }
            
        elif method == 'empirical':
            # Calculer I(q) cinématique de référence
            from debyecalculator import DebyeCalculator
            calc = DebyeCalculator(qmin=q_exp.min(), qmax=q_exp.max(), qstep=0.01)
            q_kin, I_kin = calc.iq(self.atoms)
            I_kin_interp = np.interp(q_exp, q_kin, I_kin)
            
            # Calibrer et appliquer damping
            damping_params = calibrate_damping_params(q_exp, I_exp, I_kin_interp, thickness_estimate)
            I_corrected = empirical_dynamic_correction(q_exp, I_exp, I_kin_interp, 
                                                    thickness_estimate, damping_params)
            
            correction_info = {
                'method': 'empirical_damping',
                'damping_params': damping_params,
                'thickness': thickness_estimate
            }
            
        elif method == 'two_beam':
            I_corrected = two_beam_correction(q_exp, I_exp, thickness_estimate, voltage)
            
            correction_info = {
                'method': 'two_beam_approximation',
                'thickness': thickness_estimate,
                'voltage': voltage
            }
        
        # Visualisation
        self._plot_correction_results(q_exp, I_exp, I_corrected, correction_info)
        
        return q_exp, I_corrected, correction_info

    def _plot_correction_results(self, q_exp, I_exp, I_corrected, info):
        """Visualise les résultats de la correction."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # I(q) avant/après
        axes[0, 0].plot(q_exp, I_exp, label='Experimental (dynamic)', lw=2)
        axes[0, 0].plot(q_exp, I_corrected, label='Corrected (kinematic approx.)', lw=2, alpha=0.7)
        axes[0, 0].set_xlabel('q (Å⁻¹)')
        axes[0, 0].set_ylabel('I(q)')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'Dynamic Correction - Method: {info["method"]}')
        
        # Facteur de correction
        if 'correction_factor' in info:
            axes[0, 1].plot(q_exp, info['correction_factor'], lw=2)
            axes[0, 1].axhline(1, color='k', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel('q (Å⁻¹)')
            axes[0, 1].set_ylabel('Correction factor (I_dyn/I_kin)')
            axes[0, 1].set_title('Dynamic Enhancement Factor')
        
        # PDF avant correction
        from pdfextraction import compute_PDF
        r_exp, G_exp = compute_PDF(q_exp, I_exp, composition=self.composition, xray=False)
        axes[1, 0].plot(r_exp, G_exp, lw=2)
        axes[1, 0].set_xlabel('r (Å)')
        axes[1, 0].set_ylabel('G(r)')
        axes[1, 0].set_title('PDF from Uncorrected Data (Dynamic)')
        axes[1, 0].set_xlim(0, 20)
        
        # PDF après correction
        r_corr, G_corr = compute_PDF(q_exp, I_corrected, composition=self.composition, xray=False)
        axes[1, 1].plot(r_corr, G_corr, lw=2, color='C1')
        axes[1, 1].set_xlabel('r (Å)')
        axes[1, 1].set_ylabel('G(r)')
        axes[1, 1].set_title('PDF from Corrected Data (Kinematic Approx.)')
        axes[1, 1].set_xlim(0, 20)
        
        plt.tight_layout()
        figname = f'{self.path}/experimental_dynamic_correction.png'
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        print(f"✅ Correction results saved: {figname}")

    def calibrate_damping_params(q_exp, I_exp, I_kin, thickness_estimate):
        """
        Calibre les paramètres de damping pour minimiser l'écart avec le modèle cinématique.
        """
        from scipy.optimize import minimize
        
        def objective(params):
            A, q0, sigma = params
            damping_params = {'A': A, 'q0': q0, 'sigma': sigma, 't0': 50.0}
            I_corrected = empirical_dynamic_correction(q_exp, I_exp, I_kin, 
                                                    thickness_estimate, damping_params)
            
            # Minimiser la différence avec le modèle cinématique
            return np.sum((I_corrected - I_kin)**2)
        
        # Optimisation
        result = minimize(objective, x0=[0.5, 3.0, 2.0], 
                        bounds=[(0, 2), (1, 10), (0.5, 5)])
        
        return {'A': result.x[0], 'q0': result.x[1], 'sigma': result.x[2], 't0': 50.0}
    
    def empirical_dynamic_correction(q, I_exp, I_kinematic, thickness, damping_params=None):
        """
        Applique une fonction de damping empirique pour corriger les effets dynamiques.
        
        Modèle : I_dyn(q) ≈ I_kin(q) · [1 + f(q,t)]
        où f(q,t) est une fonction empirique d'amplification dynamique.
        
        Parameters
        ----------
        q : array
            Vecteurs de diffusion
        I_exp : array
            Intensité expérimentale
        I_kinematic : array
            Intensité cinématique (calculée avec DebyeCalculator)
        thickness : float
            Épaisseur estimée de l'échantillon (Å)
        damping_params : dict, optional
            Paramètres de la fonction de damping
        
        Returns
        -------
        I_corrected : array
            Intensité corrigée
        """
        if damping_params is None:
            # Paramètres par défaut (à ajuster selon votre système)
            damping_params = {
                'A': 0.5,       # amplitude des oscillations
                'q0': 3.0,      # position du maximum d'effet
                'sigma': 2.0,   # largeur
                't0': 50.0      # épaisseur de référence
            }
        
        # Fonction de damping empirique
        # Les effets dynamiques sont maximaux à q intermédiaire
        dynamic_enhancement = damping_params['A'] * \
                            (thickness / damping_params['t0']) * \
                            np.exp(-((q - damping_params['q0']) / damping_params['sigma'])**2)
        
        # Correction
        I_corrected = I_exp / (1 + dynamic_enhancement)
        
        return I_corrected
        
        


    