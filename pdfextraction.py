from matplotlib import pyplot as plt
from abtem.parametrizations import LobatoParametrization
import numpy as np
import re
import ipywidgets as widgets
from IPython.display import display
from numpy.polynomial import Polynomial

def parse_formula(formula):
    """
    Parse a chemical formula string into elements and their stoichiometric ratios.
    
    Args:
        formula (str): Chemical formula like "SiO2", "Al2O3", etc.
    
    Returns:
        tuple: (elements_list, ratios_list) where ratios are normalized to sum to 1
    
    Example:
        parse_formula("SiO2") -> (['Si', 'O'], [0.333, 0.667])
    """
    # Extract element symbols and their counts using regex
    # Pattern matches: Capital letter + optional lowercase + optional digits
    tokens = re.findall(r'([A-Z][a-z]*)([0-9.]+)?', formula)
    
    elements = []
    counts = []
    for (elem, count) in tokens:
        elements.append(elem)
        counts.append(float(count) if count else 1.0)  # Default count is 1 if not specified
    
    counts = np.array(counts)
    ratios = counts / counts.sum()  # Normalize to get atomic fractions
    return elements, ratios.tolist()

def compute_avg_scattering_factor(
    formula,
    x_max: float = 3.0,
    x_step: float = 0.005,
    qvalues: bool = False,
    plot: bool = False,
    xray: bool = False
):
    elements, ratios = parse_formula(formula)

    # --- Convert to s (Lobato internal variable) ---
    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max = x_max
        s_step = x_step

    parametrization = LobatoParametrization()
    name = "x_ray_scattering_factor" if xray else "scattering_factor"

    scattering_factor = parametrization.line_profiles(
        elements,
        cutoff=s_max,
        sampling=s_step,
        name=name
    )

    # 🔑 TRUE number of points (trust abTEM, not math)
    npts = scattering_factor.array.shape[1]

    # --- Build axis consistent with abTEM ---
    s_array = np.arange(npts) * s_step
    q_array = 2 * np.pi * s_array

    # --- Weighted average ---
    avg_scattering_factor = np.zeros(npts)
    for i in range(len(elements)):
        avg_scattering_factor += ratios[i] * scattering_factor.array[i]

    if plot:
        fig, ax = plt.subplots()
        for i, symbol in enumerate(elements):
            ax.plot(q_array, scattering_factor.array[i], label=symbol)
        ax.plot(q_array, avg_scattering_factor, label="<f(q)>", lw=2)
        ax.set_xlabel(r"$Q\ (\AA^{-1})$")
        ax.legend()

    return q_array, avg_scattering_factor


    return q_array, avg_scattering_factor
def compute_f2avg(formula, x_max:float=3, x_step:float=0.005, qvalues:bool=False, plot=False, xray=False):
    """
    Compute composition-weighted average of squared atomic scattering factors.
    
    This function is similar to compute_avg_scattering_factor but computes <f²(q)>
    instead of <f(q)>². This distinction is important for PDF analysis.
    
    Args:
        formula (str): Chemical formula
        x_max (float): Maximum scattering vector value
        x_step (float): Step size for scattering vector
        qvalues (bool): If True, x_max/x_step are in q units; if False, in s units
        plot (bool): Whether to create a plot
        xray (bool): If True, compute X-ray scattering factors; if False, electron scattering
    
    Returns:
        tuple: (q_array, f2avg) where f2avg = <f²(q)>
    """
    # Parse the chemical formula
    elements, ratios = parse_formula(formula)

    # Set up the scattering vector axis (same logic as above)
    if not qvalues:
        s_array = np.arange(0, x_max, x_step)
        q_array = 2*np.pi*s_array
    else:
        q_array = np.arange(0, x_max, x_step)
        coeff = 2*np.pi
        x_max = x_max/coeff
        x_step = x_step/coeff

    # Compute atomic scattering factors
    parametrization = LobatoParametrization()
    if xray:
        scattering_factor = parametrization.line_profiles(
            elements, cutoff=x_max, sampling=x_step, name="x_ray_scattering_factor"
        )
    else:
        scattering_factor = parametrization.line_profiles(
            elements, cutoff=x_max, sampling=x_step, name="scattering_factor"
        )
    
    # Compute weighted average of squared factors: <f²(q)> = Σ(ci * fi²(q))
    f2avg = 0
    for i, symbol in enumerate(elements):
        f2avg += ratios[i] * scattering_factor.array[i]**2

    # Optional plotting
    if plot:
        fig, ax = plt.subplots()
        # Plot individual squared scattering factors
        for i, symbol in enumerate(elements):
            ax.plot(q_array, scattering_factor.array[i]**2, label=symbol)
        # Plot average squared scattering factor
        ax.plot(q_array, f2avg, label='<f²(q)>')
        ax.legend()
        ax.set_xlabel('Q ($\AA^{-1}$)')
    
    return q_array, f2avg

def fit_polynomial_background(q, Fm, rpoly=0.9, qmin=None, qmax=None):
    """
    Fit a polynomial background to the modified intensity function F(q).
    
    This is used to remove systematic background trends from the scattering data
    before Fourier transformation to real space.
    
    Args:
        q (array): Scattering vector values
        Fm (array): Modified intensity function F(q)
        rpoly (float): Factor controlling polynomial degree (degree ≈ rpoly * qmax/π)
        qmin (float): Minimum q for fitting range
        qmax (float): Maximum q for fitting range
    
    Returns:
        array: Fitted polynomial background evaluated at all q points
    """
    if qmin is None:
        qmin = 0.3
    if qmax is None:
        qmax = q.max()

    # Create mask for fitting range
    mask = (q >= qmin) & (q <= qmax)
    
    # Determine polynomial degree based on data range
    deg = round(rpoly * qmax / np.pi)
    deg = max(1, min(deg, len(q[mask]) - 1))  # Ensure valid degree

    # Fit polynomial to F(q)/q to reduce oscillations
    y = Fm[mask] / q[mask]
    poly = Polynomial.fit(q[mask], y, deg=deg, domain=[qmin, qmax])
    
    # Return polynomial background multiplied by q
    return q * poly(q)

def compute_PDF(
    q,
    Iexp,
    Iref=None,
    bgscale: float = 1,
    qmin: float = 0.3,
    qmax: float = None,
    qmaxinst: float = None,
    composition: str = None,
    rmin: float = 0,
    rmax: float = 50,
    rstep: float = 0.01,
    rpoly: float = 0.9,
    Qdamp: float = 0.09,
    Lorch: bool = True,
    plot: bool = False,
    xray: bool = False,):
    """
    Compute the Pair Distribution Function (PDF) G(r) from experimental scattering data.
    
    This is the main function that performs the complete PDF analysis workflow:
    1. Background subtraction
    2. Normalization by scattering factors
    3. Conversion to structure factor
    4. Background fitting and subtraction
    5. Fourier transform to real space
    
    Args:
        q (array): Scattering vector values
        Iexp (array): Experimental intensity data
        Iref (array, optional): Reference background intensity
        bgscale (float): Scaling factor for reference background
        qmin (float): Minimum q for PDF calculation
        qmax (float): Maximum q for PDF calculation
        qmaxinst (float): Maximum q for background fitting
        composition (str): Chemical formula for normalization
        rmin (float): Minimum r for PDF output
        rmax (float): Maximum r for PDF output  
        rstep (float): Step size for r grid
        rpoly (float): Polynomial background fitting parameter
        Qdamp (float): damping factor obtained through instrument calibration
        Lorch: apply Lorch window correction to eliminate termination ripples
        plot (bool): Whether to create diagnostic plots
        
    
    Returns:
        tuple: (r, G) where r is distance array and G is the PDF
    """
    # Set default values
    if qmax is None:
        qmax = np.max(q)
    if qmaxinst is None:
        qmaxinst = qmax

    # Background subtraction if reference provided
    if Iref is not None:
        Iexp = Iexp - bgscale * Iref

    # Get scattering step size for atomic form factor calculation
    qstep = q[1] - q[0]
    
    # Compute average atomic scattering factor for normalization
    if composition is not None:
        q_favg_e, favg_e = compute_avg_scattering_factor(
            composition,
            x_max=qmax,
            x_step=qstep,
            qvalues=True,
            plot=False,
            xray=xray,
        )
    else:
        raise ValueError("Veuillez fournir une composition chimique")

    # Interpolate scattering factor to match experimental q grid
    favg_e = np.interp(q, q_favg_e, favg_e)
    
    # Estimate high-q intensity limit for normalization
    mask_inf = q > 0.9 * np.max(q)  # Use top 10% of q range
    I_infty = np.mean(Iexp[mask_inf])

    # Normalize intensity by squared scattering factor
    Inorm = Iexp / (favg_e**2)
    
    # Compute modified intensity function F(q) = q[I(q)/I∞ - 1]
    F_m = q * (Inorm / I_infty - 1)

    # Fit and subtract polynomial background
    background = fit_polynomial_background(q, F_m, rpoly=rpoly, qmin=qmin, qmax=qmaxinst)
    
    # Apply corrections: background subtraction + thermal damping
    b = Qdamp**2 / 2
    F_c = (F_m - background) * np.exp(-b * q**2)
    

    # Set up r-space grid for PDF
    r = np.linspace(rmin, rmax, int((rmax - rmin) / rstep) + 1)
    
    # Apply q-range mask for Fourier transform
    mask = (q >= qmin) & (q <= qmax)
    qv = q[mask]
    # Apply Lorch window function if specified
    if Lorch:
        Lorch_window = np.sinc(qv / qmax)
        Fv = F_c[mask] * Lorch_window
    else:
        Fv = F_c[mask]
    W = np.ones_like(qv)  # Uniform weighting (could implement q-dependent weights)

    # Fourier transform: G(r) = (2/π) ∫ F(q) sin(qr) dq
    integrand = (Fv * W)[None, :] * np.sin(np.outer(r, qv))
    G = (2 / np.pi) * np.trapz(integrand, qv, axis=1)

    # Optional diagnostic plots
    if plot:
        fig, ax = plt.subplots(3, figsize=(4, 6))
        
        # Plot 1: Raw intensities
        ax[0].plot(q, Iexp, label="Iexp")
        if Iref is not None:
            ax[0].plot(q, bgscale * Iref, label="Ref*bgscale")
        ax[0].legend()
        ax[0].set_xlabel("Q ($\\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")

        # Plot 2: Corrected structure factor
        ax[1].plot(q, F_c, label=f"rpoly={rpoly:.2f}, b={b:.4f}")
        ax[1].legend()
        ax[1].set_xlabel("Q ($\\AA^{-1}$)")
        ax[1].set_ylabel("F(Q)")

        # Plot 3: Final PDF
        ax[2].plot(r, G, label=f"rpoly={rpoly:.2f}, b={b:.4f}")
        ax[2].legend()
        ax[2].set_xlabel("r ($\\AA$)")
        ax[2].set_ylabel("G(r)")

        fig.tight_layout()
        plt.show()

    return r, G

# ------------------
# Interactive GUI Class
# ------------------
class PDFInteractive:
    """
    Interactive widget-based interface for PDF parameter optimization.
    
    This class provides real-time parameter adjustment with immediate visual feedback,
    making it easier to optimize PDF processing parameters interactively.
    """
    
    def __init__(self, q, Iexp, composition, Iref=None, rmin=0, rmax=50, rstep=0.01,xray: bool = False):
        """
        Initialize the interactive PDF interface.
        
        Args:
            q (array): Scattering vector values
            Iexp (array): Experimental intensity data
            composition (str): Chemical formula
            Iref (array, optional): Reference background
            rmin (float): Minimum r for PDF
            rmax (float): Maximum r for PDF
            rstep (float): Step size for r
        """
        # Store PDF computation parameters
        self.xray = xray
        self.pdf_config = dict(
            q=q, Iexp=Iexp, Iref=Iref, composition=composition,
            rmin=rmin, rmax=rmax, rstep=rstep,
        )
        
        # Storage for last computed results (for saving)
        self.last_r = None
        self.last_G = None

        # Create parameter control sliders
        self.bgscale_slider = widgets.FloatSlider(
            value=1, min=0, max=1, step=0.01, 
            description="bgscale", readout_format=".2f"
        )
        self.qmin_slider = widgets.FloatSlider(
            value=np.min(q), min=np.min(q), max=np.max(q), step=0.01,
            description="qmin", readout_format=".2f"
        )
        self.qmax_slider = widgets.FloatSlider(
            value=np.max(q), min=np.min(q), max=np.max(q), step=0.01,
            description="qmax", readout_format=".2f"
        )
        self.qmaxinst_slider = widgets.FloatSlider(
            value=np.max(q), min=np.min(q), max=np.max(q), step=0.01,
            description="qmaxinst", readout_format=".2f"
        )
        self.rpoly_slider = widgets.FloatSlider(
            value=0.9, min=0.1, max=2.5, step=0.01,
            description="rpoly", readout_format=".2f"
        )
        self.Qdamp_slider = widgets.FloatSlider(
            value=0.09, min=0, max=0.5, step=0.0001,
            description="Qdamp", readout_format=".4f"
        )

        self.lorch_checkbox = widgets.Checkbox(
            value=True,
            description="apply Lorch window correction to eliminate termination ripples",
            indent=False)

        # Save button for exporting results
        self.save_button = widgets.Button(description="💾 Save CSV", button_style="success")
        self.save_button.on_click(self.save_results)

        # Organize widgets in vertical layout
        self.sliders = widgets.VBox([
            self.bgscale_slider,
            self.qmin_slider,
            self.qmax_slider,
            self.qmaxinst_slider,
            self.rpoly_slider,
            self.Qdamp_slider,
            self.lorch_checkbox,
            self.save_button])

        # Output area for plots
        self.plot_output = widgets.Output()

        # Link sliders to update function for real-time feedback
        widgets.interactive_output(self.update_plot, {
            "bgscale": self.bgscale_slider,
            "qmin": self.qmin_slider,
            "qmax": self.qmax_slider,
            "qmaxinst": self.qmaxinst_slider,
            "rpoly": self.rpoly_slider,
            "Qdamp": self.Qdamp_slider,
            "lorch": self.lorch_checkbox})

    def update_plot(self, bgscale, qmin, qmax, qmaxinst, rpoly, Qdamp, lorch):
        """
        Update the PDF calculation and plots when parameters change.
        
        This function is called automatically when any slider value changes.
        """
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            # Recompute PDF with new parameters
            r, G = compute_PDF(
                **self.pdf_config,
                bgscale=bgscale, qmin=qmin, qmax=qmax,
                qmaxinst=qmaxinst, rpoly=rpoly, Qdamp=Qdamp, plot=True, Lorch=lorch, xray=self.xray)
            # Store results for potential saving
            self.last_r, self.last_G = r, G

    def save_results(self, b):
        """
        Save the last computed PDF results to CSV file.
        
        Args:
            b: Button widget (unused, required by widget callback signature)
        """
        if self.last_r is None or self.last_G is None:
            print("⚠️ Aucun résultat à sauvegarder (génère d'abord un plot).")
            return

        fname = "PDF_results.csv"
        # Combine r and G into two-column format
        data = np.column_stack((self.last_r, self.last_G))
        np.savetxt(fname, data, delimiter=",", header="r,G", comments="")
        print(f"✅ Résultats sauvegardés dans {fname}")

    def show(self):
        """
        Display the interactive interface.
        
        Creates a horizontal layout with sliders on the left and plots on the right.
        """
        ui = widgets.HBox([self.sliders, self.plot_output])
        display(ui)
        
        # Generate initial plot with default parameter values
        self.update_plot(
            self.bgscale_slider.value, self.qmin_slider.value,
            self.qmax_slider.value, self.qmaxinst_slider.value,
            self.rpoly_slider.value, self.Qdamp_slider.value,self.lorch_checkbox.value
        )