from dataclasses import dataclass


@dataclass
class Defaults:
    """
    Global default parameters for the InvarLock framework. This dataclass
    centralizes hyperparameters for different lenses, especially for the
    'auto' calibration modes.
    """

    # --- Lens 1: FFT Energy ---
    # In 'auto' mode, keep heads that constitute this percentage of cumulative energy.
    fft_energy_keep: float = 0.95

    # --- Lens 2: Mutual Information ---
    # In 'auto' mode, keep neurons whose cumulative MI constitutes this percentage of the total.
    mi_info_keep: float = 0.90

    # --- Lens 3: Stability (Koopman) ---
    # Default spectral norm margin when not using auto-calibration.
    koopman_margin: float = 1.05  # A safe default slightly above 1.0

    # --- Lens 4: RMT Clipping ---
    # Default alpha for upper spectral edge clipping.
    mp_alpha: float = 1.5
    # Default beta for lower spectral edge clipping.
    # mp_beta: float = 0.10

    # --- Global Compression Target ---
    # Fraction of original trainable parameters to aim for.
    # e.g., 0.70 -> aim for ~70% of the baseline parameter count.
    # This is a high-level objective that the auto-mode translates
    # into concrete head and neuron pruning ratios.
    target_param_keep: float = 0.70

    # --- General ---
    # Default seed for all deterministic operations.
    seed: int = 42

    # Variance Equalization minimum gain threshold
    ve_min_gain: float = 0.30


# Create a singleton instance for easy import
CFG = Defaults()


def get_default_config():
    """
    Get the default configuration for InvarLock.

    Returns:
        Defaults: A dataclass instance with default configuration values
    """
    return Defaults()
