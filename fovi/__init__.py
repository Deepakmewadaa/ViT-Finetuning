from .adapter import FOVIAdapter
from .analysis import (
    backproject_receptive_fields,
    compute_layer_rf_diameters,
    fit_motter_isotropy_aspect_ratios,
    plot_motter_isotropy,
    plot_prf_scaling,
    plot_sensor_manifold,
)
from .baselines import (
    LogPolarSampler,
    LogPolarSensor,
    UniformDownsampleSampler,
    WarpedCartesianSampler,
    make_weak_fovi_sensor,
)
from .benchmark import BenchmarkResult, benchmark_model, estimate_flops, format_benchmark_table
from .geometry import (
    FoveatedSensor,
    SensorGrid,
    choose_radial_samples,
    cmf_scaling_factor,
    cortical_magnification,
    count_samples_for_radial_samples,
    find_discrete_a_for_patches,
    integrated_cmf,
    inverse_integrated_cmf,
    make_foveated_grid,
)
from .knn import covering_knn_size, knn_indices
from .layers import (
    FOVIViTPatchEmbed,
    KNNConv,
    KNNMaxPool,
    compute_reference_kernel_grid,
)
from .lora import (
    LoRALinear,
    LoRAReplacement,
    apply_fovi_lora_to_vit,
    apply_lora_to_linears,
)
from .models import (
    FOVICNN,
    FOVIResNet,
    FOVIResidualBlock,
    FOVIViT,
    fovi_vit_base,
    fovi_vit_huge,
    fovi_vit_small,
)
from .sampling import FoveatedSampler, random_fixations
from .training import (
    FOVITrainer,
    SyntheticImageDataset,
    TrainMetrics,
    build_cifar_dataset,
    build_image_folder_dataset,
)

__version__ = "0.2.0"

__all__ = [
    # Geometry
    "SensorGrid",
    "FoveatedSensor",
    "cortical_magnification",
    "cmf_scaling_factor",
    "integrated_cmf",
    "inverse_integrated_cmf",
    "count_samples_for_radial_samples",
    "choose_radial_samples",
    "find_discrete_a_for_patches",
    "make_foveated_grid",
    # kNN & Layers
    "knn_indices",
    "covering_knn_size",
    "compute_reference_kernel_grid",
    "KNNConv",
    "KNNMaxPool",
    "FOVIViTPatchEmbed",
    # Sampling & Baselines
    "FoveatedSampler",
    "random_fixations",
    "LogPolarSensor",
    "LogPolarSampler",
    "WarpedCartesianSampler",
    "UniformDownsampleSampler",
    "make_weak_fovi_sensor",
    # Models
    "FOVICNN",
    "FOVIResNet",
    "FOVIResidualBlock",
    "FOVIViT",
    "fovi_vit_small",
    "fovi_vit_base",
    "fovi_vit_huge",
    # Adapter & LoRA
    "FOVIAdapter",
    "LoRALinear",
    "LoRAReplacement",
    "apply_lora_to_linears",
    "apply_fovi_lora_to_vit",
    # Analysis & Biology
    "backproject_receptive_fields",
    "compute_layer_rf_diameters",
    "fit_motter_isotropy_aspect_ratios",
    "plot_sensor_manifold",
    "plot_prf_scaling",
    "plot_motter_isotropy",
    # Training & Profiling
    "SyntheticImageDataset",
    "build_cifar_dataset",
    "build_image_folder_dataset",
    "FOVITrainer",
    "TrainMetrics",
    "BenchmarkResult",
    "benchmark_model",
    "format_benchmark_table",
    "estimate_flops",
]
