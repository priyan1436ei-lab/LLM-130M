"""Hardware and device detection utilities."""

import os
import torch
from typing import Dict, Any, Tuple


def get_device_info() -> Dict[str, Any]:
    """Inspect and return current compute environment information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": "cpu",
        "gpu_name": "N/A",
        "vram_gb": 0.0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_threads": os.cpu_count() or 1,
        "supports_bf16": False,
        "supports_fp16": False,
    }

    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["supports_bf16"] = torch.cuda.is_bf16_supported()
        info["supports_fp16"] = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_type"] = "mps"
        info["gpu_name"] = "Apple Silicon MPS"
        info["supports_fp16"] = True
    else:
        info["device_type"] = "cpu"
        info["gpu_name"] = "CPU"
        # Modern PyTorch CPU can technically run BF16 autocast, but FP32 is safest default
        info["supports_bf16"] = hasattr(torch, "bfloat16")

    return info


def print_device_info(selected_precision: str = "FP32") -> None:
    """Print hardware diagnostic summary as specified in project vision."""
    info = get_device_info()
    print("=" * 50)
    print("HARDWARE & ENVIRONMENT DETECTION")
    print("=" * 50)
    print(f"Device:         {info['device_type'].upper()}")
    print(f"GPU:            {info['gpu_name']}")
    print(f"VRAM:           {info['vram_gb']} GB" if info['cuda_available'] else "VRAM:           N/A")
    print(f"CUDA:           {info['cuda_version']}")
    print(f"PyTorch:        {info['pytorch_version']}")
    print(f"Precision:      {selected_precision.upper()}")
    print(f"CPU threads:    {info['cpu_threads']}")
    print(f"Number of GPUs: {info['num_gpus']}")
    print("=" * 50)


def select_device_and_dtype(requested_precision: str = "auto") -> Tuple[torch.device, torch.dtype, bool]:
    """
    Determine the optimal device, torch dtype, and whether to use GradScaler.

    Returns:
        (device, dtype, use_scaler)
    """
    info = get_device_info()

    if info["cuda_available"]:
        device = torch.device("cuda")
        if requested_precision.lower() == "auto":
            if info["supports_bf16"]:
                dtype = torch.bfloat16
                use_scaler = False
            else:
                dtype = torch.float16
                use_scaler = True
        elif requested_precision.lower() == "bf16":
            if not info["supports_bf16"]:
                raise ValueError("BF16 requested but current GPU does not support BF16.")
            dtype = torch.bfloat16
            use_scaler = False
        elif requested_precision.lower() == "fp16":
            dtype = torch.float16
            use_scaler = True
        elif requested_precision.lower() == "fp32":
            dtype = torch.float32
            use_scaler = False
        else:
            raise ValueError(f"Unknown precision: {requested_precision}")
    elif info["device_type"] == "mps":
        device = torch.device("mps")
        if requested_precision.lower() in ("fp16", "auto"):
            dtype = torch.float16
        else:
            dtype = torch.float32
        use_scaler = False
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        use_scaler = False

    return device, dtype, use_scaler
