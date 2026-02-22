# Intel-Arc-A770-LLM-Finetuning-The-Absolute-Beginner's-Guide
The definitive guide for stable, 2x faster LLM finetuning on Intel Arc A770 (16GB) using Unsloth &amp; Windows. Solves oneAPI path errors, VRAM spikes, and CUDA-check failures. No more Colab—train locally!
# 🚀 Intel Arc A770 LLM Finetuning: The Absolute Beginner's Guide

[![Unsloth](https://img.shields.io/badge/Library-Unsloth-blue)](https://github.com/unslothai/unsloth)
[![Hardware](https://img.shields.io/badge/GPU-Intel%20Arc%20A770-intel)](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html)
[![Guide](https://img.shields.io/badge/Difficulty-Beginner-green)](#)

This repository provides a stable, "battle-tested" roadmap to finetune Large Language Models (LLMs) like **DeepSeek-R1-7B** locally on an **Intel Arc A770 (16GB)**. This guide is specifically designed for absolute beginners who want to move away from Google Colab and utilize their own local hardware.

---

## 📖 Phase 1: Core Prerequisites
Intel GPUs require a specific "handshake" with Windows to work. Install these strictly on your **C: Drive** to avoid path resolution errors.

### 1. Visual Studio Community 2026
* **Download:** [Visual Studio Community 2026](https://visualstudio.microsoft.com/vs/community/)
* **Installation:** In the installer, you **must** select the **"Desktop development with C++"** workload.
* **Essential:** Ensure **"MSVC v143 - VS 2022 C++ x64/x86 build tools"** (or the latest version) is ticked in the summary sidebar.

### 2. Intel oneAPI Base Toolkit (v2025.2+)
* **Download:** [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* **Installation:** Choose **Custom Installation** and ensure **"Intel® oneAPI DPC++/C++ Compiler"** and **"Intel® oneAPI Level Zero SDK"** are selected.

### 3. Miniconda
* **Download:** [Miniconda for Windows](https://docs.anaconda.com/free/miniconda/)
* **Installation:** Tick the box to **"Add Miniconda3 to my PATH environment variable"** for easier access in PowerShell.

---

## 🛠 Phase 2: Environment Setup & Jupyter
Open the **Anaconda PowerShell Prompt** and run these commands to build your "clean room" for AI.

```powershell
# 1. Create and enter the environment
conda create -n unsloth-xpu python=3.10 -y
conda activate unsloth-xpu

# 2. Install Intel-optimized PyTorch (XPU)
pip install torch==2.9.0+xpu intel-extension-for-pytorch==2.9.0 --extra-index-url [https://pytorch-extension.intel.com/release-whl/stable/xpu/us/](https://pytorch-extension.intel.com/release-whl/stable/xpu/us/)

# 3. Install Unsloth & Jupyter Lab
pip install "unsloth[xpu] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install jupyterlab ipykernel ipywidgets transformers datasets trl accelerate

# 4. Register the environment as a Jupyter Kernel
python -m ipykernel install --user --name unsloth-xpu --display-name "AI-Finetune (Arc A770)"
```
---

## 🔍 Phase 3: Hardware Verification
Before training, run this script in your Jupyter Notebook to ensure your GPU and compilers are talking to each other.

```python
import torch
import intel_extension_for_pytorch as ipex
import os

print(f"--- Intel Arc Verification ---")
print(f"IPEX Version: {ipex.__version__}")
print(f"GPU Detected: {torch.xpu.get_device_name(0)}")
print(f"VRAM Capacity: {torch.xpu.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check if the compiler is correctly linked
intel_bin = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
if os.path.exists(os.path.join(intel_bin, "icpx.exe")):
    print("✅ Compiler Found on C: Drive")
else:
    print("❌ ERROR: Compiler not found. Reinstall oneAPI to C: drive.")

if torch.xpu.is_available():
    print("✅ SYSTEM READY FOR FINETUNING")
```
---
## 🏗 Phase 4: The "Ironclad" Finetuning Script
Standard scripts crash on Windows/Intel due to a 6.5GB VRAM warmup spike. You must include this boilerplate at the top of your Notebook to bypass these errors.
```python
import os, sys, torch, transformers, torch._dynamo
from unsloth import FastLanguageModel

# 1. HARDWARE ANCHOR (Fixes "Failed to find C compiler")
intel_bin = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
os.environ["CC"] = os.path.join(intel_bin, "icpx.exe")
os.environ["CXX"] = os.environ["CC"]
os.environ["PATH"] = intel_bin + os.pathsep + os.environ["PATH"]
os.environ["VS2022INSTALLDIR"] = r"C:\Program Files\Microsoft Visual Studio\2026\Community"

# 2. THE MEMORY PATCH (Kills the 6.5GB loading spike)
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
torch._dynamo.config.disable = True 
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["UNSLOTH_USE_TRITON"] = "1"
os.environ["ZE_PATH"] = r"C:\Program Files\LevelZeroSDK\1.26.1"

# 3. LOAD MODEL (Explicit XPU Anchor)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    max_seq_length = 1024, # Safe 16GB limit
    load_in_4bit = True,
    device_map = {"": "xpu:0"}, 
)
```
---
## 💾 Phase 5: CUDA-Free Adapter Merging
On Windows, exporting to GGUF often crashes while searching for NVIDIA "CUDA." This "Mock" patch allows you to merge weights using System RAM and Storage.
```python
import torch

# Mock CUDA to bypass NVIDIA-specific checks in Unsloth
if not hasattr(torch, "cuda"):
    class MockCuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def get_device_name(i): return "Intel Arc A770 (XPU)"
    torch.cuda = MockCuda
torch.cuda.get_device_name = lambda device=None: "Intel Arc A770 Graphics"

# 1. Save LoRA Adapters first
model.save_pretrained("./lora_adapters")
tokenizer.save_pretrained("./lora_adapters")

# 2. Merge and Export to GGUF (Uses CPU RAM/Storage)
model.save_pretrained_gguf("./model_gguf", tokenizer, quantization_method = "q4_k_m")
```
---
## 📸 Proof of Success
Below is the training log from a successful run on an Intel Arc A770.
<img width="1478" height="485" alt="image" src="https://github.com/user-attachments/assets/0db8ea95-4a9d-4139-869e-bd7b26ab2a57" />
<img width="131" height="634" alt="image" src="https://github.com/user-attachments/assets/f6907fbd-405c-4c15-a607-7c1e7fbc80f2" /> <img width="136" height="600" alt="image" src="https://github.com/user-attachments/assets/8942a45e-fe27-4ee8-a919-5c1c95e74ebb" /> <img width="132" height="605" alt="image" src="https://github.com/user-attachments/assets/bc06935d-2ec7-4938-9219-91867a98ec9c" />

Successfully achieved a Loss drop from 2.82 to 1.35 in 60 steps.
---
## 🛠 Troubleshooting for Beginners
"Out of Memory": Lower max_seq_length to 512 and ensure per_device_train_batch_size = 1.

"Visual Studio not found": Ensure you installed VS 2026 and the workload "Desktop development with C++".

AI Debugger Prompt: If you get stuck, tell an AI: "I'm on Intel Arc, Windows, using VS 2026. Here is my error log [Paste Log]. Check my CC path and VRAM spikes."
---
## 📄 License
This project is licensed under the MIT License.



