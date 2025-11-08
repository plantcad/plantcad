
## Local installation

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/kuleshov-group/PlantCaduceus.git
cd PlantCaduceus

# Create and activate environment
conda create -n PlantCAD python=3.11
conda activate PlantCAD
pip install -r requirements.txt
```

**Verify torch**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Install mamba-ssm**
```bash
pip install mamba-ssm==2.2.4 causal-conv1d==1.5.0.post8 --no-build-isolation
```

## Verify installation
```python
# Test core dependencies
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/PlantCaduceus_l32', trust_remote_code=True)
device = 'cuda:0'
model.to(device)
print("✅ Installation successful!")
```

## If your server doesn't have Internet access

Download the model on a device **with Internet**, then copy the folder to your GPU server (via `scp`, shared storage, etc.):

```bash
# Clone the full model folder from Hugging Face
git lfs install
git clone https://huggingface.co/kuleshov-group/PlantCaduceus_l32
```

💡 If you are in China, you can use the mirror: https://hf-mirror.com

Once transferred, test that the model loads correctly from the local folder:

```python
# Test core dependencies
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained('PlantCaduceus_l32', trust_remote_code=True)
device = 'cuda:0'
model.to(device)
print("✅ Installation successful!")
```

## Troubleshooting installation

**mamba_ssm issues (most common):**
```bash
# If mamba_ssm import fails, reinstall with:
pip uninstall mamba-ssm
pip install mamba-ssm==2.4.0 --no-build-isolation
```

**CUDA/GPU issues:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- For CPU-only usage: Models will work but be significantly slower