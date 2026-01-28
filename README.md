# PhysicsNemo – Instalacja Środowiska na CentOS 8

## Wymagania wstępne

- Sterownik NVIDIA kompatybilny z CUDA 12+
- GCC w wersji 9 lub wyższej (`gcc-toolset-9`)
- Conda (Miniconda/Anaconda)

---

## 1. Aktywacja GCC 9

```bash
scl enable gcc-toolset-9 bash
gcc --version
```

## 2. Utworzenie środowiska Conda

```bash
conda env create -f environment.yml
conda activate physicsnemo
```

## 3. Test instalacji

```python
python << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("CUDA runtime version:", torch.version.cuda)

from physicsnemo.models.mlp.fully_connected import FullyConnected
model = FullyConnected(in_features=32, out_features=64)
input = torch.randn(128, 32)
output = model(input)
print("Model output shape:", output.shape)
EOF
```

---

## Uwagi

- Upewnij się, że sterownik NVIDIA jest kompatybilny z CUDA 12+, np. poprzez `nvidia-smi`
- GCC musi być w wersji ≥ 9, aby zbudować rozszerzenia PhysicsNemo-SYM
- Nie jest wymagana aktualizacja sterownika NVIDIA, jeśli obecny już obsługuje CUDA 12+