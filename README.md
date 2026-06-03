## RTX 50 Series Installation

This fork provides updated installation steps for NVIDIA RTX 50-series GPUs.

Before installing MuseTalk-RTX50, please make sure **FFmpeg** is already installed and available in your system `PATH`.

You can check FFmpeg with:

```powershell
ffmpeg -version
```

If the command is not found, install FFmpeg first and restart your terminal.

---

### 1. Create Conda Environment

```powershell
conda create -n musetalk python=3.12 -y
conda activate musetalk
```

---

### 2. Install PyTorch for CUDA 12.8

```powershell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

---

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

---

### 4. Create Model Directories

```powershell
mkdir models
mkdir models\musetalk
mkdir models\musetalkV15
mkdir models\syncnet
mkdir models\dwpose
mkdir models\face-parse-bisent
mkdir models\sd-vae
mkdir models\whisper
```

---

### 5. Download Model Files

```powershell
hf download TMElyralab/MuseTalk --local-dir models

hf download stabilityai/sd-vae-ft-mse config.json --local-dir models\sd-vae
hf download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin --local-dir models\sd-vae

hf download openai/whisper-tiny config.json --local-dir models\whisper
hf download openai/whisper-tiny pytorch_model.bin --local-dir models\whisper
hf download openai/whisper-tiny preprocessor_config.json --local-dir models\whisper

hf download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir models\dwpose

hf download ByteDance/LatentSync latentsync_syncnet.pt --local-dir models\syncnet

hf download ManyOtherFunctions/face-parse-bisent 79999_iter.pth --local-dir models\face-parse-bisent
hf download ManyOtherFunctions/face-parse-bisent resnet18-5c106cde.pth --local-dir models\face-parse-bisent
```

---

## Usage

Edit the following config file:

```text
configs\inference\test.yaml
```

Modify these two fields:

```yaml
video_path: path\to\your\video.mp4
audio_path: path\to\your\audio.wav
```

Then run inference:

```powershell
python -m scripts.inference --inference_config configs\inference\test.yaml --result_dir results\test --unet_model_path models\musetalkV15\unet.pth --unet_config models\musetalkV15\musetalk.json --version v15
```

The output video will be saved in:

```text
results\test\v15
```
