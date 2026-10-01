# Autorotate Incoming Photos using ResNet

Fork of [Image Orientation Detection Using ResNet](https://github.com/parsapoorsh/resnet-ixion), with some changes for storage usage and usability.

Main code changes:
- Updated `run_onnx.py` to replace torchvision.transforms with a pure numpy + PIL pre-processing function (preprocess_image), eliminating torch and torchvision dependencies during inference.
- Updated `requirements.txt` to reflect the slim dependency list (torch+torchvision -> onnxruntime+numpy).
  - Original environment size: 5.6 GB (PyTorch + CUDA binaries)
  - New environment size: 200 MB (ONNX runtime + NumPy)

Now, running this in Termux on your phone requires only ~200 MB for the Python environment + 232 MB for the .onnx model file (under 450 MB total).

## Instructions

### Setup

(Assumes you're in the project root directory)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
wget https://github.com/parsapoorsh/resnet-ixion/releases/download/1.0.0/resnet152_ixion_e3-fac493d9.onnx
```

### Usage

Example - Autorotate photos on Termux that have been taken on by a Pixel device.

```bash
python3 autorotate_incoming_photos.py --watch-folder "~/storage/shared/DCIM/Camera/" --pattern "PXL_*.jpg" --state "~/storage/shared/DCIM/Camera/state.json" --log "~/storage/shared/DCIM/Camera/test.log"
```

## License
`apache-2.0`
