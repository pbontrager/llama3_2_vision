# llama3.2 Vision
Llama3.2 Vision Builders and Utils for torchtune

## Installation

**Step 1:** [Install PyTorch](https://pytorch.org/get-started/locally/). torchtune is tested with the latest stable PyTorch release as well as the preview nightly version. torchtune leverages
torchvision for fine-tuning multimodal LLMs and torchao for the latest in quantization techniques, you should install these as well.

```
# Install stable version of PyTorch libraries using pip
pip install torch torchvision torchao

# Nightly install for latest features
pip install --pre torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu121
```

**Step 2:** Install from source to ensure access to the latest changes.

```
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install -e ".[dev]"
```

> [!NOTE]
> You will need access to llama-3.2-11b-vision-instruct to run this code

## Running recipes

All recipes are in torchtune while the configs are local. Run recipes as shown below. For more tune options see [torchtune Docs](https://pytorch.org/torchtune/stable/tune_cli.html)

```bash
tune run finetune_single_device --config ./11B_full_single_device.yaml
```
