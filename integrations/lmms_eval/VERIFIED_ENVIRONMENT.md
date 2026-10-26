# Verified environment

The downstream results were reproduced with the following environment:

| Component | Version |
|---|---|
| Python | 3.11.14 |
| LMMs-Eval | 0.6.1 + the OTFS adapter snapshot |
| official base commit | `ee31b5a65ea79c8a2a4c9d6fcab736751d39513d` |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| Transformers | 5.2.0 |
| Accelerate | 1.12.0 |
| Datasets | 4.6.1 |
| Decord | 0.6.0 |
| qwen-vl-utils | 0.0.14 |
| NumPy | 2.4.2 |
| PyAV | 15.1.0 |
| flash-attn | 2.6.3 |
| tokenizers | 0.22.2 |
| safetensors | 0.7.0 |
| huggingface-hub | 1.5.0 |
| Pillow | 12.1.1 |

`uv.lock` pins the environment resolved for the `qwen` and `video` extras. The
historical environment also contained `flash-attn==2.6.3`; it is not required
when the adapter uses the default attention implementation, but can be
installed separately when a compatible CUDA compiler is available.
