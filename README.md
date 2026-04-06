# Over-the-Air Semantic Alignment via Stacked Intelligent Metasurfaces

<h5 align="center">
     
 
[![arXiv](https://img.shields.io/badge/Arxiv-2512.05657-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.05657)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/semantic-alignment-via-sim/blob/main/LICENSE)

 <br>

</h5>


> [!TIP]
> Semantic communication systems aim to transmit task-relevant information between devices capable of artificial intelligence, but their performance can degrade when heterogeneous transmitter--receiver models produce misaligned latent representations. Existing semantic alignment methods typically rely on additional digital processing at the transmitter or receiver, increasing overall device complexity. In this work, we introduce the first over-the-air semantic alignment framework based on stacked intelligent metasurfaces (SIM), which enables latent-space alignment directly in the wave domain, reducing substantially the computational burden at the device level. We model SIMs as trainable linear operators capable of emulating both supervised linear aligners and zero-shot Parseval-frame-based equalizers. To realize these operators physically, we develop a gradient-based optimization procedure that tailors the metasurface transfer function to a desired semantic mapping. Experiments with heterogeneous vision transformer (ViT) encoders show that SIMs can accurately reproduce both supervised and zero-shot semantic equalizers, achieving up to 90% task accuracy in regimes with high signal-to-noise ratio (SNR), while maintaining strong robustness even at low SNR values.

## Simulations

This section provides the necessary commands to run the simulations required for the experiments. The commands execute different training scripts with specific configurations. Each simulation subsection contains both the `python` command and `uv` counterpart.

### Accuracy Vs SIM Layers
```bash
# SIM Meta Atoms Intermediate Layers 16x16
python scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers

# SIM Meta Atoms Intermediate Layers 32x32
python scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers

# SIM Meta Atoms Intermediate Layers 64x64
python scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers
```

```bash
# SIM Meta Atoms Intermediate Layers 16x16
uv run scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers

# SIM Meta Atoms Intermediate Layers 32x32
uv run scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers

# SIM Meta Atoms Intermediate Layers 64x64
uv run scripts/classification.py -m sim.layers=1,2,3,4,5,6,7,8,9,10,15,20,25 alignment.type=Linear,PPFE sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 seed=27,42,100,123,144,200 simulation=accuracyVSsimlayers
```

### Accuracy Vs SNR

```bash
# SIM Meta Atoms Intermediate Layers 16x16
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr

# SIM Meta Atoms Intermediate Layers 32x32
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr

# SIM Meta Atoms Intermediate Layers 64x64
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr
```

```bash
# SIM Meta Atoms Intermediate Layers 16x16
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr

# SIM Meta Atoms Intermediate Layers 32x32
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr

# SIM Meta Atoms Intermediate Layers 64x64
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 channel.snr_db=-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0 alignment.type=PPFE,Linear seed=27,42,100,123,144,200 simulation=accuracyVSsnr
```

### Accuracy Vs Thickness

```bash
# SIM Meta Atoms Intermediate Layers 16x16
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness

# SIM Meta Atoms Intermediate Layers 32x32
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness

# SIM Meta Atoms Intermediate Layers 64x64
python scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness
```

```bash
# SIM Meta Atoms Intermediate Layers 16x16
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=16 sim.meta_atoms_intermediate_y=16 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness

# SIM Meta Atoms Intermediate Layers 32x32
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=32 sim.meta_atoms_intermediate_y=32 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness

# SIM Meta Atoms Intermediate Layers 64x64
uv run scripts/classification.py -m sim.layers=10 sim.meta_atoms_intermediate_x=64 sim.meta_atoms_intermediate_y=64 alignment.type=PPFE sim.thickness_multiplier=4,5,6,7,8,9,10 seed=27,42,100,123,144,200 sim.spacing_divisor_input=1.5,2.0 simulation=accuracyVSthickness
```

### Classifiers

The following command will initiate training of the required classifiers for the above simulations. However, this step is not strictly necessary, as the simulation scripts will automatically check for the presence of pretrained classifiers in the `models/classifiers` subfolder. If the classifiers are not found, a pretrained version (used in our paper) will be downloaded from Drive.

```bash
# Classifiers
python scripts/train_classifier.py seed=27,42,100,123,144,200 -m
```

```bash
# Classifiers
uv run scripts/train_classifier.py seed=27,42,100,123,144,200 -m
```

## Dependencies  

### Using `pip` package manager  

It is highly recommended to create a Python virtual environment before installing dependencies. In a terminal, navigate to the root folder and run:  

```bash
python -m venv <venv_name>
```

Activate the environment:  

- On macOS/Linux:  

  ```bash
  source <venv_name>/bin/activate
  ```

- On Windows:  

  ```bash
  <venv_name>\Scripts\activate
  ```

Once the virtual environment is active, install the dependencies:  

```bash
pip install -r requirements.txt
```

You're ready to go! 🚀  

### Using `uv` package manager (Highly Recommended)  

[`uv`](https://github.com/astral-sh/uv) is a modern Python package manager that is significantly faster than `pip`.  

#### Install `uv`  

To install `uv`, follow the instructions from the [official installation guide](https://github.com/astral-sh/uv#installation).  

#### Set up the environment and install dependencies  

Run the following command in the root folder:  

```bash
uv sync
```

This will automatically create a virtual environment (if none exists) and install all dependencies.  

You're ready to go! 🚀  

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@misc{pandolfo2025overtheairsemanticalignmentstacked,
      title={Over-the-Air Semantic Alignment with Stacked Intelligent Metasurfaces}, 
      author={Mario Edoardo Pandolfo and Kyriakos Stylianopoulos and George C. Alexandropoulos and Paolo Di Lorenzo},
      year={2025},
      eprint={2512.05657},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2512.05657}, 
}
```

## Authors

- [Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ)
- [Kyriakos Stylianopoulos](https://scholar.google.com/citations?user=42KAgRMAAAAJ)
- [George C. Alexandropoulos](https://scholar.google.com/citations?user=3Ltyd9sAAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?&user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
