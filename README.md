# QMMESP - QM/MM Environment-aware RESP Charge Derivation

A Python script for computing environment-aware RESP charges of ligands using QM/MM calculations with electrostatic embedding. The script supports both ORCA and PySCF as quantum mechanics engines and uses Multiwfn for RESP charge fitting.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Input Files](#input-files)
- [Output Files](#output-files)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Overview

QMMESP calculates RESP (Restrained Electrostatic Potential) charges for ligands while considering their local protein environment through QM/MM calculations. Unlike traditional gas-phase RESP calculations, this approach:

- **Includes environmental effects**: The ligand is treated quantum mechanically while the protein environment is modeled with classical force fields
- **Uses electrostatic embedding**: Point charges from the MM region polarize the QM wavefunction
- **Supports multiple QM engines**: Choose between PySCF (default) or ORCA for quantum calculations
- **Automated workflow**: From prepared MD structures to final RESP charges in one step

## Features

- ✅ **Dual QM Engine Support**: PySCF (default) or ORCA
- ✅ **Automatic Multiwfn Detection**: Works with both GUI and no-GUI versions
- ✅ **Environment-aware Charges**: QM/MM approach includes protein effects
- ✅ **Checkpoint Recovery**: Efficient conversion from QM checkpoints to ESP grids
- ✅ **AMBER Integration**: Direct prepi file updates for MD simulations
- ✅ **Robust Error Handling**: Multiple fallback options ensure calculation completion

## Installation

### 1. Clone the Repository

```
git clone https://github.com/miqueleg/QMMESP.git
cd QMMESP
```

### 2. Set Up Python Environment

```
#Create conda environment
conda create -n qmmesp python=3.9
conda activate qmmesp

#Install required packages
pip install numpy mdtraj
```

### 3. Install Dependencies

See [Dependencies](#dependencies) section for detailed installation instructions.

## Dependencies

### Required Software

| Software | Purpose | Installation |
|----------|---------|-------------|
| **ASH** | QM/MM calculations | `pip install ash-md` |
| **PySCF** | QM engine (default) | `pip install pyscf` |
| **ORCA** | QM engine (optional) | Download from [ORCA website](https://orcaforum.kofo.mpg.de) |
| **Multiwfn** | RESP fitting | Download from [Multiwfn website](http://sobereva.com/multiwfn/) |
| **OpenMM** | MM calculations | `conda install -c conda-forge openmm` |

### Python Packages

```
pip install numpy mdtraj pyscf ash-md
```

### Environment Variables

```
#Add to your ~/.bashrc or ~/.zshrc
export Multiwfnpath="/path/to/multiwfn"
export ORCADIR="/path/to/orca" # Only if using ORCA
```

## Usage

### Basic Command

```
python QMMESP.py --pdb complex.pdb --prmtop complex.prmtop --inpcrd complex.inpcrd --resid 1
```

### Complete Example

```
python QMMESP.py
--pdb solvated_complex.pdb
--prmtop complex.prmtop
--inpcrd complex.inpcrd
--prepi ligand.prepi
--resid 285
--charge 0
--functional HF
--basis 6-31G*
--qm_engine pyscf
--output resp_results
--numcores 4
```

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--pdb` | ✅ | - | Input PDB file (solvated complex) |
| `--prmtop` | ✅ | - | AMBER topology file |
| `--inpcrd` | ✅ | - | AMBER coordinate file |
| `--resid` | ✅ | - | Residue ID of the ligand |
| `--prepi` | ❌ | - | Original prepi file for charge updates |
| `--charge` | ❌ | 0 | Net charge of the QM region |
| `--mult` | ❌ | 1 | Multiplicity of the QM region |
| `--functional` | ❌ | HF | DFT functional (HF, B3LYP, PW6B95, etc.) |
| `--basis` | ❌ | 6-31G* | Basis set |
| `--qm_engine` | ❌ | pyscf | QM engine: `pyscf` or `orca` |
| `--output` | ❌ | qmmm_resp | Output directory |
| `--numcores` | ❌ | 8 | Number of CPU cores |
| `--multiwfnpath` | ❌ | $Multiwfnpath | Path to Multiwfn installation |
| `--orcadir` | ❌ | $ORCADIR | Path to ORCA installation |

## Input Files

### Required Files

1. **PDB File** (`--pdb`): Solvated protein-ligand complex
   - Must contain properly numbered residues
   - Ligand should have a unique residue name
   - Can include crystallographic waters

2. **Topology File** (`--prmtop`): AMBER parameter/topology file
   - Generated using tleap with appropriate force fields
   - Must include parameters for all system components

3. **Coordinate File** (`--inpcrd`): AMBER coordinate file
   - Matching the topology file
   - Typically from energy minimization or equilibration

### Optional Files

4. **Prepi File** (`--prepi`): AMBER residue library file
   - Will be updated with new RESP charges
   - Useful for subsequent MD simulations

## Output Files

### Generated Files

| File | Description |
|------|-------------|
| `substrate_[engine]_charges.txt` | RESP charges for each atom |
| `esp.molden` | Molecular orbital file for ESP analysis |
| `[residue]_[engine].prepi` | Updated prepi file with RESP charges |
| `multiwfn.log` | Multiwfn calculation log |
| `multiwfn_input.txt` | Input file for Multiwfn |
| `esp.chg` | Raw charge data from Multiwfn |


## Examples

### Example 1: Basic RESP Calculation with PySCF

```
python QMMESP.py
--pdb protein_ligand.pdb
--prmtop system.prmtop
--inpcrd system.inpcrd
--resid 500
--charge -1
```

### Example 2: High-Level RESP2-like Charges with ORCA

```
python QMMESP.py
--pdb complex_solvated.pdb
--prmtop complex.prmtop
--inpcrd complex.inpcrd
--resid 285
--functional PW6B95
--basis aug-cc-pVDZ
--qm_engine orca
--numcores 16
--output resp2_results
```

### Example 3: Update Prepi File for MD

```
python QMMESP.py
--pdb equilibrated.pdb
--prmtop system.prmtop
--inpcrd system.inpcrd
--prepi original_ligand.prepi
--resid 1
--functional B3LYP
--basis 6-31G*
--output updated_params
```

### Performance Tips

- **Use PySCF for routine calculations**: Generally faster and more stable
- **Optimize core usage**: Start with 4-8 cores, increase for large systems
- **Monitor memory usage**: Large basis sets may require significant RAM
- **Use recomended basis sets**: HF/6-31G* for ff14SB, PW6B95/aug-cc-pVDZ for ff19SB

## Method Details

### QM/MM Setup

- **QM Region**: Ligand of interest (specified by residue ID)
- **MM Region**: Protein, water, and ions
- **Embedding**: Electrostatic (point charges polarize QM wavefunction)
- **Boundary**: Automatic detection based on residue boundaries

### RESP Fitting

1. **ESP Generation**: Quantum mechanical electrostatic potential on molecular surface
2. **Grid Points**: Multiwfn automatically generates fitting points
3. **Restraints**: Standard RESP restraints applied (0.0005 au for C,N,O,S; 0.001 au for H)
4. **Fitting**: Two-stage RESP procedure with buried carbon restraints

### Recommended Methods

| Purpose | Functional | Basis Set | Notes |
|---------|------------|-----------|--------|
| **RESP1** | HF | 6-31G* | Traditional RESP charges |
| **RESP2** | PW6B95 | aug-cc-pVDZ | Modern improved method |
| **Fast screening** | B3LYP | 6-31G* | Good compromise |
| **High accuracy** | PW6B95 | aug-cc-pV(T+d)Z | Research quality |

## Citation

If you use this software in your research, please cite:

```
@software{qmmesp2025,
author = {Miquel Estévez-Gay},
title = {QMMESP: Environment-aware RESP charge derivation using QM/MM},
url = {https://github.com/miqueleg/QMMESP},
year = {2025}
}
```

### Related Publications

- Bayly, C. I.; Cieplak, P.; Cornell, W.; Kollman, P. A. J. Phys. Chem. 1993, 97, 10269-10280. (Original RESP method)
- Schauperl, M.; Nerenberg, P. S.; Jang, H.; et al. J. Chem. Theory Comput. 2020, 16, 7044-7060. (RESP2 method)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or suggestions:

1. **GitHub Issues**: [Create an issue](https://github.com/miqueleg/QMMESP/issues)
2. **Email**: [Your email if you want to provide it]
3. **ORCA Forum**: For ORCA-specific questions
4. **PySCF Community**: For PySCF-related issues
5. **ASH GitHub**: For ASH-related issues

---

**Note**: This software is provided as-is for research purposes. Always validate results against experimental data or established benchmarks for your specific systems.


