#!/usr/bin/env python
"""
Environment-aware RESP charge derivation for substrates using ASH QM/MM
with ORCA or PySCF quantum calculations and electrostatic embedding
"""

import os
import argparse
import subprocess
import ash
from ash import Fragment
from ash import ORCATheory
from ash import PySCFTheory
from ash import QMMMTheory
from ash import OpenMMTheory
import mdtraj as md
import numpy as np
import shutil

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="QM/MM RESP charge derivation with ORCA or PySCF")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--prmtop", required=True, help="Input prmtop file")
    parser.add_argument("--inpcrd", required=True, help="Input inpcrd file")
    parser.add_argument("--prepi", help="Original prepi files")
    parser.add_argument("--resid", required=True, type=int, 
                      help="Residue ID of substrate")
    parser.add_argument("--charge", type=int, default=0, 
                      help="QM region charge")
    parser.add_argument("--mult", type=int, default=1, 
                      help="QM region multiplicity")
    parser.add_argument("--basis", default="6-31G*", 
                      help="Basis set (default: 6-31G*)")
    parser.add_argument("--functional", default="HF", 
                      help="DFT functional (default: HF)")
    parser.add_argument("--output", default="qmmm_resp", 
                      help="Output directory")
    parser.add_argument("--numcores", default=8, type=int,
                      help="Number of CPU cores assigned to the QM calculation")
    parser.add_argument("--qm_engine", choices=["orca", "pyscf"], default="pyscf",
                      help="QM engine to use: orca or pyscf (default: pyscf)")
    parser.add_argument("--orcadir", default=None,
                      help="Specify Orca installation directory")
    parser.add_argument('--multiwfnpath', type=str,
                    default=os.getenv('Multiwfnpath'),
                    help='Path to software (defaults to $Multiwfnpath)')
    parser.add_argument("--Noautostart", action="store_true",
                      help="Add Noautostart to ORCA input to prevent using previous GBW files")

    args = parser.parse_args()

    if args.multiwfnpath is None:
        parser.error("No path provided via --multiwfnpath or $Multiwfnpath environment variable.")
    
    # Convert prepi path to absolute path before any directory changes
    if args.prepi:
        args.prepi = os.path.abspath(args.prepi)
        print(f"Using absolute path for prepi file: {args.prepi}")
        
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load molecular system from PDB
    print(f"Loading PDB file: {args.pdb}")
    system = Fragment(amber_prmtopfile=args.prmtop, amber_inpcrdfile=args.inpcrd)
    
    # Define QM region (substrate) by residue ID
    top = md.load(args.pdb).topology
    qm_atoms = top.select(f'resid {args.resid-1}') 
    
    if len(qm_atoms) == 0:
        raise ValueError(f"No atoms found for residue ID {args.resid}")
    print(f"Defined QM region with {len(qm_atoms)} atoms")
    
    # Setup QM calculator based on chosen engine
    if args.qm_engine == "orca":
        qm_calc = setup_orca_theory(args)
        print(f"Using ORCA with {args.functional}/{args.basis}")
    elif args.qm_engine == "pyscf":
        qm_calc = setup_pyscf_theory(args)
        print(f"Using PySCF with {args.functional}/{args.basis}")
    
    # Setup MM calculator
    mm_calc = OpenMMTheory(
        Amberfiles=True,
        amberprmtopfile=args.prmtop,
        periodic=True,
        platform="CUDA",
        autoconstraints=None,
        rigidwater=False
    )
    
    # Create QM/MM object with electrostatic embedding
    qmmm = QMMMTheory(
        fragment=system,
        qmatoms=qm_atoms,
        qm_theory=qm_calc,
        mm_theory=mm_calc,
        embedding='elstat',
        printlevel=2,
    )
    
    # Run single-point calculation
    print(f"Running QM/MM single-point calculation with {args.qm_engine.upper()}...")
    result = ash.Singlepoint(
        theory=qmmm,
        fragment=system,
        charge=args.charge,
        mult=args.mult
    )
    
    print(f"QM/MM Energy: {result.energy} Hartree")
    
    # Extract RESP charges
    extract_resp_charges("./", args.output, qm_atoms, system, args, top)
    
    print(f"QM/MM RESP calculation with {args.qm_engine.upper()} completed successfully!")

def setup_orca_theory(args):
    """Setup ORCA QM theory object"""
    orca_input = f"!{args.functional} {args.basis} keepdens"
    
    # Add Noautostart if requested
    if args.Noautostart:
        orca_input += " Noautostart"
    
    return ORCATheory(
        orcasimpleinput=orca_input,
        numcores=args.numcores,
        orcadir=args.orcadir
    )

def setup_pyscf_theory(args):
    """Setup PySCF QM theory object"""
    return PySCFTheory(
        scf_type='RHF',
        functional=args.functional,
        basis=args.basis,
        numcores=args.numcores,
        write_chkfile_name='pyscf.chk'  # Ensure checkpoint file is saved
    )

def find_multiwfn_executable(multiwfnpath):
    """Find the appropriate Multiwfn executable (noGUI preferred, then GUI version)"""
    # Check for Multiwfn_noGUI first (preferred for automated scripts)
    nogui_path = os.path.join(multiwfnpath, "Multiwfn_noGUI")
    if os.path.exists(nogui_path) and os.access(nogui_path, os.X_OK):
        print("Using Multiwfn_noGUI executable")
        return nogui_path
    
    # Check for regular Multiwfn
    gui_path = os.path.join(multiwfnpath, "Multiwfn")
    if os.path.exists(gui_path) and os.access(gui_path, os.X_OK):
        print("Using Multiwfn executable (with GUI)")
        return gui_path
    
    # Check if they're in the system PATH
    try:
        subprocess.run(["Multiwfn_noGUI", "--help"], capture_output=True, check=True)
        print("Using Multiwfn_noGUI from system PATH")
        return "Multiwfn_noGUI"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        subprocess.run(["Multiwfn", "--help"], capture_output=True, check=True)
        print("Using Multiwfn from system PATH")
        return "Multiwfn"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    raise FileNotFoundError(f"No Multiwfn executable found in {multiwfnpath} or system PATH")

def extract_resp_charges(orca_dir, output_dir, qm_atoms, system, args, top):
    """Extract RESP charges from QM calculation output"""
    print("Extracting RESP charges...")
    
    if args.qm_engine == "orca":
        extract_resp_charges_orca(orca_dir, output_dir, qm_atoms, system, args, top)
    elif args.qm_engine == "pyscf":
        extract_resp_charges_pyscf(orca_dir, output_dir, qm_atoms, system, args, top)

def extract_resp_charges_orca(orca_dir, output_dir, qm_atoms, system, args, top):
    """Extract RESP charges from ORCA calculation"""
    print("Extracting RESP charges from ORCA output...")
    
    # Generate Molden file from ORCA output
    orca_basename = os.path.join(orca_dir, "orca")
    molden_file = os.path.join(output_dir, "esp.molden")
    
    # Run orca_2mkl to convert to Molden format
    try:
        subprocess.run(["orca_2mkl", orca_basename, "-molden"], check=True)
        subprocess.run(["mv", f"{orca_basename}.molden.input", molden_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running orca_2mkl: {e}")
        return
    
    # Run Multiwfn for RESP fitting
    run_multiwfn_resp(molden_file, output_dir, args, qm_atoms, system, top)

def extract_resp_charges_pyscf(pyscf_dir, output_dir, qm_atoms, system, args, top):
    """Extract RESP charges from PySCF calculation using checkpoint file - Method 2"""
    print("Extracting RESP charges from PySCF checkpoint...")
    
    # Look for PySCF checkpoint file
    checkpoint_file = os.path.join(pyscf_dir, "pyscf.chk")
    
    if not os.path.exists(checkpoint_file):
        print(f"PySCF checkpoint file not found at {checkpoint_file}")
        # Fall back to Mulliken charges directly
        calculate_esp_directly_pyscf_inline(pyscf_dir, output_dir, qm_atoms, system, args, top)
        return
    
    try:
        # Import PySCF tools directly
        from pyscf.tools import chkfile_util, molden
        from pyscf.lib import chkfile
        
        molden_file = os.path.join(output_dir, "esp.molden")
        
        try:
            print("Trying molden.from_chkfile...")
            molden.from_chkfile(molden_file, checkpoint_file, key="scf/mo_coeff")
            print("Successfully converted checkpoint to molden using from_chkfile")
        except Exception as e2:
            print(f"molden.from_chkfile failed: {e2}")
            raise Exception("Both molden conversion methods failed")
        
        if os.path.exists(molden_file) and os.path.getsize(molden_file) > 0:
            print(f"Successfully created molden file: {molden_file}")
            # Run Multiwfn for RESP fitting
            run_multiwfn_resp(molden_file, output_dir, args, qm_atoms, system, top)
        else:
            raise Exception("Molden file was not created or is empty")
            
    except Exception as e:
        print(f"Error in PySCF checkpoint conversion: {e}")
        # Fall back to Mulliken charges
        calculate_esp_directly_pyscf_inline(pyscf_dir, output_dir, qm_atoms, system, args, top)

def calculate_esp_directly_pyscf_inline(pyscf_dir, output_dir, qm_atoms, system, args, top):
    """Calculate Mulliken charges directly without creating script files"""
    print("Attempting direct Mulliken charge extraction from PySCF checkpoint...")
    
    checkpoint_file = os.path.join(pyscf_dir, "pyscf.chk")
    
    try:
        # Import PySCF modules directly
        from pyscf import gto, scf
        from pyscf.lib import chkfile
        import numpy as np
        
        # Load molecular data from checkpoint
        mol_dict = chkfile.load(checkpoint_file, 'mol')
        scf_dict = chkfile.load(checkpoint_file, 'scf')
        
        # Reconstruct molecule object
        mol = gto.Mole()
        mol.build(
            atom=mol_dict['atom'],
            basis=mol_dict['basis'],
            charge=mol_dict.get('charge', 0),
            spin=mol_dict.get('spin', 0)
        )
        
        # Reconstruct SCF object
        mf = scf.RHF(mol)
        mf.mo_coeff = scf_dict['mo_coeff']
        mf.mo_energy = scf_dict['mo_energy']
        mf.mo_occ = scf_dict['mo_occ']
        
        # Calculate Mulliken charges
        mulliken_charges = mf.mulliken_charges()
        
        # Convert to list for compatibility
        resp_charges = mulliken_charges.tolist()
        
        print("Warning: Using Mulliken charges as fallback for RESP charges")
        print(f"Extracted {len(resp_charges)} Mulliken charges")
        
        # Save charges directly
        save_charges(resp_charges, qm_atoms, system, output_dir, top, args)
        
    except Exception as e:
        print(f"Error in direct charge extraction: {e}")
        print("Could not extract any charges from PySCF calculation")

def run_multiwfn_resp(molden_file, output_dir, args, qm_atoms, system, top):
    """Run Multiwfn for RESP fitting with automatic executable detection"""
    # Use absolute paths to avoid nested directory issues
    abs_output_dir = os.path.abspath(output_dir)
    abs_molden_file = os.path.abspath(molden_file)
    
    # Find the appropriate Multiwfn executable
    try:
        multiwfn_exe = find_multiwfn_executable(args.multiwfnpath)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Change to output directory for Multiwfn
    original_dir = os.getcwd()
    os.chdir(abs_output_dir)
    
    try:
        # Run Multiwfn for RESP fitting
        multiwfn_input = """7    ! Population analysis
18   ! RESP fitting
1    ! Start the fitting
y    ! Yes, continue
0    ! Return
0    ! Return
q    ! Quit
"""
        
        multiwfn_input_file = "multiwfn_input.txt"
        with open(multiwfn_input_file, "w") as f:
            f.write(multiwfn_input)
        
        multiwfn_log = "multiwfn.log"
        
        # Use the detected executable
        cmd = f"{multiwfn_exe} {os.path.basename(abs_molden_file)} < {multiwfn_input_file} > {multiwfn_log}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Parse and save RESP charges
        resp_charges = parse_multiwfn_resp("esp.chg")
        save_charges(resp_charges, qm_atoms, system, abs_output_dir, top, args)
        
    except subprocess.CalledProcessError as e:
        print("Warning: Multiwfn calculation failed")
        print(e)
    finally:
        os.chdir(original_dir)

def parse_multiwfn_resp(multiwfn_chg):
    """Parse RESP charges from Multiwfn output"""
    resp_charges = []
    
    try:
        with open(multiwfn_chg, 'r') as f:
            for line in f:
                try:
                    resp_charges.append(float(line.split()[-1]))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"No charges found in {multiwfn_chg}")
    
    return resp_charges

def update_prepi_charges(original_prepi_path, charge_data, output_prepi_path):
    """
    Update charges in an AMBER prepi file.
    
    Parameters:
        original_prepi_path: Path to the original prepi file
        charge_data: Dictionary mapping atom names to new charges
                    OR list of (atom_name, charge) tuples
        output_prepi_path: Path for the new prepi file
    """
    with open(original_prepi_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    atom_section = False
    
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
            
        # Check if we're in the atom section
        if line.strip().startswith('DUMM') or (len(line.split()) > 3 and line.split()[1] in ['DUMM', 'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']):
            atom_section = True
            
        if atom_section and not line.strip().startswith(('LOOP', 'IMPROPER', 'DONE')):
            parts = line.split()
            if len(parts) >= 3:
                atom_idx = int(parts[0])
                atom_name = parts[1]
                
                # Check if this is an actual atom line (not DUMM)
                if atom_idx >= 4:  # Real atoms start after DUMM atoms
                    # Find the atom in charge_data
                    new_charge = None
                    
                    if isinstance(charge_data, dict):
                        new_charge = charge_data.get(atom_name)
                    elif isinstance(charge_data, list):
                        for name, charge in charge_data:
                            if name == atom_name:
                                new_charge = charge
                                break
                    
                    if new_charge is not None:
                        # Replace the last field (charge)
                        parts[-1] = f"{new_charge:.6f}"
                        
                        # Reconstruct the line with proper spacing
                        # Keep original format using fixed width fields
                        new_line = f"{atom_idx:4d}  {atom_name:<6s}{parts[2]:<6s}{parts[3]:1s}"
                        new_line += f"{int(parts[4]):5d}{int(parts[5]):4d}{int(parts[6]):4d}"
                        new_line += f"{float(parts[7]):10.3f}{float(parts[8]):10.3f}{float(parts[9]):10.3f}{float(parts[-1]):10.6f}\n"
                        
                        new_lines.append(new_line)
                        continue
            
            new_lines.append(line)
        elif line.strip() in ('LOOP', 'IMPROPER', 'DONE'):
            atom_section = False
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    with open(output_prepi_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated prepi file created at {output_prepi_path}")

def save_charges(charges, qm_atoms, system, output_dir, top, args):
    """Save charges to output files"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping of atom names to charges
    charge_map = {}
    for i, atom_idx in enumerate(qm_atoms):
        if i < len(charges):
            atom = top.atom(atom_idx)
            charge_map[atom.name] = charges[i]

    # Print debug info
    print(f"Generated charge map with {len(charge_map)} entries:")
    for atom_name, charge in charge_map.items():
        print(f"  {atom_name}: {charge:.6f}")

    # Save charges to text file
    charge_file = os.path.join(output_dir, f"substrate_{args.qm_engine}_charges.txt")
    with open(charge_file, 'w') as f:
        f.write(f"# RESP charges calculated with {args.qm_engine.upper()}\n")
        f.write(f"# Functional: {args.functional}, Basis: {args.basis}\n")
        for atom_name, charge in charge_map.items():
            f.write(f"{atom_name:>6s} {charge:10.6f}\n")

    # Update prepi file if provided
    if args.prepi:
        substrate_name = top.atom(qm_atoms[0]).residue.name
        output_prepi = os.path.join(output_dir, f"{substrate_name}_{args.qm_engine}.prepi")
        update_prepi_charges(args.prepi, charge_map, output_prepi)
        print(f"RESP charges saved to {output_prepi}")

    print(f"RESP charges saved to {charge_file}")

if __name__ == "__main__":
    main()

