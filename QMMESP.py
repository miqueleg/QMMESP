#!/usr/bin/env python
"""
Environment-aware RESP charge derivation for substrates using ASH QM/MM
with ORCA quantum calculations and electrostatic embedding
"""

import os
import argparse
import subprocess
import ash
from ash import Fragment
from ash import ORCATheory
from ash import QMMMTheory
from ash import OpenMMTheory
import mdtraj as md

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="QM/MM RESP charge derivation with ORCA")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--prmtop", required=True, help="Input prmtop file")
    parser.add_argument("--inpcrd", required=True, help="Input inpcrd file")
    parser.add_argument("--resid", required=True, type=int, 
                      help="Residue ID of substrate")
    parser.add_argument("--charge", type=int, default=0, 
                      help="QM region charge")
    parser.add_argument("--mult", type=int, default=1, 
                      help="QM region multiplicity")
    parser.add_argument("--basis", default="6-31G*", 
                      help="Basis set (default: 6-31G*)")
    parser.add_argument("--functional", default="B3LYP", 
                      help="DFT functional (default: B3LYP)")
    parser.add_argument("--output", default="qmmm_resp", 
                      help="Output directory")
    parser.add_argument("--numcores", default=8,
                      help="Number of CPU cores assigned to the QM calculation")
    parser.add_argument("--orcadir", default=None,
                      help="Specify Orca installation directory")

    args = parser.parse_args()

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
    
    # Setup ORCA QM calculator
    qm_calc = ORCATheory(
        orcasimpleinput=f"!{args.functional} {args.basis} keepdens",
        numcores=args.numcores,
        orcadir=args.orcadir
    )
    
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
    print("Running QM/MM single-point calculation...")
    result = ash.Singlepoint(
        theory=qmmm,
        fragment=system,
        charge=args.charge,
        mult=args.mult
    )
    
    print(f"QM/MM Energy: {result.energy} Hartree")
    
    # Extract RESP charges
    extract_resp_charges("./", args.output, qm_atoms, system)
    
    print("QM/MM RESP calculation completed successfully!")

def extract_resp_charges(orca_dir, output_dir, qm_atoms, system):
    """Extract RESP charges from ORCA calculation"""
    print("Extracting RESP charges...")
    
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
    multiwfn_input = """7    ! Population analysis
18   ! RESP fitting
1    ! Start the fitting
y    ! Yes, continue
2    ! Use medium quality grid
1    ! Use restraint, RESP fitting
0    ! Save results and exit
q    ! Quit
"""
    
    multiwfn_input_file = os.path.join(output_dir, "multiwfn_input.txt")
    with open(multiwfn_input_file, "w") as f:
        f.write(multiwfn_input)
    
    multiwfn_log = os.path.join(output_dir, "multiwfn.log")
    try:
        subprocess.run(
            f"Multiwfn {molden_file} < {multiwfn_input_file} > {multiwfn_log}",
            shell=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print("Warning: Multiwfn calculation failed")
        print(e)
        return
    
    # Parse and save RESP charges
    resp_charges = parse_multiwfn_resp(multiwfn_log)
    save_charges(resp_charges, qm_atoms, system, output_dir)

def parse_multiwfn_resp(multiwfn_log):
    """Parse RESP charges from Multiwfn output"""
    resp_charges = []
    resp_section = False
    
    with open(multiwfn_log, 'r') as f:
        for line in f:
            if "RESP charges:" in line:
                resp_section = True
                next(f, None)  # Skip header
                continue
            
            if resp_section and "Sum" in line:
                resp_section = False
                continue
                
            if resp_section and len(line.strip()) > 0:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        resp_charge = float(parts[2])
                        resp_charges.append(resp_charge)
                    except (ValueError, IndexError):
                        continue
    
    return resp_charges

def save_charges(charges, qm_atoms, system, output_dir):
    """Save charges to output files"""
    # Plain text format
    resp_file = os.path.join(output_dir, "substrate_resp_charges.txt")
    with open(resp_file, "w") as f:
        f.write("# RESP charges for substrate atoms\n")
        f.write("# AtomIndex AtomName ResID RESPCharge\n")
        for i, atom_idx in enumerate(qm_atoms):
            atom = system.atoms[atom_idx]
            if i < len(charges):
                f.write(f"{atom_idx+1:5d} {atom.name:4s} {atom.resid:4d} {charges[i]:10.6f}\n")
    
    # AMBER mol2 format
    resname = system.atoms[qm_atoms[0]].resname
    amber_file = os.path.join(output_dir, "substrate_resp.mol2")
    with open(amber_file, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{resname}\n")
        f.write(f"{len(qm_atoms)} 0 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n\n")
        
        f.write("@<TRIPOS>ATOM\n")
        for i, atom_idx in enumerate(qm_atoms):
            atom = system.atoms[atom_idx]
            if i < len(charges):
                coords = system.coords[atom_idx]
                f.write(f"{i+1:5d} {atom.name:4s} {coords[0]:9.4f} {coords[1]:9.4f} "
                       f"{coords[2]:9.4f} {atom.element:4s} 1 {resname} {charges[i]:9.6f}\n")
    
    print(f"RESP charges saved to {resp_file} and {amber_file}")

if __name__ == "__main__":
    main()

