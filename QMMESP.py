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
    parser.add_argument("--prepi", help="Original prepi files")
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
    parser.add_argument('--multiwfnpath', type=str,
                    default=os.getenv('Multiwfnpath'),
                    help='Path to software (defaults to $Multiwfnpath)')

    args = parser.parse_args()

    if args.multiwfnpath is None:
        parser.error("No path provided via --path or $Multiwfnpath environment variable.")


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
    extract_resp_charges("./", args.output, qm_atoms, system, args, top)
    
    # We need to use the correct way to access atom information in ASH
    print("QM/MM RESP calculation completed successfully!")

def extract_resp_charges(orca_dir, output_dir, qm_atoms, system, args, top):
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
0    ! Return
0    ! Return
q    ! Quit
"""
    
    multiwfn_input_file = os.path.join(output_dir, "multiwfn_input.txt")
    with open(multiwfn_input_file, "w") as f:
        f.write(multiwfn_input)
    
    multiwfn_log = os.path.join(output_dir, "multiwfn.log")
    try:
        subprocess.run(
            f"{args.multiwfnpath}/Multiwfn_noGUI {molden_file} < {multiwfn_input_file} > {multiwfn_log}",
            shell=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print("Warning: Multiwfn calculation failed")
        print(e)
        return
    
    # Parse and save RESP charges
    resp_charges = parse_multiwfn_resp("esp.chg")
    save_charges(resp_charges, qm_atoms, system, output_dir, top, args)

def parse_multiwfn_resp(multiwfn_chg):
    """Parse RESP charges from Multiwfn output"""
    resp_charges = []
    
    with open(multiwfn_chg, 'r') as f:
        for line in f:
            try:
                resp_charges.append(float(line.split()[-1]))
            except:
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

    # Update prepi file if provided
    if args.prepi:
        substrate_name = top.atom(qm_atoms[0]).residue.name
        output_prepi = os.path.join(output_dir, f"{substrate_name}_resp.prepi")
        update_prepi_charges(args.prepi, charge_map, output_prepi)

    print(f"RESP charges saved to {output_prepi}")


if __name__ == "__main__":
    main()

