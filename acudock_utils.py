"""
AcuDock Utilities - Shared helper functions for AcuDock Pro notebook.

Provides protein preparation, ligand preparation, docking execution
(Vina + Gnina), consensus scoring, and visualization utilities.

Designed to be written to disk via %%writefile in Google Colab.
"""

import os
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import meeko

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Protein Preparation
# ---------------------------------------------------------------------------

def prepare_protein(pdb_id, output_dir='/content/acudock_pro'):
    """Fetch and prepare a protein structure for docking.

    Uses PDBFixer to:
      - Download structure from RCSB PDB
      - Fix missing residues and atoms
      - Replace non-standard residues
      - Remove heterogens (water, ions, ligands)
      - Add hydrogens at pH 7.4

    Returns path to the prepared PDB file.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    os.makedirs(output_dir, exist_ok=True)

    fixer = PDBFixer(pdbid=pdb_id)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    pdb_path = os.path.join(output_dir, f'{pdb_id}_prepared.pdb')
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return pdb_path


def pdb_to_pdbqt(pdb_path, output_path=None, is_receptor=True):
    """Convert PDB to PDBQT format using OpenBabel.

    Args:
        pdb_path: Path to input PDB file.
        output_path: Path for output PDBQT. If None, derived from input.
        is_receptor: If True, adds -xr flag for rigid receptor conversion.

    Returns path to the PDBQT file.
    """
    if output_path is None:
        output_path = pdb_path.replace('.pdb', '.pdbqt')

    cmd = ['obabel', pdb_path, '-O', output_path]
    if is_receptor:
        cmd.append('-xr')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f'OpenBabel warning: {result.stderr.strip()}')

    return output_path


# ---------------------------------------------------------------------------
# Ligand Preparation
# ---------------------------------------------------------------------------

def prepare_ligand(smiles, name='ligand', output_dir='/content/acudock_pro'):
    """Convert SMILES to 3D structure and PDBQT for docking.

    Pipeline: SMILES -> RDKit Mol -> 3D embed (ETKDGv3) -> MMFF optimize -> Meeko -> PDBQT

    Returns (pdbqt_path, rdkit_mol).
    """
    os.makedirs(output_dir, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    # Convert to PDBQT via Meeko
    preparator = meeko.MoleculePreparation()
    mol_setup_list = preparator.prepare(mol)
    pdbqt_string = meeko.PDBQTWriterLegacy.write_string(mol_setup_list[0])

    pdbqt_path = os.path.join(output_dir, f'{name}.pdbqt')
    with open(pdbqt_path, 'w') as f:
        f.write(pdbqt_string[0])

    return pdbqt_path, mol


def get_ligand_properties(smiles):
    """Calculate basic molecular properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MW': round(Descriptors.MolWt(mol), 1),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': round(Descriptors.TPSA(mol), 1),
    }


# ---------------------------------------------------------------------------
# Docking Engines
# ---------------------------------------------------------------------------

def run_vina(receptor_pdbqt, ligand_pdbqt, center, box_size,
             exhaustiveness=32, n_poses=20):
    """Run AutoDock Vina docking.

    Returns (vina_object, energies_array, poses_pdbqt_path).
    """
    from vina import Vina

    v = Vina(sf_name='vina')
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    energies = v.energies()

    poses_path = receptor_pdbqt.replace('_prepared.pdbqt', '_vina_poses.pdbqt')
    v.write_poses(poses_path, n_poses=n_poses, overwrite=True)

    return v, energies, poses_path


def run_gnina_rescore(receptor_pdbqt, poses_pdbqt, output_dir='/content/acudock_pro'):
    """Rescore Vina poses using Gnina CNN scoring.

    Gnina applies a convolutional neural network trained on PDBBind to
    re-evaluate binding poses. CNN scoring improves redocking success
    from ~58% (Vina) to ~73% (Gnina).

    Returns DataFrame with CNN scores, or None if Gnina unavailable.
    """
    gnina_path = '/usr/local/bin/gnina'
    if not os.path.isfile(gnina_path):
        print('Gnina not found. Install with:')
        print('  !wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O /usr/local/bin/gnina')
        print('  !chmod +x /usr/local/bin/gnina')
        return None

    output_sdf = os.path.join(output_dir, 'gnina_rescored.sdf')
    cmd = [
        gnina_path,
        '-r', receptor_pdbqt,
        '-l', poses_pdbqt,
        '-o', output_sdf,
        '--score_only',
        '--cnn_scoring', 'rescore',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f'Gnina error: {result.stderr}')
        return None

    # Parse Gnina output
    scores = []
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if 'CNNscore' in line or 'CNNaffinity' in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores.append({
                        'metric': parts[0].strip(':'),
                        'value': float(parts[1])
                    })
                except ValueError:
                    pass

    return scores


def consensus_score(vina_energies, gnina_scores=None, alpha=0.5):
    """Combine Vina and Gnina scores into a consensus ranking.

    Uses z-score normalization then weighted combination:
      consensus = alpha * z(vina) + (1-alpha) * z(gnina_cnn)

    Args:
        vina_energies: Array of Vina energies (kcal/mol).
        gnina_scores: List of Gnina CNN scores (higher = better).
        alpha: Weight for Vina scores (0-1). Default 0.5 (equal weight).

    Returns DataFrame with consensus rankings.
    """
    from scipy import stats

    df = pd.DataFrame({
        'Pose': range(1, len(vina_energies) + 1),
        'Vina_Score': [e[0] for e in vina_energies],
    })

    # Z-score normalize Vina (invert: more negative is better -> higher z = better)
    df['Vina_Z'] = stats.zscore(df['Vina_Score']) * -1

    if gnina_scores is not None and len(gnina_scores) == len(vina_energies):
        df['CNN_Score'] = gnina_scores
        df['CNN_Z'] = stats.zscore(df['CNN_Score'])
        df['Consensus'] = alpha * df['Vina_Z'] + (1 - alpha) * df['CNN_Z']
    else:
        df['Consensus'] = df['Vina_Z']

    df = df.sort_values('Consensus', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'

    return df


# ---------------------------------------------------------------------------
# Search Box
# ---------------------------------------------------------------------------

def get_binding_site_center(pdb_path, chain='A', residues=None):
    """Calculate center of mass of specified residues for the search box.

    If no residues given, uses the whole protein centroid (heavy atoms).
    Returns [x, y, z] in Angstroms.
    """
    from openmm.app import PDBFile

    pdb = PDBFile(pdb_path)
    positions = pdb.positions
    topology = pdb.topology

    coords = []
    for atom in topology.atoms():
        if residues is not None:
            if atom.residue.chain.id == chain and int(atom.residue.id) in residues:
                pos = positions[atom.index]
                coords.append([pos.x, pos.y, pos.z])
        else:
            if atom.element.symbol != 'H':
                pos = positions[atom.index]
                coords.append([pos.x, pos.y, pos.z])

    if not coords:
        raise ValueError('No matching atoms found.')

    center = np.array(coords).mean(axis=0) * 10  # nm -> Angstroms
    return center.tolist()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_pose(protein_pdb, poses_pdbqt, pose_index=0,
                   width=800, height=600):
    """Create a py3Dmol view of a docked pose in the protein.

    Returns a py3Dmol.view object (call .show() to render).
    """
    import py3Dmol

    with open(protein_pdb, 'r') as f:
        protein_data = f.read()
    with open(poses_pdbqt, 'r') as f:
        poses_data = f.read()

    # Extract specific pose from multi-model PDBQT
    models = poses_data.split('MODEL')
    if len(models) > 1:
        pose_data = 'MODEL' + models[pose_index + 1].split('ENDMDL')[0] + 'ENDMDL'
    else:
        pose_data = poses_data

    view = py3Dmol.view(width=width, height=height)
    view.addModel(protein_data, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum', 'opacity': 0.8}})

    view.addModel(pose_data, 'pdb')
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
    view.addSurface(py3Dmol.VDW, {'opacity': 0.25, 'color': 'green'}, {'model': 1})

    view.zoomTo({'model': 1})
    view.zoom(0.7)
    return view


def visualize_multi_poses(protein_pdb, poses_pdbqt, n_poses=3,
                          width=800, height=600):
    """Overlay multiple poses on the protein structure.

    Returns a py3Dmol.view object.
    """
    import py3Dmol

    with open(protein_pdb, 'r') as f:
        protein_data = f.read()
    with open(poses_pdbqt, 'r') as f:
        poses_data = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(protein_data, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'white', 'opacity': 0.5}})

    colors = ['green', 'cyan', 'magenta', 'orange', 'yellow']
    models = poses_data.split('MODEL')
    for i in range(min(n_poses, len(models) - 1)):
        pose = 'MODEL' + models[i + 1].split('ENDMDL')[0] + 'ENDMDL'
        view.addModel(pose, 'pdb')
        view.setStyle({'model': i + 1}, {'stick': {'color': colors[i % len(colors)], 'radius': 0.15}})

    view.zoomTo({'model': 1})
    view.zoom(0.7)
    return view


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def energies_to_dataframe(energies, ligand_name='Ligand'):
    """Convert Vina energies array to a DataFrame with estimated Kd."""
    R = 1.987e-3  # kcal/(mol*K)
    T = 298.15

    df = pd.DataFrame({
        'Pose': range(1, len(energies) + 1),
        'Name': ligand_name,
        'Score_kcal_mol': [e[0] for e in energies],
        'RMSD_lb': [e[1] for e in energies],
        'RMSD_ub': [e[2] for e in energies],
    })
    df['Est_Kd_uM'] = np.exp(df['Score_kcal_mol'] / (R * T)) * 1e6
    return df


def score_interpretation(score):
    """Return a human-readable interpretation of a Vina docking score."""
    if score > -5:
        return 'Very weak / no binding'
    elif score > -6:
        return 'Weak (mM range)'
    elif score > -7:
        return 'Moderate-weak'
    elif score > -8:
        return 'Moderate (low uM)'
    elif score > -9:
        return 'Good (sub-uM)'
    elif score > -10:
        return 'Strong (nM range)'
    else:
        return 'Very strong (verify - may be artifact)'
