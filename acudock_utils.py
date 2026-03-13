"""
AcuDock Utilities - Shared helper functions for AcuDock notebooks.

Provides protein preparation, ligand preparation, docking execution
(Vina + Gnina + Uni-Dock GPU), consensus scoring, visualization,
and interactive 3D viewer generation via 3Dmol.js.

Designed for use in Google Colab with ipywidgets interfaces.
"""

import os
import sys
import subprocess
import tempfile
import warnings
import base64
import uuid
import io
import math

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, RDConfig
from rdkit.Chem import QED as _QED_module
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import meeko

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Lazy singletons for expensive objects
# ---------------------------------------------------------------------------

_pains_catalog = None
_sa_scorer_fn = None


def _get_pains_catalog():
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog(params)
    return _pains_catalog


def _get_sa_scorer():
    global _sa_scorer_fn
    if _sa_scorer_fn is None:
        try:
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            from sascorer import calculateScore
            _sa_scorer_fn = calculateScore
        except Exception:
            _sa_scorer_fn = lambda mol: None
    return _sa_scorer_fn

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


def get_ligand_properties(smiles, docking_score=None):
    """Calculate molecular properties including drug-likeness metrics.

    Args:
        smiles: SMILES string.
        docking_score: Optional docking score (kcal/mol) for efficiency metrics.

    Returns dict with MW, LogP, HBD, HBA, RotBonds, TPSA, HeavyAtoms,
    QED, SA_Score, PAINS (list), PAINS_Count, and optionally LE, LLE.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    heavy = mol.GetNumHeavyAtoms()

    # PAINS alerts
    catalog = _get_pains_catalog()
    pains = []
    for entry in catalog.GetMatches(mol):
        pains.append(entry.GetDescription())

    # Synthetic accessibility
    sa_fn = _get_sa_scorer()
    sa_val = sa_fn(mol)

    props = {
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': round(Descriptors.TPSA(mol), 1),
        'HeavyAtoms': heavy,
        'QED': round(_QED_module.qed(mol), 3),
        'SA_Score': round(sa_val, 2) if sa_val is not None else None,
        'PAINS': pains,
        'PAINS_Count': len(pains),
    }

    if docking_score is not None and docking_score < 0:
        R, T = 1.987e-3, 298.15
        kd_M = math.exp(docking_score / (R * T))
        pKd = -math.log10(kd_M) if kd_M > 0 else 0
        props['LE'] = round(-docking_score / max(1, heavy), 3)
        props['LLE'] = round(pKd - logp, 2)

    return props


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
    if len(energies) == 0:
        raise RuntimeError(
            'Vina produced no poses. Check that the search box covers the '
            'binding site and that the ligand is valid.')

    poses_path = receptor_pdbqt.replace('_prepared.pdbqt', '_vina_poses.pdbqt')
    v.write_poses(poses_path, n_poses=n_poses, overwrite=True)

    return v, energies, poses_path


def run_gnina_rescore(receptor_pdbqt, poses_pdbqt, output_dir='/content/acudock_pro'):
    """Rescore Vina poses using Gnina CNN scoring.

    Gnina applies a convolutional neural network trained on PDBBind to
    re-evaluate binding poses. CNN scoring improves redocking success
    from ~58% (Vina) to ~73% (Gnina).

    Returns list of score dicts, or None if Gnina unavailable.
    """
    # Check common locations for the gnina binary
    gnina_path = None
    for candidate in ['/content/gnina', '/usr/local/bin/gnina']:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            gnina_path = candidate
            break

    if gnina_path is None:
        print('Gnina not found or not executable. Install with:')
        print('  !wget https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 -O /content/gnina')
        print('  !chmod +x /content/gnina')
        return None

    # Convert poses from PDBQT to SDF to avoid Gnina parse errors.
    # Meeko generates PDBQT with non-standard atom types that Gnina
    # cannot parse, but SDF is universally compatible.
    poses_sdf = os.path.join(output_dir, 'poses_for_gnina.sdf')
    conv = subprocess.run(
        ['obabel', poses_pdbqt, '-O', poses_sdf],
        capture_output=True, text=True, timeout=60
    )
    if conv.returncode != 0 or not os.path.exists(poses_sdf):
        print(f'OpenBabel PDBQT->SDF conversion failed: {conv.stderr}')
        # Fall back to trying PDBQT directly
        ligand_input = poses_pdbqt
    else:
        ligand_input = poses_sdf

    output_sdf = os.path.join(output_dir, 'gnina_rescored.sdf')
    cmd = [
        gnina_path,
        '-r', receptor_pdbqt,
        '-l', ligand_input,
        '-o', output_sdf,
        '--score_only',
        '--cnn_scoring', 'rescore',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f'Gnina error: {result.stderr[:500]}')
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


# Common non-ligand heterogens to exclude during binding site detection
_EXCLUDED_HETEROGENS = {
    'HOH', 'WAT', 'NA', 'CL', 'MG', 'ZN', 'CA', 'K', 'MN', 'FE', 'CO',
    'NI', 'CU', 'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'DMS', 'BME', 'IOD',
    'MPD', 'PEG', 'PGE', 'EPE', 'TRS', 'MES', 'CIT', 'FMT', 'NH4',
}


def detect_binding_site(pdb_id, output_dir='/content'):
    """Auto-detect the binding site from co-crystallized ligands in a PDB structure.

    Downloads the original PDB (before preparation removes heterogens),
    finds the largest non-solvent heterogen, and returns its centroid
    plus nearby protein residues.

    Returns dict with keys:
        center: [x, y, z] in Angstroms, or None if no ligand found
        het_name: 3-letter code of detected ligand
        het_atoms: number of heavy atoms in the ligand
        nearby_residues: list of residue numbers within 5A
        method: 'heterogen' or 'none'
    """
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    raw_pdb = os.path.join(output_dir, f'{pdb_id}_raw.pdb')

    # Download original PDB
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    try:
        urllib.request.urlretrieve(url, raw_pdb)
    except Exception as e:
        print(f'Could not download PDB {pdb_id}: {e}')
        return {'center': None, 'method': 'none'}

    # Parse HETATM records
    het_groups = {}  # res_name -> list of (x, y, z)
    protein_atoms = []  # list of (x, y, z, res_id)

    with open(raw_pdb, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name in _EXCLUDED_HETEROGENS:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    het_groups.setdefault(res_name, []).append((x, y, z))
                except (ValueError, IndexError):
                    pass
            elif line.startswith('ATOM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_id = int(line[22:26].strip())
                    protein_atoms.append((x, y, z, res_id))
                except (ValueError, IndexError):
                    pass

    if not het_groups:
        return {'center': None, 'method': 'none'}

    # Pick the largest heterogen group
    best_het = max(het_groups, key=lambda k: len(het_groups[k]))
    coords = np.array(het_groups[best_het])
    center = coords.mean(axis=0).tolist()

    # Find protein residues within 5A of the ligand
    nearby = set()
    if protein_atoms:
        prot_arr = np.array([(a[0], a[1], a[2]) for a in protein_atoms])
        for hx, hy, hz in coords:
            dists = np.sqrt(((prot_arr - np.array([hx, hy, hz])) ** 2).sum(axis=1))
            close_idx = np.where(dists < 5.0)[0]
            for idx in close_idx:
                nearby.add(protein_atoms[idx][3])

    return {
        'center': center,
        'het_name': best_het,
        'het_atoms': len(het_groups[best_het]),
        'nearby_residues': sorted(nearby),
        'method': 'heterogen',
    }


# ---------------------------------------------------------------------------
# Protein-Ligand Interaction Fingerprints (ProLIF)
# ---------------------------------------------------------------------------

def compute_interaction_fingerprint(protein_pdb, poses_pdbqt, pose_index=0,
                                     smiles=None):
    """Compute protein-ligand interaction fingerprint using ProLIF.

    Args:
        protein_pdb: Path to prepared protein PDB.
        poses_pdbqt: Path to multi-model PDBQT poses file.
        pose_index: Which pose to analyze (0 = best).
        smiles: Ligand SMILES (used for bond order assignment).

    Returns:
        (summary_df, interaction_counts) or (None, None) if ProLIF unavailable.

        summary_df: DataFrame with columns [Residue, Type] listing each
                    detected interaction.
        interaction_counts: dict mapping interaction type -> count.
    """
    try:
        import prolif
        import MDAnalysis as mda
    except ImportError:
        return None, None

    try:
        # Load protein
        prot = mda.Universe(protein_pdb)
        prot_mol = prolif.Molecule.from_mda(prot)

        # Extract pose and write temp PDB
        pose_data = extract_pose_from_pdbqt(poses_pdbqt, pose_index)
        clean_data = _clean_pdbqt_for_viewer(pose_data)
        tmp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w')
        tmp_pdb.write(clean_data)
        tmp_pdb.close()

        lig_u = mda.Universe(tmp_pdb.name)
        lig_mol = prolif.Molecule.from_mda(lig_u)

        # Compute fingerprint
        fp = prolif.Fingerprint()
        fp.run_from_iterable([lig_mol], prot_mol)
        df = fp.to_dataframe()

        os.unlink(tmp_pdb.name)

        # Parse into summary
        interactions = []
        counts = {}
        if not df.empty:
            for col in df.columns:
                if df[col].any():
                    # Column format: (ligand_resname, protein_resname, interaction_type)
                    if isinstance(col, tuple) and len(col) >= 3:
                        res = str(col[1])
                        itype = str(col[2])
                    else:
                        res = str(col)
                        itype = 'Unknown'
                    interactions.append({'Residue': res, 'Type': itype})
                    counts[itype] = counts.get(itype, 0) + 1

        summary_df = pd.DataFrame(interactions) if interactions else pd.DataFrame(
            columns=['Residue', 'Type'])
        return summary_df, counts

    except Exception as e:
        print(f'ProLIF interaction analysis failed: {e}')
        return None, None


def interaction_fingerprint_to_html(summary_df):
    """Convert interaction summary DataFrame to a styled HTML table."""
    if summary_df is None or summary_df.empty:
        return '<p><em>No interactions detected.</em></p>'

    type_colors = {
        'HBDonor': '#2196F3', 'HBAcceptor': '#03A9F4',
        'Hydrophobic': '#4CAF50', 'PiStacking': '#9C27B0',
        'PiCation': '#E91E63', 'SaltBridge': '#F44336',
        'Anionic': '#FF5722', 'Cationic': '#FF9800',
        'FaceToFace': '#9C27B0', 'EdgeToFace': '#7B1FA2',
    }

    rows = []
    for _, r in summary_df.iterrows():
        color = type_colors.get(r['Type'], '#607D8B')
        rows.append(
            f'<tr><td>{r["Residue"]}</td>'
            f'<td><span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:3px;">{r["Type"]}</span></td></tr>'
        )

    return (
        '<table style="border-collapse:collapse;font-size:13px;">'
        '<tr style="background:#f5f5f5;"><th style="padding:4px 12px;">Residue</th>'
        '<th style="padding:4px 12px;">Interaction</th></tr>'
        + ''.join(rows) + '</table>'
    )


# ---------------------------------------------------------------------------
# 3D View Capture (for PDF reports)
# ---------------------------------------------------------------------------

def capture_3d_views_matplotlib(protein_pdb, poses_pdbqt, output_dir,
                                 pose_index=0, prefix='pose'):
    """Render 6-axis 3D views of a docked pose using matplotlib.

    Creates simple stick-model renderings from +X, -X, +Y, -Y, +Z, -Z.
    Returns dict mapping orientation name to PNG file path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Parse protein atoms (CA only for speed)
    prot_coords = []
    with open(protein_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    prot_coords.append((x, y, z))
                except (ValueError, IndexError):
                    pass

    # Parse ligand atoms
    pose_data = extract_pose_from_pdbqt(poses_pdbqt, pose_index)
    clean = _clean_pdbqt_for_viewer(pose_data)
    lig_coords = []
    lig_elements = []
    for line in clean.split('\n'):
        if line.startswith(('ATOM', 'HETATM')):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) > 76 else line[12:14].strip()
                if not elem:
                    elem = line[12:16].strip()[0]
                lig_coords.append((x, y, z))
                lig_elements.append(elem.upper())
            except (ValueError, IndexError):
                pass

    if not lig_coords:
        return {}

    prot_arr = np.array(prot_coords) if prot_coords else np.empty((0, 3))
    lig_arr = np.array(lig_coords)

    elem_colors = {
        'C': '#808080', 'N': '#3050F8', 'O': '#FF0D0D', 'S': '#FFFF30',
        'H': '#FFFFFF', 'F': '#90E050', 'CL': '#1FF01F', 'BR': '#A62929',
        'P': '#FF8000', 'I': '#940094',
    }

    # 6 orientations: (elevation, azimuth, label)
    orientations = [
        (0, 0, 'front'), (0, 180, 'back'),
        (0, 90, 'right'), (0, -90, 'left'),
        (90, 0, 'top'), (-90, 0, 'bottom'),
    ]

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Center on ligand
    lig_center = lig_arr.mean(axis=0)
    lig_range = max(lig_arr.max(axis=0) - lig_arr.min(axis=0)) * 0.7 + 5

    for elev, azim, name in orientations:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')

        # Set view angle FIRST before adding data
        ax.view_init(elev=elev, azim=azim)

        # Draw nearby protein backbone
        if len(prot_arr) > 0:
            dists = np.sqrt(((prot_arr - lig_center) ** 2).sum(axis=1))
            nearby = prot_arr[dists < 15]
            if len(nearby) > 1:
                ax.plot(nearby[:, 0], nearby[:, 1], nearby[:, 2],
                        'o-', color='#cccccc', markersize=1, linewidth=0.5, alpha=0.4)

        # Draw ligand atoms
        for i, (x, y, z) in enumerate(lig_coords):
            elem = lig_elements[i] if i < len(lig_elements) else 'C'
            color = elem_colors.get(elem, '#FF69B4')
            ax.scatter(x, y, z, c=color, s=40, edgecolors='black', linewidths=0.3)

        # Draw ligand bonds (atoms within 1.9A)
        for i in range(len(lig_arr)):
            for j in range(i + 1, len(lig_arr)):
                dist = np.linalg.norm(lig_arr[i] - lig_arr[j])
                if dist < 1.9:
                    ax.plot([lig_arr[i, 0], lig_arr[j, 0]],
                            [lig_arr[i, 1], lig_arr[j, 1]],
                            [lig_arr[i, 2], lig_arr[j, 2]],
                            color='#404040', linewidth=1.5)

        ax.set_xlim(lig_center[0] - lig_range, lig_center[0] + lig_range)
        ax.set_ylim(lig_center[1] - lig_range, lig_center[1] + lig_range)
        ax.set_zlim(lig_center[2] - lig_range, lig_center[2] + lig_range)
        # Re-apply view angle after setting limits (matplotlib can reset it)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(name.capitalize(), fontsize=10, pad=-5)

        path = os.path.join(output_dir, f'{prefix}_{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths[name] = path

    return paths


def capture_3d_views_js(protein_pdb_data, ligand_pdb_data):
    """Generate JavaScript code for 6-axis 3Dmol.js PNG capture in Colab.

    Returns a JS string that, when executed via google.colab.output.eval_js(),
    produces a JSON dict mapping orientation names to base64 PNG data.
    Use this from within a notebook cell.
    """
    prot_b64 = base64.b64encode(protein_pdb_data.encode()).decode()
    lig_b64 = base64.b64encode(ligand_pdb_data.encode()).decode()

    js = f"""
    (async function() {{
        await new Promise(r => {{
            if (window.$3Dmol) {{ r(); return; }}
            var s = document.createElement('script');
            s.src = 'https://3Dmol.org/build/3Dmol-min.js';
            s.onload = r;
            document.head.appendChild(s);
        }});

        var div = document.createElement('div');
        div.style.width = '600px';
        div.style.height = '600px';
        div.style.position = 'fixed';
        div.style.left = '-9999px';
        document.body.appendChild(div);

        var v = $3Dmol.createViewer(div, {{backgroundColor: 'white'}});
        v.addModel(atob("{prot_b64}"), "pdb");
        v.setStyle({{model:0}}, {{cartoon:{{color:'spectrum',opacity:0.8}}}});
        v.addModel(atob("{lig_b64}"), "pdb");
        v.setStyle({{model:1}}, {{stick:{{colorscheme:'greenCarbon',radius:0.2}}}});
        v.zoomTo({{model:1}});
        v.zoom(0.7);
        v.render();

        var orientations = [
            [0, 0, 1, 'front'], [0, 0, -1, 'back'],
            [1, 0, 0, 'right'], [-1, 0, 0, 'left'],
            [0, 1, 0, 'top'], [0, -1, 0, 'bottom']
        ];

        var results = {{}};
        for (var o of orientations) {{
            v.setCameraParameters({{direction: {{x:o[0], y:o[1], z:o[2]}}}});
            v.render();
            await new Promise(r => setTimeout(r, 200));
            var uri = v.pngURI();
            results[o[3]] = uri.split(',')[1];
        }}

        document.body.removeChild(div);
        return JSON.stringify(results);
    }})()
    """
    return js


def capture_3d_views(protein_pdb, poses_pdbqt, output_dir,
                      pose_index=0, prefix='pose'):
    """Capture 6-axis 3D views using matplotlib.

    Returns dict mapping orientation name to PNG file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = capture_3d_views_matplotlib(
        protein_pdb, poses_pdbqt, output_dir,
        pose_index=pose_index, prefix=prefix
    )
    if paths:
        print(f'  Rendered {len(paths)} 3D views via matplotlib')
    return paths


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

def mount_google_drive(mount_point='/content/drive'):
    """Mount Google Drive in Colab. Returns True on success."""
    try:
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        return True
    except Exception:
        return False


def ensure_drive_dir(base='/content/drive/MyDrive/AcuDock'):
    """Create AcuDock directory on Drive. Returns path or None."""
    try:
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _clean_pdbqt_for_viewer(pdbqt_data):
    """Convert PDBQT string to PDB-compatible string for py3Dmol.

    PDBQT files contain extra columns (partial charges, AutoDock atom
    types) and keywords (ROOT, BRANCH, ENDBRANCH, TORSDOF) that
    py3Dmol's PDB parser cannot handle for many molecules. This
    strips those extras so the data parses as valid PDB.
    """
    lines = []
    for line in pdbqt_data.split('\n'):
        # Skip PDBQT-specific directives
        if line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF')):
            continue
        # Trim ATOM/HETATM to 66 chars (PDB standard) to drop charge + type cols
        if line.startswith(('ATOM', 'HETATM')):
            lines.append(line[:66])
        else:
            lines.append(line)
    return '\n'.join(lines)


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

    view.addModel(_clean_pdbqt_for_viewer(pose_data), 'pdb')
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
        view.addModel(_clean_pdbqt_for_viewer(pose), 'pdb')
        view.setStyle({'model': i + 1}, {'stick': {'color': colors[i % len(colors)], 'radius': 0.15}})

    view.zoomTo({'model': 1})
    view.zoom(0.7)
    return view


def display_3d_viewer(view):
    """Display a py3Dmol view reliably inside ipywidgets Output contexts.

    Uses an iframe with data URI to bypass JavaScript injection issues
    that occur when py3Dmol runs inside nested Output widgets in Colab.
    """
    from IPython.display import display, HTML

    html_content = view._make_html()
    b64 = base64.b64encode(html_content.encode()).decode()
    w = getattr(view, 'width', 800)
    h = getattr(view, 'height', 600)
    display(HTML(
        f'<iframe src="data:text/html;base64,{b64}" '
        f'width="{w + 20}" height="{h + 20}" '
        f'style="border:1px solid #ddd; border-radius:4px;"></iframe>'
    ))


def create_download_link(file_path, description=None):
    """Create a Colab-compatible download link using a data URI.

    Works inside ipywidgets Output widgets where FileLink does not.
    For large files (>10 MB), falls back to google.colab.files.download().
    """
    from IPython.display import display, HTML

    filename = os.path.basename(file_path)
    desc = description or f'Download {filename}'

    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024:
        # Large file: use Colab download API
        try:
            from google.colab import files
            files.download(file_path)
            return
        except ImportError:
            pass

    with open(file_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.csv': 'text/csv', '.pdf': 'application/pdf',
        '.png': 'image/png', '.txt': 'text/plain',
    }
    mime = mime_types.get(ext, 'application/octet-stream')

    display(HTML(
        f'<a href="data:{mime};base64,{data}" download="{filename}" '
        f'style="display:inline-block; padding:8px 16px; background:#1565C0; '
        f'color:white; text-decoration:none; border-radius:4px; margin:4px 0;">'
        f'{desc}</a>'
    ))


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


# ---------------------------------------------------------------------------
# GPU Detection & Uni-Dock Integration
# ---------------------------------------------------------------------------

def check_gpu_available():
    """Check if an NVIDIA GPU is available (for Uni-Dock)."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup_unidock_env(prefix='/content/unidock_env'):
    """No-op kept for backward compatibility with older notebook versions."""
    return False


def check_unidock_available():
    """Check if the Uni-Dock binary is on PATH and executable."""
    try:
        result = subprocess.run(
            ['unidock', '--help'], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_unidock_colab(dest='/usr/local/bin/unidock'):
    """Install Uni-Dock GPU binary on Google Colab.

    Downloads the pre-built CUDA 12 binary (10 MB) from the official
    dptech-corp release. Requires NVIDIA GPU with compute capability >= 7.0.

    Returns True on success, False on failure.
    """
    url = ('https://github.com/dptech-corp/Uni-Dock/releases/download/'
           '1.1.0/unidock-1.1.0-cuda120-linux-x86_64')
    try:
        result = subprocess.run(
            ['wget', '-q', url, '-O', dest],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f'Uni-Dock download failed: {result.stderr}')
            return False
        os.chmod(dest, 0o755)
        return check_unidock_available()
    except Exception as e:
        print(f'Uni-Dock install error: {e}')
        return False


def run_unidock(receptor_pdbqt, ligand_pdbqt_files, center, box_size,
                exhaustiveness=32, num_modes=9, output_dir='.'):
    """Run Uni-Dock GPU-accelerated batch docking.

    Uni-Dock achieves >1000x speedup over Vina on GPU by docking many
    ligands in parallel. Uses the same PDBQT format as Vina.

    IMPORTANT: Uni-Dock is designed for batch screening (10+ ligands).
    For single-ligand docking, use run_vina() instead — Uni-Dock falls
    back to CPU mode with a single ligand.

    Args:
        receptor_pdbqt: Path to receptor PDBQT.
        ligand_pdbqt_files: List of paths to ligand PDBQT files (10+ recommended).
        center: [x, y, z] center coordinates in Angstroms.
        box_size: [x, y, z] box dimensions in Angstroms.
        exhaustiveness: Search thoroughness (default 32, higher = more thorough).
        num_modes: Max binding poses per ligand (default 9).
        output_dir: Directory for output files.

    Returns:
        dict mapping input basename -> list of (score, rmsd_lb, rmsd_ub).

    Raises:
        RuntimeError: If Uni-Dock is not available or fails.
    """
    if not check_unidock_available():
        raise RuntimeError(
            'Uni-Dock not available. Install the GPU binary from: '
            'https://github.com/dptech-corp/Uni-Dock/releases'
        )

    result_dir = os.path.join(output_dir, 'unidock_results')
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        'unidock',
        '--receptor', receptor_pdbqt,
        '--gpu_batch',
    ] + ligand_pdbqt_files + [
        '--scoring', 'vina',
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(box_size[0]),
        '--size_y', str(box_size[1]),
        '--size_z', str(box_size[2]),
        '--exhaustiveness', str(exhaustiveness),
        '--num_modes', str(num_modes),
        '--dir', result_dir,
        '--seed', '42',
    ]

    print(f'[Uni-Dock] Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.stdout:
        print(f'[Uni-Dock] stdout: {result.stdout[:500]}')
    if result.stderr:
        print(f'[Uni-Dock] stderr: {result.stderr[:500]}')
    if result.returncode != 0:
        raise RuntimeError(f'Uni-Dock failed (exit code {result.returncode}): {result.stderr}')

    # Parse results from output PDBQT files
    results = {}
    for lig_file in ligand_pdbqt_files:
        basename = os.path.basename(lig_file)
        out_file = os.path.join(result_dir, basename.replace('.pdbqt', '_out.pdbqt'))
        if not os.path.exists(out_file):
            out_file = os.path.join(result_dir, basename)
        if os.path.exists(out_file):
            scores = parse_pdbqt_scores(out_file)
            results[basename] = scores
        else:
            print(f'[Uni-Dock] Warning: no output file for {basename}')

    return results


def run_unidock_single(receptor_pdbqt, ligand_pdbqt, center, box_size,
                       exhaustiveness=32, num_modes=20, output_dir='.'):
    """Run Uni-Dock for a single ligand (convenience wrapper).

    Returns (energies_list, poses_pdbqt_path) matching Vina output format.
    Returns ([], None) if docking fails or produces implausible results.
    """
    result_dir = os.path.join(output_dir, 'unidock_results')
    os.makedirs(result_dir, exist_ok=True)

    results = run_unidock(
        receptor_pdbqt, [ligand_pdbqt],
        center=center, box_size=box_size,
        exhaustiveness=exhaustiveness, num_modes=num_modes,
        output_dir=output_dir
    )

    basename = os.path.basename(ligand_pdbqt)
    scores = results.get(basename, [])

    # Build energies in Vina format: [(score, rmsd_lb, rmsd_ub), ...]
    energies = [(s, r1, r2) for s, r1, r2 in scores] if scores else []

    # Validate: scores should be negative for successful docking.
    # A positive score (e.g. +103) means the GPU search failed.
    if energies and energies[0][0] > 0:
        print(f'[Uni-Dock] Warning: best score is {energies[0][0]:.2f} kcal/mol '
              f'(positive = failed search). Results unreliable.')
        return [], None

    if not energies:
        print('[Uni-Dock] Warning: no poses generated.')
        return [], None

    # Locate output poses
    out_file = os.path.join(result_dir, basename.replace('.pdbqt', '_out.pdbqt'))
    if not os.path.exists(out_file):
        out_file = os.path.join(result_dir, basename)

    return energies, out_file


def parse_pdbqt_scores(pdbqt_path):
    """Parse Vina/Uni-Dock scores from a PDBQT output file.

    Looks for REMARK VINA RESULT lines:
        REMARK VINA RESULT:    -8.2      0.000      0.000

    Returns list of (score, rmsd_lb, rmsd_ub) tuples.
    """
    scores = []
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if 'VINA RESULT' in line:
                parts = line.split()
                try:
                    score = float(parts[3])
                    rmsd_lb = float(parts[4]) if len(parts) > 4 else 0.0
                    rmsd_ub = float(parts[5]) if len(parts) > 5 else 0.0
                    scores.append((score, rmsd_lb, rmsd_ub))
                except (IndexError, ValueError):
                    pass
    return scores


def extract_pose_from_pdbqt(poses_pdbqt_path, pose_index=0):
    """Extract a single pose from a multi-model PDBQT file.

    Returns the pose data as a string.
    """
    with open(poses_pdbqt_path, 'r') as f:
        data = f.read()

    models = data.split('MODEL')
    if len(models) > 1 and pose_index + 1 < len(models):
        return 'MODEL' + models[pose_index + 1].split('ENDMDL')[0] + 'ENDMDL'
    return data


# ---------------------------------------------------------------------------
# 3D Viewer HTML
# ---------------------------------------------------------------------------

def make_3d_viewer_html(protein_pdb_data, ligand_data=None,
                        width='100%', height='500px',
                        protein_style='cartoon', ligand_style='stick'):
    """Generate an HTML string with an interactive 3Dmol.js viewer.

    Uses the 3Dmol.js library loaded from CDN. Data is base64-encoded
    to avoid escaping issues. Returns a self-contained HTML snippet
    wrapped in an iframe for use with IPython.display.HTML() or
    ipywidgets.HTML().

    Args:
        protein_pdb_data: Protein PDB file contents as string.
        ligand_data: Optional ligand PDB/PDBQT data as string.
        width: Viewer width (CSS value).
        height: Viewer height (CSS value).
        protein_style: 'cartoon' or 'stick' for protein rendering.
        ligand_style: 'stick' or 'sphere' for ligand rendering.

    Returns HTML string.
    """
    vid = 'viewer_' + uuid.uuid4().hex[:8]
    prot_b64 = base64.b64encode(protein_pdb_data.encode()).decode()

    js_lines = [
        f'var el=document.getElementById("{vid}");',
        'var v=$3Dmol.createViewer(el,{backgroundColor:"white"});',
        f'v.addModel(atob("{prot_b64}"),"pdb");',
    ]

    if protein_style == 'cartoon':
        js_lines.append(
            'v.setStyle({model:0},{cartoon:{color:"spectrum",opacity:0.8}});'
        )
    else:
        js_lines.append(
            'v.setStyle({model:0},{stick:{colorscheme:"whiteCarbon",radius:0.1}});'
        )

    if ligand_data:
        lig_b64 = base64.b64encode(ligand_data.encode()).decode()
        js_lines.append(f'v.addModel(atob("{lig_b64}"),"pdb");')
        if ligand_style == 'stick':
            js_lines.append(
                'v.setStyle({model:1},{stick:{colorscheme:"greenCarbon",radius:0.2}});'
            )
        else:
            js_lines.append(
                'v.setStyle({model:1},{sphere:{scale:0.3,colorscheme:"greenCarbon"}});'
            )
        js_lines.append('v.zoomTo({model:1});v.zoom(0.7);')
    else:
        js_lines.append('v.zoomTo();')

    js_lines.append('v.render();')

    js_block = '\n'.join(js_lines)

    # Wrap in an iframe srcdoc so 3Dmol.js scripts execute in an
    # isolated context. Works with IPython.display.HTML() and
    # ipywidgets.HTML().
    inner_html = (
        '<!DOCTYPE html><html><head>'
        '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'
        '</head><body style="margin:0;padding:0;">'
        f'<div id="{vid}" style="width:100%;height:100%;position:relative;"></div>'
        f'<script>(function(){{{js_block}}})()</script>'
        '</body></html>'
    )
    escaped = inner_html.replace('&', '&amp;').replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped}" style="width:{width};height:{height};border:none;" sandbox="allow-scripts allow-same-origin"></iframe>'


def visualize_poses(protein_pdb_data, poses_pdbqt_data, n_poses=3,
                    width='100%', height='500px'):
    """Generate HTML overlaying multiple docked poses on the protein.

    Args:
        protein_pdb_data: Protein PDB file contents.
        poses_pdbqt_data: Multi-model PDBQT file contents.
        n_poses: Number of top poses to overlay.

    Returns HTML string.
    """
    vid = 'viewer_' + uuid.uuid4().hex[:8]
    prot_b64 = base64.b64encode(protein_pdb_data.encode()).decode()

    colors = ['green', 'cyan', 'magenta', 'orange', 'yellow']
    models = poses_pdbqt_data.split('MODEL')

    js_lines = [
        f'var el=document.getElementById("{vid}");',
        'var v=$3Dmol.createViewer(el,{backgroundColor:"white"});',
        f'v.addModel(atob("{prot_b64}"),"pdb");',
        'v.setStyle({model:0},{cartoon:{color:"white",opacity:0.5}});',
    ]

    for i in range(min(n_poses, len(models) - 1)):
        pose = 'MODEL' + models[i + 1].split('ENDMDL')[0] + 'ENDMDL'
        pose_b64 = base64.b64encode(pose.encode()).decode()
        color = colors[i % len(colors)]
        js_lines.append(f'v.addModel(atob("{pose_b64}"),"pdb");')
        js_lines.append(
            f'v.setStyle({{model:{i + 1}}},{{stick:{{color:"{color}",radius:0.15}}}});'
        )

    js_lines.append('v.zoomTo({model:1});v.zoom(0.7);v.render();')
    js_block = '\n'.join(js_lines)

    inner_html = (
        '<!DOCTYPE html><html><head>'
        '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'
        '</head><body style="margin:0;padding:0;">'
        f'<div id="{vid}" style="width:100%;height:100%;position:relative;"></div>'
        f'<script>(function(){{{js_block}}})()</script>'
        '</body></html>'
    )
    escaped = inner_html.replace('&', '&amp;').replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped}" style="width:{width};height:{height};border:none;" sandbox="allow-scripts allow-same-origin"></iframe>'


def get_docking_engine_status():
    """Return a status summary of available docking engines."""
    engines = ['Vina (CPU): Available']
    gpu = check_gpu_available()
    if gpu:
        try:
            smi = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            gpu_info = smi.stdout.strip()
            engines.append(f'GPU: {gpu_info}')
        except Exception:
            engines.append('GPU: Detected (NVIDIA)')
    else:
        engines.append('GPU: Not detected')
    if check_unidock_available():
        engines.append('Uni-Dock (GPU): Available')
    else:
        engines.append('Uni-Dock (GPU): Not installed')
    return '\n'.join(engines)


# ---------------------------------------------------------------------------
# Multi-Protein Docking (1 ligand x N proteins)
# ---------------------------------------------------------------------------

def _dock_single_protein(args):
    """Worker function for parallel multi-protein docking.

    Designed to be called via concurrent.futures. Prepares one protein
    and docks the pre-prepared ligand against it.

    Args:
        args: Tuple of (pdb_id, ligand_pdbqt, residues, box, exhaustiveness,
              n_poses, output_dir, use_unidock).

    Returns:
        dict with docking results for this protein.
    """
    import time

    pdb_id, ligand_pdbqt, residues, box, exhaustiveness, n_poses, output_dir, use_unidock = args
    R, T = 1.987e-3, 298.15

    row = {'PDB_ID': pdb_id, 'Best_Score': None, 'Est_Kd_uM': None,
           'Interpretation': 'Failed', 'Num_Poses': 0,
           'Center': '', 'Prep_Time_s': 0, 'Dock_Time_s': 0,
           'Engine': 'Vina', 'Error': ''}

    try:
        # Prepare protein
        t0 = time.time()
        pdb_dir = os.path.join(output_dir, pdb_id)
        protein_pdb = prepare_protein(pdb_id, output_dir=pdb_dir)
        receptor_pdbqt = pdb_to_pdbqt(protein_pdb)
        row['Prep_Time_s'] = round(time.time() - t0, 1)

        # Binding site center (auto-detect if no residues given)
        if residues:
            center = get_binding_site_center(protein_pdb, chain='A',
                                             residues=residues)
        else:
            detected = detect_binding_site(pdb_id, output_dir=pdb_dir)
            if detected.get('center'):
                center = detected['center']
            else:
                center = get_binding_site_center(protein_pdb, chain='A',
                                                 residues=None)
        row['Center'] = f'[{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]'

        # Dock — try Uni-Dock first if requested, fall back to Vina
        t1 = time.time()
        energies = []
        poses_path = None

        if use_unidock:
            try:
                energies, poses_path = run_unidock_single(
                    receptor_pdbqt, ligand_pdbqt,
                    center=center, box_size=box,
                    exhaustiveness=int(exhaustiveness),
                    num_modes=int(n_poses),
                    output_dir=pdb_dir
                )
                if energies:
                    row['Engine'] = 'Uni-Dock'
            except Exception:
                energies = []
                poses_path = None

        if not energies:
            _, energies, poses_path = run_vina(
                receptor_pdbqt, ligand_pdbqt,
                center=center, box_size=box,
                exhaustiveness=int(exhaustiveness), n_poses=int(n_poses)
            )
            row['Engine'] = 'Vina (fallback)' if use_unidock else 'Vina'

        row['Dock_Time_s'] = round(time.time() - t1, 1)

        if len(energies) > 0:
            score = energies[0][0]
            row['Best_Score'] = round(score, 2)
            row['Est_Kd_uM'] = round(np.exp(score / (R * T)) * 1e6, 4)
            row['Interpretation'] = score_interpretation(score)
            row['Num_Poses'] = len(energies)

        row['_protein_pdb'] = protein_pdb
        row['_poses_path'] = poses_path

    except Exception as e:
        row['Error'] = str(e)

    return row


def dock_multi_protein(smiles, pdb_ids, name='ligand',
                       residues_map=None, box_size=20,
                       exhaustiveness=32, n_poses=5,
                       output_dir='/content/acudock_multi',
                       progress_callback=None,
                       use_unidock=False, max_workers=3):
    """Dock one ligand against multiple protein targets in parallel.

    Prepares the ligand once and docks it against each protein using
    concurrent workers. Each protein is fetched, prepared, and docked
    independently. Optionally uses Uni-Dock GPU acceleration per target
    with automatic Vina fallback.

    Args:
        smiles: SMILES string for the ligand.
        pdb_ids: List of PDB ID strings.
        name: Ligand name for labeling.
        residues_map: Optional dict {pdb_id: [residue_ints]} for
            per-protein active site definition. If None or missing
            for a given PDB ID, uses whole-protein centroid.
        box_size: Search box size in Angstroms (applied uniformly).
        exhaustiveness: Vina exhaustiveness.
        n_poses: Number of poses per protein.
        output_dir: Working directory.
        progress_callback: Optional callable(current, total, pdb_id, status_msg).
        use_unidock: If True, attempt Uni-Dock GPU for each target
            (falls back to Vina on failure).
        max_workers: Max parallel protein preparations/docks (default 3).
            Capped to avoid Colab memory issues.

    Returns:
        (results_df, best_pdb_id, best_protein_pdb, best_poses_path)

        results_df columns: PDB_ID, Best_Score, Est_Kd_uM, Interpretation,
                            Num_Poses, Center, Prep_Time_s, Dock_Time_s,
                            Engine, Error
        best_pdb_id: PDB ID with the best (most negative) score.
        best_protein_pdb: Path to the prepared PDB of the best target.
        best_poses_path: Path to the Vina poses PDBQT of the best target.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(output_dir, exist_ok=True)
    if residues_map is None:
        residues_map = {}
    box = [int(box_size)] * 3

    # If Uni-Dock requested, verify it's available
    if use_unidock and not check_unidock_available():
        if progress_callback:
            progress_callback(0, len(pdb_ids), '',
                              'Uni-Dock not found, using Vina for all targets.')
        use_unidock = False

    # Prepare ligand once
    ligand_pdbqt, _ = prepare_ligand(smiles, name=name, output_dir=output_dir)

    # Build work items
    clean_ids = []
    work_args = []
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        if not pdb_id:
            continue
        clean_ids.append(pdb_id)
        residues = residues_map.get(pdb_id)
        work_args.append((pdb_id, ligand_pdbqt, residues, box,
                          exhaustiveness, n_poses, output_dir, use_unidock))

    # Cap workers to avoid memory issues on Colab
    n_workers = min(max(1, int(max_workers)), len(clean_ids), 4)

    results = []
    best_score = float('inf')
    best_pdb_id = None
    best_protein_pdb = None
    best_poses_path = None

    if n_workers <= 1 or len(clean_ids) <= 1:
        # Sequential fallback for single protein or explicit single-worker
        for i, args in enumerate(work_args):
            pdb_id = args[0]
            if progress_callback:
                progress_callback(i, len(clean_ids), pdb_id,
                                  'Preparing & docking...')
            row = _dock_single_protein(args)

            protein_pdb = row.pop('_protein_pdb', None)
            poses_path = row.pop('_poses_path', None)

            if row.get('Best_Score') is not None and row['Best_Score'] < best_score:
                best_score = row['Best_Score']
                best_pdb_id = pdb_id
                best_protein_pdb = protein_pdb
                best_poses_path = poses_path

            results.append(row)
            if progress_callback:
                status = (f"Done ({row['Best_Score']} kcal/mol)"
                          if row['Best_Score'] is not None
                          else f"Failed: {row['Error']}")
                progress_callback(i, len(clean_ids), pdb_id, status)
    else:
        # Parallel execution
        if progress_callback:
            progress_callback(0, len(clean_ids), '',
                              f'Parallel docking with {n_workers} workers...')

        future_to_pdb = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for args in work_args:
                future = executor.submit(_dock_single_protein, args)
                future_to_pdb[future] = args[0]

            completed = 0
            for future in as_completed(future_to_pdb):
                pdb_id = future_to_pdb[future]
                completed += 1

                try:
                    row = future.result()
                except Exception as e:
                    row = {'PDB_ID': pdb_id, 'Best_Score': None,
                           'Est_Kd_uM': None, 'Interpretation': 'Failed',
                           'Num_Poses': 0, 'Center': '', 'Prep_Time_s': 0,
                           'Dock_Time_s': 0, 'Engine': '', 'Error': str(e)}

                protein_pdb = row.pop('_protein_pdb', None)
                poses_path = row.pop('_poses_path', None)

                if (row.get('Best_Score') is not None
                        and row['Best_Score'] < best_score):
                    best_score = row['Best_Score']
                    best_pdb_id = pdb_id
                    best_protein_pdb = protein_pdb
                    best_poses_path = poses_path

                results.append(row)

                if progress_callback:
                    status = (f"Done ({row['Best_Score']} kcal/mol)"
                              if row['Best_Score'] is not None
                              else f"Failed: {row['Error']}")
                    progress_callback(completed - 1, len(clean_ids),
                                      pdb_id, status)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Best_Score', ascending=True,
                                            na_position='last').reset_index(drop=True)

    return results_df, best_pdb_id, best_protein_pdb, best_poses_path
