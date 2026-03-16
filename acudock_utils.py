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

def _has_metal_atoms(mol):
    """Check if an RDKit molecule contains metal atoms.

    Returns a list of metal element symbols found, or empty list if none.
    """
    metals = {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
        57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  # d-block + common
    }
    found = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in metals:
            found.append(atom.GetSymbol())
    return found


def _prepare_ligand_obabel(smiles, pdbqt_path):
    """Prepare a ligand PDBQT using OpenBabel (fallback for metal compounds).

    OpenBabel handles metal coordination geometry better than RDKit's MMFF/UFF.
    Returns True on success, False on failure.
    """
    import tempfile
    smi_path = pdbqt_path.replace('.pdbqt', '.smi')
    with open(smi_path, 'w') as f:
        f.write(smiles)

    try:
        # SMILES -> 3D SDF with OpenBabel (handles metals)
        sdf_path = pdbqt_path.replace('.pdbqt', '.sdf')
        result = subprocess.run(
            ['obabel', smi_path, '-O', sdf_path, '--gen3d', '--best'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0 or not os.path.exists(sdf_path):
            # Try without --best (faster, less strict)
            result = subprocess.run(
                ['obabel', smi_path, '-O', sdf_path, '--gen3d'],
                capture_output=True, text=True, timeout=60
            )
        if result.returncode != 0 or not os.path.exists(sdf_path):
            return False

        # SDF -> PDBQT
        result = subprocess.run(
            ['obabel', sdf_path, '-O', pdbqt_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not os.path.exists(pdbqt_path):
            return False

        # Verify non-empty
        with open(pdbqt_path, 'r') as f:
            content = f.read()
        if 'ATOM' not in content and 'HETATM' not in content:
            return False

        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def prepare_ligand(smiles, name='ligand', output_dir='/content/acudock_pro'):
    """Convert SMILES to 3D structure and PDBQT for docking.

    Pipeline: SMILES -> RDKit Mol -> 3D embed (ETKDGv3) -> MMFF optimize -> Meeko -> PDBQT

    Returns (pdbqt_path, rdkit_mol).
    Raises ValueError for metal-containing compounds (not supported by Vina).
    """
    os.makedirs(output_dir, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    # AutoDock Vina only supports organic atom types (C, N, O, S, H, F, Cl,
    # Br, I, P).  Metal atoms (Pt, Ru, Fe, etc.) are not valid AutoDock types
    # and will crash the docking engine.  Detect early and give a clear error.
    metal_atoms = _has_metal_atoms(mol)
    if metal_atoms:
        unique = sorted(set(metal_atoms))
        raise ValueError(
            f'Metal atoms detected: {", ".join(unique)}. '
            f'AutoDock Vina does not support metal-containing compounds '
            f'(chelates, metallocenes, coordination complexes). '
            f'Only organic molecules with C, N, O, S, H, F, Cl, Br, I, P '
            f'are supported. Consider removing the metal center or using a '
            f'docking program that supports metals (e.g. GOLD, Glide).'
        )

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params2 = AllChem.ETKDG()
        params2.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params2)

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:
            pass  # Use unoptimized coordinates

    pdbqt_path = os.path.join(output_dir, f'{name}.pdbqt')

    # Convert to PDBQT via Meeko
    meeko_ok = False
    try:
        preparator = meeko.MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)
        pdbqt_string = meeko.PDBQTWriterLegacy.write_string(mol_setup_list[0])
        if pdbqt_string and pdbqt_string[0].strip():
            with open(pdbqt_path, 'w') as f:
                f.write(pdbqt_string[0])
            meeko_ok = True
    except Exception:
        raise

    # Fallback: use OpenBabel for metal compounds or Meeko failures
    if not meeko_ok:
        print('  Trying OpenBabel for ligand preparation...')
        if _prepare_ligand_obabel(smiles, pdbqt_path):
            print('  OpenBabel preparation succeeded')
        else:
            raise ValueError(
                f'Cannot prepare ligand for docking. '
                f'{"Metal-containing compounds (chelates) have limited support. " if has_metal else ""}'
                f'Please check the SMILES string or try a different compound.'
            )

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
        'RMSD_lb': [round(e[1], 2) if len(e) > 1 else 0.0 for e in vina_energies],
        'RMSD_ub': [round(e[2], 2) if len(e) > 2 else 0.0 for e in vina_energies],
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
    """Render protein overview and binding site closeup for PDF reports.

    Overview uses spectrum-colored ribbon-like backbone (matching py3Dmol
    cartoon style) with green ligand sticks and translucent surface.
    Binding site shows nearby residue sticks with H-bond contacts.

    Returns dict mapping view name to PNG file path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize
    from matplotlib import cm

    # Parse protein atoms — track chain and residue ordering
    prot_ca = []       # (x, y, z, res_index) for backbone spline
    prot_residues = {} # (res_name, res_num) -> [(x, y, z, element, atom_name)]
    chain_breaks = []  # indices where chain breaks occur
    prev_ca = None
    res_index = 0
    seen_residues = set()

    with open(protein_pdb, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                chain_id = line[21] if len(line) > 21 else 'A'
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
            except (ValueError, IndexError):
                continue
            if elem.upper() == 'H':
                continue

            res_key = (chain_id, res_name, res_num)
            if res_key not in seen_residues:
                seen_residues.add(res_key)
                res_index += 1

            if atom_name == 'CA':
                pos = np.array([x, y, z])
                if prev_ca is not None and np.linalg.norm(pos - prev_ca) > 5.0:
                    chain_breaks.append(len(prot_ca))
                prev_ca = pos
                prot_ca.append((x, y, z, res_index))

            key = (res_name, res_num)
            if key not in prot_residues:
                prot_residues[key] = []
            prot_residues[key].append((x, y, z, elem.upper(), atom_name))

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

    lig_arr = np.array(lig_coords)
    lig_center = lig_arr.mean(axis=0)
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    elem_colors = {
        'C': '#2E7D32', 'N': '#3050F8', 'O': '#FF0D0D', 'S': '#FFFF30',
        'H': '#FFFFFF', 'F': '#90E050', 'CL': '#1FF01F', 'BR': '#A62929',
        'P': '#FF8000', 'I': '#940094',
    }

    def _draw_ligand_sticks(ax, with_surface=False):
        """Draw ligand as colored sticks (green carbon) matching py3Dmol."""
        # Draw bonds first (behind atoms)
        for i in range(len(lig_arr)):
            for j in range(i + 1, len(lig_arr)):
                d = np.linalg.norm(lig_arr[i] - lig_arr[j])
                if d < 1.9:
                    mid = (lig_arr[i] + lig_arr[j]) / 2
                    ei = lig_elements[i] if i < len(lig_elements) else 'C'
                    ej = lig_elements[j] if j < len(lig_elements) else 'C'
                    ci = elem_colors.get(ei, '#2E7D32')
                    cj = elem_colors.get(ej, '#2E7D32')
                    ax.plot([lig_arr[i, 0], mid[0]],
                            [lig_arr[i, 1], mid[1]],
                            [lig_arr[i, 2], mid[2]],
                            color=ci, linewidth=3.5, solid_capstyle='round',
                            zorder=9)
                    ax.plot([mid[0], lig_arr[j, 0]],
                            [mid[1], lig_arr[j, 1]],
                            [mid[2], lig_arr[j, 2]],
                            color=cj, linewidth=3.5, solid_capstyle='round',
                            zorder=9)
        # Draw atom spheres on top
        for i, (x, y, z) in enumerate(lig_coords):
            elem = lig_elements[i] if i < len(lig_elements) else 'C'
            color = elem_colors.get(elem, '#2E7D32')
            size = 40 if elem == 'C' else 55
            ax.scatter(x, y, z, c=color, s=size, edgecolors='none',
                       alpha=0.9, zorder=10, depthshade=True)

        if with_surface:
            # Draw a translucent green shell around the ligand
            u = np.linspace(0, 2 * np.pi, 20)
            v_ang = np.linspace(0, np.pi, 15)
            for (x, y, z) in lig_coords:
                r = 1.5
                xs = x + r * np.outer(np.cos(u), np.sin(v_ang))
                ys = y + r * np.outer(np.sin(u), np.sin(v_ang))
                zs = z + r * np.outer(np.ones_like(u), np.cos(v_ang))
                ax.plot_surface(xs, ys, zs, color='#4CAF50', alpha=0.08,
                                shade=False, zorder=8)

    def _smooth_spline(coords, n_points=500):
        """Interpolate CA coordinates with a smooth cubic spline."""
        try:
            from scipy.interpolate import make_interp_spline
            t = np.linspace(0, 1, len(coords))
            t_new = np.linspace(0, 1, n_points)
            spline = make_interp_spline(t, coords, k=3)
            return spline(t_new)
        except (ImportError, ValueError):
            return coords

    # === View 1: Protein Overview (ribbon-like with spectrum coloring) ===
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if prot_ca:
        ca_arr = np.array([(c[0], c[1], c[2]) for c in prot_ca])
        ca_idx = np.array([c[3] for c in prot_ca], dtype=float)
        max_idx = ca_idx.max() if ca_idx.max() > 0 else 1.0

        # Split by chain breaks and draw each segment
        segments = []
        seg_indices = []
        start = 0
        for brk in chain_breaks:
            if brk > start:
                segments.append(ca_arr[start:brk])
                seg_indices.append(ca_idx[start:brk])
            start = brk
        segments.append(ca_arr[start:])
        seg_indices.append(ca_idx[start:])

        cmap = cm.get_cmap('rainbow')

        for seg, seg_idx in zip(segments, seg_indices):
            if len(seg) < 4:
                # Too few points for spline, draw straight
                norm_idx = seg_idx / max_idx
                for k in range(len(seg) - 1):
                    color = cmap(norm_idx[k])
                    ax.plot(seg[k:k+2, 0], seg[k:k+2, 1], seg[k:k+2, 2],
                            color=color, linewidth=4.0, solid_capstyle='round',
                            alpha=0.85)
                continue

            # Smooth spline interpolation
            n_pts = max(len(seg) * 5, 200)
            smooth = _smooth_spline(seg, n_points=n_pts)
            # Interpolate residue indices for coloring
            t_orig = np.linspace(0, 1, len(seg))
            t_new = np.linspace(0, 1, n_pts)
            smooth_idx = np.interp(t_new, t_orig, seg_idx)
            norm = Normalize(vmin=0, vmax=max_idx)

            # Draw as colored segments (thick tube-like line)
            for k in range(len(smooth) - 1):
                color = cmap(norm(smooth_idx[k]))
                ax.plot(smooth[k:k+2, 0], smooth[k:k+2, 1], smooth[k:k+2, 2],
                        color=color, linewidth=4.5, solid_capstyle='round',
                        alpha=0.85)

    _draw_ligand_sticks(ax, with_surface=True)

    # Camera: face the ligand from a nice angle
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_title('Protein Structure with Docked Ligand',
                 fontsize=11, fontweight='bold', pad=10)

    path = os.path.join(output_dir, f'{prefix}_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    plt.close(fig)
    paths['overview'] = path

    # === View 2: Binding Site Multi-Angle Panel ===
    # Find residues with any heavy atom within 5A of any ligand atom
    contact_dist = 5.0
    nearby = {}
    for key, atoms in prot_residues.items():
        for (px, py, pz, pe, pa) in atoms:
            pv = np.array([px, py, pz])
            if np.min(np.linalg.norm(lig_arr - pv, axis=1)) < contact_dist:
                nearby[key] = atoms
                break

    # Pre-compute H-bonds and hydrophobic contacts
    polar = ('N', 'O')
    polar_lig = [(x, y, z) for (x, y, z), e
                 in zip(lig_coords, lig_elements) if e in polar]
    carbon_lig = [(x, y, z) for (x, y, z), e
                  in zip(lig_coords, lig_elements) if e == 'C']
    hbond_pairs = []   # (prot_xyz, lig_xyz, distance)
    hydro_pairs = []   # (prot_xyz, lig_xyz, distance)
    for key, atoms in nearby.items():
        for (px, py, pz, pe, pa) in atoms:
            if pe in polar:
                for (lx, ly, lz) in polar_lig:
                    d = np.sqrt((px - lx)**2 + (py - ly)**2 + (pz - lz)**2)
                    if 2.0 < d < 3.5:
                        hbond_pairs.append(((px, py, pz), (lx, ly, lz), d))
            if pe == 'C':
                for (lx, ly, lz) in carbon_lig:
                    d = np.sqrt((px - lx)**2 + (py - ly)**2 + (pz - lz)**2)
                    if 2.0 < d < 4.0:
                        hydro_pairs.append(((px, py, pz), (lx, ly, lz), d))

    res_elem_colors = {
        'C': '#90CAF9', 'N': '#3050F8', 'O': '#FF0D0D', 'S': '#FFFF30',
    }

    def _draw_binding_site(ax, view_elev, view_azim, show_labels=True):
        """Draw the binding site on the given axes."""
        # Nearby residue sticks
        for key, atoms in nearby.items():
            coords_r = np.array([(a[0], a[1], a[2]) for a in atoms])
            elems = [a[3] for a in atoms]
            for i in range(len(coords_r)):
                for j in range(i + 1, len(coords_r)):
                    if np.linalg.norm(coords_r[i] - coords_r[j]) < 1.9:
                        mid = (coords_r[i] + coords_r[j]) / 2
                        ci = res_elem_colors.get(elems[i], '#90CAF9')
                        cj = res_elem_colors.get(elems[j], '#90CAF9')
                        ax.plot([coords_r[i, 0], mid[0]],
                                [coords_r[i, 1], mid[1]],
                                [coords_r[i, 2], mid[2]],
                                color=ci, linewidth=2.0, alpha=0.8)
                        ax.plot([mid[0], coords_r[j, 0]],
                                [mid[1], coords_r[j, 1]],
                                [mid[2], coords_r[j, 2]],
                                color=cj, linewidth=2.0, alpha=0.8)
            if show_labels:
                ca_pos = None
                for (x, y, z, e, a) in atoms:
                    if a == 'CA':
                        ca_pos = (x, y, z)
                        break
                if ca_pos is None:
                    ca_pos = tuple(coords_r.mean(axis=0))
                res_name, res_num = key
                ax.text(ca_pos[0], ca_pos[1], ca_pos[2],
                        f'{res_name}{res_num}', fontsize=6, color='#1565C0',
                        fontweight='bold', ha='center', va='bottom', zorder=5)

        _draw_ligand_sticks(ax, with_surface=False)

        # H-bonds with distance labels
        for (p_xyz, l_xyz, d) in hbond_pairs:
            ax.plot([p_xyz[0], l_xyz[0]], [p_xyz[1], l_xyz[1]],
                    [p_xyz[2], l_xyz[2]],
                    ':', color='#F57F17', linewidth=2.5, alpha=0.9, zorder=8)
            if show_labels:
                mx = (p_xyz[0] + l_xyz[0]) / 2
                my = (p_xyz[1] + l_xyz[1]) / 2
                mz = (p_xyz[2] + l_xyz[2]) / 2
                ax.text(mx, my, mz, f'{d:.1f}A', fontsize=5,
                        color='#E65100', ha='center', va='center',
                        fontweight='bold', zorder=12,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))

        # Hydrophobic contacts (thin gray)
        for (p_xyz, l_xyz, d) in hydro_pairs:
            ax.plot([p_xyz[0], l_xyz[0]], [p_xyz[1], l_xyz[1]],
                    [p_xyz[2], l_xyz[2]],
                    ':', color='#BDBDBD', linewidth=1.0, alpha=0.6, zorder=7)

        # Set view limits around ligand
        lig_range_v = max(lig_arr.max(axis=0) - lig_arr.min(axis=0)) * 0.7 + 5
        ax.set_xlim(lig_center[0] - lig_range_v, lig_center[0] + lig_range_v)
        ax.set_ylim(lig_center[1] - lig_range_v, lig_center[1] + lig_range_v)
        ax.set_zlim(lig_center[2] - lig_range_v, lig_center[2] + lig_range_v)
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

    # Three separate binding site views — each saved as its own image
    # so the PDF renderer can place them individually at full page width.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2E7D32', linewidth=3, label='Ligand C'),
        Line2D([0], [0], color='#3050F8', linewidth=3, label='N'),
        Line2D([0], [0], color='#FF0D0D', linewidth=3, label='O'),
        Line2D([0], [0], color='#FFFF30', linewidth=3, label='S'),
        Line2D([0], [0], color='#90CAF9', linewidth=2, label='Protein C'),
        Line2D([0], [0], color='#F57F17', linewidth=2, linestyle=':',
               label='H-bond'),
        Line2D([0], [0], color='#BDBDBD', linewidth=1, linestyle=':',
               label='Hydrophobic'),
    ]

    angles = [
        (15, 60, 'Binding Site \u2014 Front View', 'binding_front'),
        (15, 150, 'Binding Site \u2014 Side View (90\u00b0)', 'binding_side'),
        (75, 60, 'Binding Site \u2014 Top-Down View', 'binding_top'),
    ]
    for elev, azim, title, key in angles:
        fig = plt.figure(figsize=(10, 9), dpi=150)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        _draw_binding_site(ax, elev, azim, show_labels=True)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   fontsize=8, frameon=True, fancybox=True,
                   borderpad=0.4, handlelength=2.0)
        fig.subplots_adjust(bottom=0.08)
        path = os.path.join(output_dir, f'{prefix}_{key}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white',
                    edgecolor='none')
        plt.close(fig)
        paths[key] = path

    return paths


def capture_py3dmol_overview(protein_pdb, poses_pdbqt, output_dir,
                             pose_index=0, prefix='pose'):
    """Capture py3Dmol ribbon view as PNG using JavaScript pngURI() in Colab.

    Creates the same spectrum-colored cartoon protein + green ligand sticks
    that the interactive viewer shows, and captures it as a static PNG for
    PDF reports.  The div must be visible (even briefly) so WebGL can render.

    Returns path to the saved PNG, or None if JS capture fails.
    """
    try:
        from google.colab import output as colab_output
    except ImportError:
        print('  Not in Colab — skipping JS capture')
        return None

    with open(protein_pdb, 'r') as f:
        protein_data = f.read()

    pose_data = extract_pose_from_pdbqt(poses_pdbqt, pose_index)
    ligand_data = _clean_pdbqt_for_viewer(pose_data)

    prot_b64 = base64.b64encode(protein_data.encode()).decode()
    lig_b64 = base64.b64encode(ligand_data.encode()).decode()

    # Generate contact analysis JS (pocket residues, labels, H-bonds)
    contact_js_lines, _ = _generate_contact_js(
        protein_data, pose_data, pocket_model_idx=2
    )
    contact_js_block = '\n            '.join(contact_js_lines)

    # The div must be in the visible viewport for WebGL to initialize.
    # We use opacity:0.01 so it's nearly invisible but still rendered.
    js = f"""
    (async function() {{
        try {{
            await new Promise((resolve, reject) => {{
                if (window.$3Dmol) {{ resolve(); return; }}
                var s = document.createElement('script');
                s.src = 'https://3Dmol.org/build/3Dmol-min.js';
                s.onload = resolve;
                s.onerror = reject;
                document.head.appendChild(s);
            }});

            var div = document.createElement('div');
            div.style.width = '1000px';
            div.style.height = '700px';
            div.style.position = 'absolute';
            div.style.zIndex = '-1';
            div.style.opacity = '0.01';
            div.style.pointerEvents = 'none';
            document.body.appendChild(div);

            var v = $3Dmol.createViewer(div, {{backgroundColor: 'white'}});
            v.addModel(atob("{prot_b64}"), "pdb");
            v.setStyle({{model:0}}, {{cartoon:{{color:'spectrum',opacity:0.7}}}});
            v.addModel(atob("{lig_b64}"), "pdb");
            v.setStyle({{model:1}}, {{stick:{{colorscheme:'greenCarbon',radius:0.2}}}});
            v.addSurface($3Dmol.SurfaceType.VDW,
                {{opacity:0.20,color:'green'}},{{model:1}});
            {contact_js_block}
            v.zoomTo({{model:1}});
            v.zoom(0.8);
            v.render();

            // Wait for WebGL render to complete
            await new Promise(r => setTimeout(r, 1500));
            v.render();
            await new Promise(r => setTimeout(r, 500));

            var uri = v.pngURI();
            document.body.removeChild(div);

            if (uri && uri.length > 100) {{
                return uri.split(',')[1];
            }}
            return '';
        }} catch(e) {{
            return '';
        }}
    }})()
    """

    try:
        result = colab_output.eval_js(js)
        if result and isinstance(result, str) and len(result) > 100:
            png_data = base64.b64decode(result)
            # Verify it's actually a valid PNG (starts with PNG header)
            if png_data[:4] == b'\x89PNG':
                out_path = os.path.join(output_dir, f'{prefix}_py3dmol_overview.png')
                with open(out_path, 'wb') as f:
                    f.write(png_data)
                return out_path
            else:
                print('  JS capture returned invalid PNG data')
    except Exception as e:
        print(f'  JS capture failed: {e}')

    return None


def capture_3d_views(protein_pdb, poses_pdbqt, output_dir,
                      pose_index=0, prefix='pose'):
    """Capture 3D views for PDF reports.

    Tries py3Dmol JS capture for the overview (ribbon/cartoon view),
    falls back to matplotlib. Always generates matplotlib binding site view.

    Returns dict mapping view name to PNG file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Try py3Dmol JS capture for overview (matches interactive viewer)
    overview_path = capture_py3dmol_overview(
        protein_pdb, poses_pdbqt, output_dir,
        pose_index=pose_index, prefix=prefix
    )
    if overview_path:
        paths['overview'] = overview_path
        print('  Captured py3Dmol ribbon view for PDF')

    # Generate matplotlib views (overview fallback + binding site)
    mpl_paths = capture_3d_views_matplotlib(
        protein_pdb, poses_pdbqt, output_dir,
        pose_index=pose_index, prefix=prefix
    )
    if mpl_paths:
        # Use matplotlib overview only if JS capture failed
        if 'overview' not in paths and 'overview' in mpl_paths:
            paths['overview'] = mpl_paths['overview']
        # Always include binding site from matplotlib
        if 'binding_site' in mpl_paths:
            paths['binding_site'] = mpl_paths['binding_site']
        print(f'  Rendered {len(mpl_paths)} views via matplotlib')

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


def _parse_pdb_atoms(pdb_data):
    """Parse PDB data into structured atom records.

    Returns list of dicts with keys: x, y, z, element, atom_name,
    res_name, res_num, chain_id, record_type.
    """
    atoms = []
    for line in pdb_data.split('\n'):
        rec = line[:6].strip()
        if rec not in ('ATOM', 'HETATM'):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_num = line[22:26].strip()
            chain_id = line[21] if len(line) > 21 else 'A'
            elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
        except (ValueError, IndexError):
            continue
        atoms.append({
            'x': x, 'y': y, 'z': z,
            'element': elem.upper(),
            'atom_name': atom_name,
            'res_name': res_name,
            'res_num': res_num,
            'chain_id': chain_id,
            'record_type': rec,
        })
    return atoms


def _find_nearby_residues(prot_atoms, lig_coords, cutoff=5.0):
    """Find protein residues with any heavy atom within *cutoff* A of ligand.

    Args:
        prot_atoms: List of atom dicts from _parse_pdb_atoms.
        lig_coords: Nx3 numpy array of ligand coordinates.
        cutoff: Distance threshold in Angstroms.

    Returns dict {(chain_id, res_name, res_num): [atom_dicts]}.
    """
    nearby = {}
    for atom in prot_atoms:
        if atom['element'] == 'H' or atom['record_type'] != 'ATOM':
            continue
        pos = np.array([atom['x'], atom['y'], atom['z']])
        if np.min(np.linalg.norm(lig_coords - pos, axis=1)) < cutoff:
            key = (atom['chain_id'], atom['res_name'], atom['res_num'])
            if key not in nearby:
                nearby[key] = []
            nearby[key].append(atom)
    return nearby


def _detect_hbonds(nearby_residues, lig_coords, lig_elements, min_d=2.0, max_d=3.5):
    """Detect potential hydrogen bonds between ligand and nearby residues.

    Returns list of dicts: {prot_xyz, lig_xyz, distance, prot_label}.
    """
    polar = ('N', 'O')
    polar_lig = [(c, e) for c, e in zip(lig_coords, lig_elements)
                 if e in polar]
    hbonds = []
    for key, atoms in nearby_residues.items():
        chain, rname, rnum = key
        for atom in atoms:
            if atom['element'] not in polar:
                continue
            pa = np.array([atom['x'], atom['y'], atom['z']])
            for (lc, le) in polar_lig:
                d = np.linalg.norm(pa - np.array(lc))
                if min_d < d < max_d:
                    hbonds.append({
                        'prot_xyz': (atom['x'], atom['y'], atom['z']),
                        'lig_xyz': lc,
                        'distance': round(d, 2),
                        'prot_label': f"{rname}{rnum}.{atom['atom_name']}",
                    })
    return hbonds


def _detect_hydrophobic_contacts(nearby_residues, lig_coords, lig_elements,
                                  max_d=4.0):
    """Detect hydrophobic C-C contacts between ligand and protein.

    Returns list of dicts: {prot_xyz, lig_xyz, distance}.
    """
    carbon_lig = [c for c, e in zip(lig_coords, lig_elements) if e == 'C']
    contacts = []
    for key, atoms in nearby_residues.items():
        for atom in atoms:
            if atom['element'] != 'C':
                continue
            pa = np.array([atom['x'], atom['y'], atom['z']])
            for lc in carbon_lig:
                d = np.linalg.norm(pa - np.array(lc))
                if 2.0 < d < max_d:
                    contacts.append({
                        'prot_xyz': (atom['x'], atom['y'], atom['z']),
                        'lig_xyz': lc,
                        'distance': round(d, 2),
                    })
    return contacts


def _get_ligand_coords_and_elements(pdbqt_data):
    """Extract coordinates and element symbols from cleaned PDBQT/PDB data.

    Returns (list_of_tuples, list_of_elements).
    """
    clean = _clean_pdbqt_for_viewer(pdbqt_data)
    coords = []
    elements = []
    for line in clean.split('\n'):
        if line.startswith(('ATOM', 'HETATM')):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) > 76 else line[12:14].strip()
                if not elem:
                    elem = line[12:16].strip()[0]
                coords.append((x, y, z))
                elements.append(elem.upper())
            except (ValueError, IndexError):
                pass
    return coords, elements


def _get_residue_pdb_block(nearby_residues):
    """Build a PDB-format string for a set of nearby residues.

    This can be loaded as a separate model in py3Dmol to style
    binding-site residues independently from the full protein.
    """
    lines = []
    serial = 1
    for (chain, rname, rnum), atoms in nearby_residues.items():
        for a in atoms:
            if a['element'] == 'H':
                continue
            lines.append(
                f"ATOM  {serial:5d} {a['atom_name']:4s} {rname:3s} "
                f"{chain}{rnum:>4s}    "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                f"  1.00  0.00          {a['element']:>2s}"
            )
            serial += 1
    lines.append('END')
    return '\n'.join(lines)


def visualize_pose(protein_pdb, poses_pdbqt, pose_index=0,
                   width=1000, height=700):
    """Create an enhanced py3Dmol view of a docked pose.

    Shows: protein cartoon, ligand sticks with VDW surface, nearby
    residue sticks with labels, H-bond dashed lines (yellow),
    hydrophobic contacts (gray), and binding pocket surface.

    Returns a py3Dmol.view object (call .show() to render).
    """
    import py3Dmol

    with open(protein_pdb, 'r') as f:
        protein_data = f.read()
    with open(poses_pdbqt, 'r') as f:
        poses_data = f.read()

    # Extract specific pose
    models = poses_data.split('MODEL')
    if len(models) > 1:
        pose_data = 'MODEL' + models[pose_index + 1].split('ENDMDL')[0] + 'ENDMDL'
    else:
        pose_data = poses_data

    lig_coords, lig_elements = _get_ligand_coords_and_elements(pose_data)
    lig_arr = np.array(lig_coords) if lig_coords else np.zeros((1, 3))

    # Analyse contacts
    prot_atoms = _parse_pdb_atoms(protein_data)
    nearby = _find_nearby_residues(prot_atoms, lig_arr, cutoff=5.0)
    hbonds = _detect_hbonds(nearby, lig_coords, lig_elements)
    hydrophobics = _detect_hydrophobic_contacts(nearby, lig_coords, lig_elements)
    pocket_pdb = _get_residue_pdb_block(nearby)

    view = py3Dmol.view(width=width, height=height)

    # Model 0 — full protein cartoon
    view.addModel(protein_data, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum', 'opacity': 0.7}})

    # Model 1 — ligand sticks + surface
    view.addModel(_clean_pdbqt_for_viewer(pose_data), 'pdb')
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
    view.addSurface(py3Dmol.VDW, {'opacity': 0.20, 'color': 'green'}, {'model': 1})

    # Model 2 — binding pocket residues as sticks
    view.addModel(pocket_pdb, 'pdb')
    view.setStyle({'model': 2}, {'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.15}})
    # Translucent pocket surface
    view.addSurface(py3Dmol.VDW, {'opacity': 0.10, 'color': 'lightblue'}, {'model': 2})

    # Residue labels
    labeled = set()
    for (chain, rname, rnum), atoms in nearby.items():
        label_key = f"{rname}{rnum}"
        if label_key in labeled:
            continue
        labeled.add(label_key)
        # Use CA position or centroid
        ca = next((a for a in atoms if a['atom_name'] == 'CA'), None)
        if ca:
            lx, ly, lz = ca['x'], ca['y'], ca['z']
        else:
            lx = np.mean([a['x'] for a in atoms])
            ly = np.mean([a['y'] for a in atoms])
            lz = np.mean([a['z'] for a in atoms])
        view.addLabel(label_key, {
            'position': {'x': lx, 'y': ly, 'z': lz},
            'fontSize': 11,
            'fontColor': 'black',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.7,
            'borderColor': 'gray',
            'borderThickness': 1,
        })

    # H-bond dashed lines (yellow)
    for hb in hbonds:
        px, py_, pz = hb['prot_xyz']
        lx, ly, lz = hb['lig_xyz']
        view.addLine({
            'start': {'x': px, 'y': py_, 'z': pz},
            'end': {'x': lx, 'y': ly, 'z': lz},
            'color': '#F5B041',
            'dashed': True,
            'dashLength': 0.2,
            'gapLength': 0.1,
            'linewidth': 3,
        })

    # Hydrophobic contacts (thin gray dashed)
    for hc in hydrophobics:
        px, py_, pz = hc['prot_xyz']
        lx, ly, lz = hc['lig_xyz']
        view.addLine({
            'start': {'x': px, 'y': py_, 'z': pz},
            'end': {'x': lx, 'y': ly, 'z': lz},
            'color': '#AAAAAA',
            'dashed': True,
            'dashLength': 0.15,
            'gapLength': 0.15,
            'linewidth': 1,
        })

    view.zoomTo({'model': 1})
    view.zoom(0.8)
    return view


def visualize_multi_poses(protein_pdb, poses_pdbqt, n_poses=3,
                          width=1000, height=700):
    """Overlay multiple poses on the protein with contact analysis.

    Shows nearby residues and H-bonds for the top pose.
    Returns a py3Dmol.view object.
    """
    import py3Dmol

    with open(protein_pdb, 'r') as f:
        protein_data = f.read()
    with open(poses_pdbqt, 'r') as f:
        poses_data = f.read()

    all_models = poses_data.split('MODEL')
    pose_strs = []
    for i in range(min(n_poses, len(all_models) - 1)):
        pose_strs.append('MODEL' + all_models[i + 1].split('ENDMDL')[0] + 'ENDMDL')

    # Contact analysis on top pose
    if pose_strs:
        lig_coords, lig_elements = _get_ligand_coords_and_elements(pose_strs[0])
        lig_arr = np.array(lig_coords) if lig_coords else np.zeros((1, 3))
        prot_atoms = _parse_pdb_atoms(protein_data)
        nearby = _find_nearby_residues(prot_atoms, lig_arr, cutoff=5.0)
        hbonds = _detect_hbonds(nearby, lig_coords, lig_elements)
        pocket_pdb = _get_residue_pdb_block(nearby)
    else:
        nearby, hbonds, pocket_pdb = {}, [], ''

    view = py3Dmol.view(width=width, height=height)

    # Model 0 — protein cartoon (faded)
    view.addModel(protein_data, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'white', 'opacity': 0.4}})

    # Add pocket residues
    model_idx = 1
    if pocket_pdb:
        view.addModel(pocket_pdb, 'pdb')
        view.setStyle({'model': model_idx}, {'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.12}})
        model_idx += 1

    # Add pose models
    colors = ['#2ECC71', '#00BCD4', '#E91E63', '#FF9800', '#FFEB3B']
    first_pose_model = model_idx
    for i, pose in enumerate(pose_strs):
        view.addModel(_clean_pdbqt_for_viewer(pose), 'pdb')
        view.setStyle({'model': model_idx}, {
            'stick': {'color': colors[i % len(colors)], 'radius': 0.18}
        })
        model_idx += 1

    # Residue labels
    labeled = set()
    for (chain, rname, rnum), atoms in nearby.items():
        label_key = f"{rname}{rnum}"
        if label_key in labeled:
            continue
        labeled.add(label_key)
        ca = next((a for a in atoms if a['atom_name'] == 'CA'), None)
        if ca:
            lx, ly, lz = ca['x'], ca['y'], ca['z']
        else:
            lx = np.mean([a['x'] for a in atoms])
            ly = np.mean([a['y'] for a in atoms])
            lz = np.mean([a['z'] for a in atoms])
        view.addLabel(label_key, {
            'position': {'x': lx, 'y': ly, 'z': lz},
            'fontSize': 10,
            'fontColor': 'black',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.6,
        })

    # H-bond lines for top pose
    for hb in hbonds:
        px, py_, pz = hb['prot_xyz']
        lx, ly, lz = hb['lig_xyz']
        view.addLine({
            'start': {'x': px, 'y': py_, 'z': pz},
            'end': {'x': lx, 'y': ly, 'z': lz},
            'color': '#F5B041',
            'dashed': True,
            'dashLength': 0.2,
            'gapLength': 0.1,
            'linewidth': 2,
        })

    view.zoomTo({'model': first_pose_model})
    view.zoom(0.8)
    return view


def display_3d_viewer(view):
    """Display a py3Dmol view reliably inside ipywidgets Output contexts.

    Uses an iframe with data URI to bypass JavaScript injection issues
    that occur when py3Dmol runs inside nested Output widgets in Colab.
    """
    from IPython.display import display, HTML

    html_content = view._make_html()
    b64 = base64.b64encode(html_content.encode()).decode()
    w = view.width if isinstance(view.width, (int, float)) else 1000
    h = view.height if isinstance(view.height, (int, float)) else 700
    display(HTML(
        f'<iframe src="data:text/html;base64,{b64}" '
        f'width="{int(w) + 20}" height="{int(h) + 20}" '
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

def _wrap_3dmol_iframe(vid, js_block, width, height):
    """Wrap 3Dmol.js code in a self-contained iframe with controls.

    Adds spin toggle and preset camera angle buttons below the viewer.
    """
    sq = "'"  # single-quote for embedding in f-strings
    btn = 'style="padding:3px 10px;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5;"'
    controls_html = (
        f'<div style="display:flex;gap:6px;padding:4px 0;font-family:sans-serif;font-size:12px;">'
        f'<button onclick="window._v_{vid}.spin(window._spin_{vid}=!window._spin_{vid})" '
        f'{btn}>Spin</button>'
        f'<button onclick="window._v_{vid}.rotate(90,{sq}y{sq});window._v_{vid}.render()" '
        f'{btn}>Rotate 90&deg;</button>'
        f'<button onclick="window._v_{vid}.rotate(90,{sq}x{sq});window._v_{vid}.render()" '
        f'{btn}>Top View</button>'
        f'<button onclick="window._v_{vid}.zoomTo();window._v_{vid}.zoom(0.8);window._v_{vid}.render()" '
        f'{btn}>Reset</button>'
        f'</div>'
    )

    inner_html = (
        '<!DOCTYPE html><html><head>'
        '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'
        '</head><body style="margin:0;padding:4px;">'
        f'<div id="{vid}" style="width:100%;height:calc(100% - 36px);position:relative;"></div>'
        f'{controls_html}'
        f'<script>(function(){{{js_block};window._v_{vid}=v;window._spin_{vid}=false;}})()</script>'
        '</body></html>'
    )
    escaped = inner_html.replace('&', '&amp;').replace('"', '&quot;')
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:{width};height:{height};border:none;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def _generate_contact_js(protein_pdb_data, ligand_pdb_data, pocket_model_idx):
    """Generate 3Dmol.js lines for contacts, labels, and pocket residues.

    Args:
        protein_pdb_data: Full protein PDB string.
        ligand_pdb_data: Cleaned ligand PDB string.
        pocket_model_idx: Model index to assign to the pocket residues.

    Returns (js_lines, next_model_idx) where js_lines is a list of
    JavaScript strings to append to the viewer setup.
    """
    import json as _json

    lig_coords, lig_elements = _get_ligand_coords_and_elements(ligand_pdb_data)
    if not lig_coords:
        return [], pocket_model_idx

    lig_arr = np.array(lig_coords)
    prot_atoms = _parse_pdb_atoms(protein_pdb_data)
    nearby = _find_nearby_residues(prot_atoms, lig_arr, cutoff=5.0)
    hbonds = _detect_hbonds(nearby, lig_coords, lig_elements)
    hydrophobics = _detect_hydrophobic_contacts(nearby, lig_coords, lig_elements)
    pocket_pdb = _get_residue_pdb_block(nearby)

    js = []
    midx = pocket_model_idx

    # Add pocket residues model
    if pocket_pdb:
        pocket_b64 = base64.b64encode(pocket_pdb.encode()).decode()
        js.append(f'v.addModel(atob("{pocket_b64}"),"pdb");')
        js.append(
            f'v.setStyle({{model:{midx}}},{{stick:{{colorscheme:"whiteCarbon",radius:0.15}}}});'
        )
        # Pocket surface
        js.append(
            f'v.addSurface($3Dmol.SurfaceType.VDW,'
            f'{{opacity:0.10,color:"lightblue"}},{{model:{midx}}});'
        )
        midx += 1

    # Residue labels
    labeled = set()
    for (chain, rname, rnum), atoms in nearby.items():
        label_key = f"{rname}{rnum}"
        if label_key in labeled:
            continue
        labeled.add(label_key)
        ca = next((a for a in atoms if a['atom_name'] == 'CA'), None)
        if ca:
            lx, ly, lz = ca['x'], ca['y'], ca['z']
        else:
            lx = np.mean([a['x'] for a in atoms])
            ly = np.mean([a['y'] for a in atoms])
            lz = np.mean([a['z'] for a in atoms])
        label_opts = _json.dumps({
            'position': {'x': round(lx, 2), 'y': round(ly, 2), 'z': round(lz, 2)},
            'fontSize': 11,
            'fontColor': 'black',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.7,
            'borderColor': 'gray',
            'borderThickness': 1,
        })
        js.append(f'v.addLabel("{label_key}",{label_opts});')

    # H-bond dashed lines (yellow-orange)
    for hb in hbonds:
        px, py_, pz = hb['prot_xyz']
        lx, ly, lz = hb['lig_xyz']
        line_opts = _json.dumps({
            'start': {'x': round(px, 2), 'y': round(py_, 2), 'z': round(pz, 2)},
            'end': {'x': round(lx, 2), 'y': round(ly, 2), 'z': round(lz, 2)},
            'color': '#F5B041',
            'dashed': True,
            'dashLength': 0.2,
            'gapLength': 0.1,
            'linewidth': 3,
        })
        js.append(f'v.addLine({line_opts});')

    # Hydrophobic contacts (thin gray)
    for hc in hydrophobics:
        px, py_, pz = hc['prot_xyz']
        lx, ly, lz = hc['lig_xyz']
        line_opts = _json.dumps({
            'start': {'x': round(px, 2), 'y': round(py_, 2), 'z': round(pz, 2)},
            'end': {'x': round(lx, 2), 'y': round(ly, 2), 'z': round(lz, 2)},
            'color': '#AAAAAA',
            'dashed': True,
            'dashLength': 0.15,
            'gapLength': 0.15,
            'linewidth': 1,
        })
        js.append(f'v.addLine({line_opts});')

    return js, midx


def make_3d_viewer_html(protein_pdb_data, ligand_data=None,
                        width='100%', height='600px',
                        protein_style='cartoon', ligand_style='stick'):
    """Generate an enhanced HTML string with an interactive 3Dmol.js viewer.

    When ligand data is provided, also shows: binding pocket residue
    sticks with labels, H-bond dashed lines (yellow-orange), hydrophobic
    contacts (gray), pocket surface, and control buttons (spin, rotate,
    top view, reset).

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
            'v.setStyle({model:0},{cartoon:{color:"spectrum",opacity:0.7}});'
        )
    else:
        js_lines.append(
            'v.setStyle({model:0},{stick:{colorscheme:"whiteCarbon",radius:0.1}});'
        )

    if ligand_data:
        clean_lig = _clean_pdbqt_for_viewer(ligand_data)
        lig_b64 = base64.b64encode(clean_lig.encode()).decode()
        js_lines.append(f'v.addModel(atob("{lig_b64}"),"pdb");')
        if ligand_style == 'stick':
            js_lines.append(
                'v.setStyle({model:1},{stick:{colorscheme:"greenCarbon",radius:0.2}});'
            )
        else:
            js_lines.append(
                'v.setStyle({model:1},{sphere:{scale:0.3,colorscheme:"greenCarbon"}});'
            )
        # Ligand VDW surface
        js_lines.append(
            'v.addSurface($3Dmol.SurfaceType.VDW,'
            '{opacity:0.20,color:"green"},{model:1});'
        )

        # Contact analysis: pocket residues, labels, H-bonds, hydrophobics
        contact_js, _ = _generate_contact_js(protein_pdb_data, ligand_data, 2)
        js_lines.extend(contact_js)

        js_lines.append('v.zoomTo({model:1});v.zoom(0.8);')
    else:
        js_lines.append('v.zoomTo();')

    js_lines.append('v.render();')
    js_block = '\n'.join(js_lines)

    return _wrap_3dmol_iframe(vid, js_block, width, height)


def visualize_poses(protein_pdb_data, poses_pdbqt_data, n_poses=3,
                    width='100%', height='600px'):
    """Generate enhanced HTML overlaying multiple docked poses on protein.

    Shows pocket residues, labels, and H-bonds based on the top pose.
    Includes spin/rotate/reset control buttons.

    Args:
        protein_pdb_data: Protein PDB file contents.
        poses_pdbqt_data: Multi-model PDBQT file contents.
        n_poses: Number of top poses to overlay.

    Returns HTML string.
    """
    vid = 'viewer_' + uuid.uuid4().hex[:8]
    prot_b64 = base64.b64encode(protein_pdb_data.encode()).decode()

    colors = ['#2ECC71', '#00BCD4', '#E91E63', '#FF9800', '#FFEB3B']
    all_models = poses_pdbqt_data.split('MODEL')

    js_lines = [
        f'var el=document.getElementById("{vid}");',
        'var v=$3Dmol.createViewer(el,{backgroundColor:"white"});',
        f'v.addModel(atob("{prot_b64}"),"pdb");',
        'v.setStyle({model:0},{cartoon:{color:"white",opacity:0.4}});',
    ]

    # Contact analysis on top pose — added as model 1
    pose_strs = []
    for i in range(min(n_poses, len(all_models) - 1)):
        pose_strs.append('MODEL' + all_models[i + 1].split('ENDMDL')[0] + 'ENDMDL')

    next_model = 1
    if pose_strs:
        contact_js, next_model = _generate_contact_js(
            protein_pdb_data, pose_strs[0], pocket_model_idx=1
        )
        js_lines.extend(contact_js)

    # Add pose models
    first_pose_model = next_model
    for i, pose in enumerate(pose_strs):
        clean = _clean_pdbqt_for_viewer(pose)
        pose_b64 = base64.b64encode(clean.encode()).decode()
        color = colors[i % len(colors)]
        js_lines.append(f'v.addModel(atob("{pose_b64}"),"pdb");')
        js_lines.append(
            f'v.setStyle({{model:{next_model}}},{{stick:{{color:"{color}",radius:0.18}}}});'
        )
        next_model += 1

    js_lines.append(f'v.zoomTo({{model:{first_pose_model}}});v.zoom(0.8);v.render();')
    js_block = '\n'.join(js_lines)

    return _wrap_3dmol_iframe(vid, js_block, width, height)


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
