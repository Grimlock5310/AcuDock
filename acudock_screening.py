"""
AcuDock Screening - Batch docking manager for active learning virtual screening.

Manages the docking workflow: library loading, batch preparation,
Vina execution, and result aggregation. Works with acudock_surrogate.py
for ML-guided compound selection.
"""

import os
import time
import random

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs


class BatchDockingManager:
    """Manages batch molecular docking for virtual screening campaigns.

    Handles compound library loading, protein preparation, batch Vina
    docking, and result collection. Designed for iterative active learning
    workflows where only a subset of the library is docked per cycle.
    """

    def __init__(self, receptor_pdbqt, center, box_size,
                 exhaustiveness=8, n_poses=5,
                 output_dir='/content/acudock_scout',
                 use_unidock=False):
        """Initialize the batch docking manager.

        Args:
            receptor_pdbqt: Path to prepared receptor PDBQT file.
            center: Binding site center [x, y, z] in Angstroms.
            box_size: Search box dimensions [x, y, z] in Angstroms.
            exhaustiveness: Vina exhaustiveness (8=fast screening, 32=thorough).
            n_poses: Number of poses per ligand.
            output_dir: Directory for output files.
            use_unidock: If True, use Uni-Dock GPU for batch docking.
                Falls back to Vina if Uni-Dock is unavailable or fails.
        """
        self.receptor_pdbqt = receptor_pdbqt
        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.output_dir = output_dir
        self.use_unidock = use_unidock

        os.makedirs(output_dir, exist_ok=True)

        # Track all docking results
        self.all_results = pd.DataFrame()
        self.docked_smiles = set()
        self.total_dock_time = 0.0
        self.total_docked = 0

    def load_library(self, source, max_compounds=None, shuffle=True):
        """Load a compound library from various sources.

        Args:
            source: Either a file path (CSV/SMI/SDF), a list of SMILES,
                    or a list of (name, SMILES) tuples.
            max_compounds: Maximum number to load (None = all).
            shuffle: Randomize order (important for unbiased bootstrap).

        Returns list of (name, SMILES) tuples.
        """
        compounds = []

        if isinstance(source, list):
            for item in source:
                if isinstance(item, tuple) and len(item) == 2:
                    compounds.append(item)
                elif isinstance(item, str):
                    compounds.append((f'Compound_{len(compounds)}', item))
        elif isinstance(source, str) and os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(source)
                smiles_col = None
                name_col = None
                for col in df.columns:
                    cl = col.lower()
                    if cl in ('smiles', 'smi', 'canonical_smiles'):
                        smiles_col = col
                    elif cl in ('name', 'id', 'compound_id', 'title'):
                        name_col = col
                if smiles_col is None:
                    smiles_col = df.columns[0]
                for i, row in df.iterrows():
                    name = row[name_col] if name_col else f'Compound_{i}'
                    compounds.append((str(name), str(row[smiles_col])))
            elif ext in ('.smi', '.smiles'):
                with open(source, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            compounds.append((parts[1], parts[0]))
                        elif len(parts) == 1:
                            compounds.append((f'Compound_{i}', parts[0]))

        # Validate SMILES
        valid = []
        for name, smi in compounds:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid.append((name, smi))

        if shuffle:
            random.shuffle(valid)

        if max_compounds is not None:
            valid = valid[:max_compounds]

        return valid

    def random_sample(self, library, n):
        """Draw a random sample from the library (for bootstrap cycle).

        Excludes compounds already docked.
        """
        available = [(name, smi) for name, smi in library
                     if smi not in self.docked_smiles]
        n = min(n, len(available))
        return random.sample(available, n)

    def _prepare_ligand_pdbqt(self, smiles, output_path):
        """Prepare a single ligand SMILES into a PDBQT file.

        Handles metal-containing compounds (chelates) by skipping force field
        optimization and falling back to OpenBabel if Meeko fails.

        Returns the output_path on success, None on failure.
        """
        import meeko
        from acudock_utils import _has_metal_atoms, _prepare_ligand_obabel

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        has_metal = _has_metal_atoms(mol)
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if has_metal:
            params.useRandomCoords = True
            params.maxIterations = 500
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            params2 = AllChem.ETKDG()
            params2.useRandomCoords = True
            AllChem.EmbedMolecule(mol, params2)

        if not has_metal:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                except Exception:
                    pass
        # else: skip FF optimization for metals (MMFF/UFF don't support them)

        try:
            preparator = meeko.MoleculePreparation()
            mol_setup_list = preparator.prepare(mol)
            pdbqt_string = meeko.PDBQTWriterLegacy.write_string(mol_setup_list[0])
            if pdbqt_string and pdbqt_string[0].strip():
                with open(output_path, 'w') as f:
                    f.write(pdbqt_string[0])
                return output_path
        except Exception:
            pass

        # Fallback to OpenBabel for metals / Meeko failures
        if _prepare_ligand_obabel(smiles, output_path):
            return output_path

        return None

    def _mol_descriptors(self, smiles, best_score=None):
        """Compute molecular descriptors for a SMILES string.

        Uses acudock_utils.get_ligand_properties() for extended metrics.
        """
        try:
            from acudock_utils import get_ligand_properties
            return get_ligand_properties(smiles, docking_score=best_score)
        except ImportError:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            return {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
            }

    def dock_batch(self, compounds, progress_callback=None):
        """Dock a batch of (name, SMILES) compounds.

        If use_unidock is True, prepares all ligands first and submits
        them as a single Uni-Dock GPU batch call (this is where the
        1000x+ speedup comes from). Falls back to sequential Vina if
        Uni-Dock is unavailable or fails.

        Args:
            compounds: List of (name, SMILES) tuples.
            progress_callback: Optional callable(current, total, name, score).

        Returns DataFrame with columns [Name, SMILES, Best_Score, MW, LogP, ...].
        """
        # Filter already-docked compounds
        todo = [(name, smi) for name, smi in compounds
                if smi not in self.docked_smiles]

        if not todo:
            return pd.DataFrame()

        if self.use_unidock:
            return self._dock_batch_unidock(todo, progress_callback)
        return self._dock_batch_vina(todo, progress_callback)

    def _dock_batch_unidock(self, compounds, progress_callback=None):
        """Dock a batch using Uni-Dock GPU (all ligands in one call).

        Prepares all ligand PDBQTs first, then submits them as a single
        Uni-Dock batch. Falls back to Vina on any failure.
        """
        from acudock_utils import run_unidock, parse_pdbqt_scores

        # Step 1: Prepare all ligand PDBQTs
        lig_dir = os.path.join(self.output_dir, 'unidock_ligands')
        os.makedirs(lig_dir, exist_ok=True)

        prepared = []  # (name, smiles, pdbqt_path)
        for i, (name, smiles) in enumerate(compounds):
            if progress_callback:
                progress_callback(i + 1, len(compounds), name, 'preparing...')
            try:
                safe_name = name.replace('/', '_').replace(' ', '_')[:50]
                pdbqt_path = os.path.join(lig_dir, f'{safe_name}_{i}.pdbqt')
                result = self._prepare_ligand_pdbqt(smiles, pdbqt_path)
                if result:
                    prepared.append((name, smiles, pdbqt_path))
            except Exception:
                pass

        if not prepared:
            return pd.DataFrame()

        # Step 2: Run Uni-Dock batch
        pdbqt_files = [p for _, _, p in prepared]
        t0 = time.time()
        try:
            unidock_results = run_unidock(
                self.receptor_pdbqt, pdbqt_files,
                center=self.center, box_size=self.box_size,
                exhaustiveness=self.exhaustiveness,
                num_modes=self.n_poses,
                output_dir=self.output_dir
            )
        except Exception as e:
            # Uni-Dock failed entirely — fall back to Vina
            if progress_callback:
                progress_callback(0, len(compounds), '',
                                  f'Uni-Dock failed ({e}), falling back to Vina...')
            return self._dock_batch_vina(compounds, progress_callback)

        batch_time = time.time() - t0

        # Step 3: Parse results
        results = []
        per_mol_time = batch_time / max(1, len(prepared))

        for idx, (name, smiles, pdbqt_path) in enumerate(prepared):
            basename = os.path.basename(pdbqt_path)
            scores = unidock_results.get(basename, [])

            if scores and scores[0][0] <= 0:
                best_score = scores[0][0]
                desc = self._mol_descriptors(smiles, best_score=best_score)
                results.append({
                    'Name': name, 'SMILES': smiles,
                    'Best_Score': best_score,
                    'MW': desc.get('MW'), 'LogP': desc.get('LogP'),
                    'HBD': desc.get('HBD'), 'HBA': desc.get('HBA'),
                    'QED': desc.get('QED'), 'SA_Score': desc.get('SA_Score'),
                    'PAINS_Count': desc.get('PAINS_Count', 0),
                    'LE': desc.get('LE'), 'LLE': desc.get('LLE'),
                    'Num_Poses': len(scores),
                    'Dock_Time_s': round(per_mol_time, 2),
                })
                self.docked_smiles.add(smiles)
                self.total_docked += 1
                self.total_dock_time += per_mol_time
            else:
                results.append({
                    'Name': name, 'SMILES': smiles,
                    'Best_Score': None, 'MW': None, 'LogP': None,
                    'HBD': None, 'HBA': None,
                    'QED': None, 'SA_Score': None,
                    'PAINS_Count': None, 'LE': None, 'LLE': None,
                    'Num_Poses': 0, 'Dock_Time_s': round(per_mol_time, 2),
                })

            if progress_callback:
                score_str = f'{results[-1]["Best_Score"]:.2f}' if results[-1]['Best_Score'] else 'failed'
                progress_callback(idx + 1, len(prepared), name, score_str)

        batch_df = pd.DataFrame(results)
        self.all_results = pd.concat(
            [self.all_results, batch_df], ignore_index=True
        )
        return batch_df

    def _dock_batch_vina(self, compounds, progress_callback=None):
        """Dock a batch sequentially using Vina CPU.

        Computes Vina affinity maps once and reuses them for all ligands.
        """
        from vina import Vina

        results = []

        v = Vina(sf_name='vina')
        v.set_receptor(self.receptor_pdbqt)
        pdbqt_path = os.path.join(self.output_dir, 'tmp_ligand.pdbqt')
        maps_ready = False

        for i, (name, smiles) in enumerate(compounds):
            if smiles in self.docked_smiles:
                continue

            t0 = time.time()
            try:
                result_path = self._prepare_ligand_pdbqt(smiles, pdbqt_path)
                if result_path is None:
                    continue

                v.set_ligand_from_file(pdbqt_path)
                if not maps_ready:
                    v.compute_vina_maps(center=self.center, box_size=self.box_size)
                    maps_ready = True
                v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)

                energies = v.energies()
                best_score = energies[0][0]
                desc = self._mol_descriptors(smiles, best_score=best_score)

                result = {
                    'Name': name, 'SMILES': smiles,
                    'Best_Score': best_score,
                    'MW': desc.get('MW'), 'LogP': desc.get('LogP'),
                    'HBD': desc.get('HBD'), 'HBA': desc.get('HBA'),
                    'QED': desc.get('QED'), 'SA_Score': desc.get('SA_Score'),
                    'PAINS_Count': desc.get('PAINS_Count', 0),
                    'LE': desc.get('LE'), 'LLE': desc.get('LLE'),
                    'Num_Poses': len(energies),
                    'Dock_Time_s': round(time.time() - t0, 2),
                }
                results.append(result)
                self.docked_smiles.add(smiles)
                self.total_docked += 1
                self.total_dock_time += time.time() - t0

                if progress_callback:
                    progress_callback(i + 1, len(compounds), name, best_score)

            except Exception as e:
                results.append({
                    'Name': name, 'SMILES': smiles,
                    'Best_Score': None, 'MW': None, 'LogP': None,
                    'HBD': None, 'HBA': None,
                    'QED': None, 'SA_Score': None,
                    'PAINS_Count': None, 'LE': None, 'LLE': None,
                    'Num_Poses': 0,
                    'Dock_Time_s': round(time.time() - t0, 2),
                })

        batch_df = pd.DataFrame(results)
        self.all_results = pd.concat(
            [self.all_results, batch_df], ignore_index=True
        )

        return batch_df

    def get_all_results(self, sort_results=True):
        """Return all docking results collected so far.

        Args:
            sort_results: If True, sort by Best_Score ascending (best first).
        """
        df = self.all_results.copy()
        if sort_results and not df.empty:
            df = df.dropna(subset=['Best_Score'])
            df = df.sort_values('Best_Score', ascending=True).reset_index(drop=True)
        return df

    def get_top_hits(self, n=50, score_threshold=None):
        """Return the top N hits from all docked compounds.

        Args:
            n: Number of top hits to return.
            score_threshold: Optional cutoff (e.g., -7.0 kcal/mol).
        """
        df = self.get_all_results(sort_results=True)
        if score_threshold is not None:
            df = df[df['Best_Score'] <= score_threshold]
        return df.head(n)

    def get_statistics(self):
        """Return screening campaign statistics."""
        df = self.all_results.dropna(subset=['Best_Score'])
        if df.empty:
            return {'total_docked': 0}

        return {
            'total_docked': self.total_docked,
            'total_time_s': round(self.total_dock_time, 1),
            'avg_time_per_mol_s': round(self.total_dock_time / max(1, self.total_docked), 2),
            'best_score': round(df['Best_Score'].min(), 2),
            'mean_score': round(df['Best_Score'].mean(), 2),
            'std_score': round(df['Best_Score'].std(), 2),
            'n_strong_hits': int((df['Best_Score'] <= -8.0).sum()),
            'n_moderate_hits': int((df['Best_Score'] <= -7.0).sum()),
        }


def compute_tanimoto_matrix(smiles_list, fp_radius=2, fp_bits=2048):
    """Compute pairwise Tanimoto similarity matrix for SMILES list.

    Returns a numpy array of shape (n, n) with similarity values.
    """
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            fps.append(fp)
            valid_idx.append(i)

    n = len(fps)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix, valid_idx


def cluster_hits(smiles_list, threshold=0.5):
    """Cluster compounds by Tanimoto similarity (simple leader-picker).

    Groups compounds where the first compound in each cluster is the
    'leader' and all others within the threshold are members.

    Returns list of cluster assignments (0-indexed cluster IDs).
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        else:
            fps.append(None)

    clusters = [-1] * len(smiles_list)
    leaders = []

    for i, fp in enumerate(fps):
        if fp is None:
            continue

        assigned = False
        for cluster_id, leader_fp in enumerate(leaders):
            sim = DataStructs.TanimotoSimilarity(fp, leader_fp)
            if sim >= threshold:
                clusters[i] = cluster_id
                assigned = True
                break

        if not assigned:
            clusters[i] = len(leaders)
            leaders.append(fp)

    return clusters
