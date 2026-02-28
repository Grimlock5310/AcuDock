# AcuDock - Project Tracking & Reference

## Project Overview

**AcuDock** is a molecular docking application for computational drug discovery, designed to run on Google Colab. It predicts how drug-like molecules (ligands) bind to protein targets, enabling virtual screening of thousands-to-millions of compounds before costly synthesis.

**License:** MIT
**Platform:** Google Colab (free tier: T4 GPU, 12-24h sessions; Pro: A100)
**Core Stack:** AutoDock Vina (Python API) + Meeko + RDKit + PDBFixer + py3Dmol

---

## Current Status

- [x] Phase 0: Research compilation (3 markdown files)
- [x] Phase 1: Approach design and ranking (5 approaches, 8 criteria)
- [x] Phase 2: CLAUDE.md tracking file
- [x] Phase 3: Build QuickDock notebook (Approach 1)
- [x] Phase 4: Build AcuDock Pro notebook (Approach 2)
- [x] Phase 5: Build AcuDock Scout notebook (Approach 5)
- [ ] Phase 6: Testing and validation in Colab
- [ ] Phase 7: Documentation and examples

---

## What Has Been Done

| Date | Action | Details |
|------|--------|---------|
| 2026-02-27 | Research analysis | Cross-referenced 3 markdown files, identified shared concepts |
| 2026-02-27 | Approach design | Designed 5 novel notebook approaches with distinct architectures |
| 2026-02-27 | Ranking matrix | Scored all 5 approaches across 8 criteria (speed, accuracy, UX, etc.) |
| 2026-02-27 | Implementation | Built top 3 approaches: QuickDock, AcuDock Pro, AcuDock Scout |

---

## What Needs to Be Done

### High Priority
- [ ] Test all notebooks in Google Colab environment
- [ ] Validate Vina docking pipeline with known PDB targets (e.g., 1HSG, 4LDE)
- [ ] Verify dependency installation on fresh Colab runtime
- [ ] Test py3Dmol 3D visualization rendering

### Medium Priority
- [ ] Add Gnina binary download and CNN rescoring (AcuDock Pro)
- [ ] Implement full active learning convergence tracking (AcuDock Scout)
- [ ] Build remaining approaches (AcuDock Suite, AcuDock ML)
- [ ] Add PoseBusters validation for ML-generated poses
- [ ] Create example compound libraries (ZINC subset, DUD-E actives)

### Low Priority
- [ ] Add ipywidgets interactivity to all notebooks
- [ ] Create video tutorials / walkthrough documentation
- [ ] Benchmark against CASF-2016 core set
- [ ] Implement consensus scoring across all engines
- [ ] Add Google Drive persistence for large datasets

---

## Key Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary docking engine | AutoDock Vina | Apache 2.0 license, Python API, Colab-compatible, 90.2% docking power (CASF) |
| Visualization library | py3Dmol | Only 3D viewer fully compatible with Colab (NGLView not supported) |
| Protein preparation | PDBFixer + OpenMM | Handles missing atoms, hydrogens at pH 7.4, non-standard residues |
| Ligand preparation | RDKit + Meeko | SMILES -> 3D (ETKDGv3), Gasteiger charges, PDBQT conversion |
| CNN rescoring | Gnina | 73% redocking success vs 58% Vina; pre-built binary available |
| ML docking engine | DiffDock | 38% blind docking top-1 success; diffusion-based pose generation |
| Active learning surrogate | Random Forest (primary) | Fast training, interpretable uncertainty, works with Morgan FPs |
| Interaction analysis | ProLIF | Generates protein-ligand interaction fingerprints for pose comparison |
| Top 3 build priority | QuickDock, Pro, Scout | Covers beginner -> intermediate -> advanced; diverse architectures |

---

## Technical Notes

### Vina Score Interpretation
- Score is in kcal/mol (more negative = better)
- -7 kcal/mol ~ 7 uM K_d (moderate)
- -10 kcal/mol ~ 50 nM K_d (strong)
- Error margin: ~2 kcal/mol at best
- Always validate by redocking co-crystallized ligand (RMSD < 2A = success)

### Critical: Preparation Quality > Algorithm Choice
All three research files agree: the quality of protein/ligand preparation has more impact on docking accuracy than the choice of scoring function or search algorithm. Key steps:
1. Add hydrogens at pH 7.4 (PROPKA/PDBFixer)
2. Careful histidine protonation states
3. Remove waters (except catalytic)
4. Fill missing residues/atoms
5. Generate 3D ligand conformations (ETKDGv3)
6. Enumerate relevant tautomers/protonation states

### Colab Constraints
- Free tier: Tesla T4 (16GB VRAM), ~12h session limit
- Pro tier: A100 (40-80GB VRAM), ~24h session limit
- Persistent storage: Google Drive mount at `/content/drive/`
- No Docker without workarounds
- ipywidgets partially supported (Colab forms preferred for reliability)
- pip install works; conda requires `condacolab` package

### Gnina Installation on Colab
```bash
# Download pre-built Gnina binary
wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O /usr/local/bin/gnina
chmod +x /usr/local/bin/gnina
```

---

## Research File Cross-Reference

### compass_artifact_wf-*.md (File 1 - "Molecular Docking Programs Guide")
- **Primary source for:** 7-step Vina pipeline with code, three-tier architecture
- **10 programs profiled:** Vina, AutoDock4, Glide, GOLD, Gnina, DOCK6, rDock, SwissDock, LeDock, PLANTS
- **Key data:** CASF benchmarks, PDBBind evaluation, DL methods (DiffDock, Uni-Mol)
- **Code snippets:** PDBFixer pipeline, RDKit ligand prep, Vina Python API, py3Dmol viewer
- **Conclusion:** Three-tier architecture (Vina -> Gnina -> DiffDock), consensus scoring for publication

### deep-research-report (7).md (File 2 - "Executive Summary")
- **Primary source for:** 7 prototype Colab workflows, Colab constraints, software comparison table
- **7 workflows:** Quick Vina, ML-Rescoring, MD Refinement, Fragment Growth, DiffDock, Ensemble, Active Learning
- **8 programs compared:** SwissDock, Vina, Glide, GOLD, DOCK, RosettaLigand, rDock, PLANTS
- **Colab section:** GPU types, session limits, installable tools, visualization options, security considerations
- **Key data:** Novel hybrid methods (differentiable docking, physics-informed generative, multi-modal learning)

### deep-research-report (8).md (File 3 - "Emerging AI Approaches")
- **Primary source for:** 8 novel AI ideas ranked, implementation roadmaps, DiffDock-Glide hybrid
- **8 ideas ranked:** Equivariant Neural, Flexible Diffusion, Fragment Growth, Ensemble+AI, Active Learning, Hybrid ML/Physics, Foundation Model, MD Refinement
- **Top 3 selected:** Flexible Diffusion, Foundation Models, Hybrid ML/Physics
- **Key data:** UniMol 77% poses <2A, VideoMol from 120M conformer videos, HASTEN >90% hits with <10% docking
- **Roadmap:** 4-6 month prototype timeline with Gantt chart for top 3 ideas

---

## Five Notebook Approaches Designed

### Ranking Matrix (1-5 scale, 5=best)

| Criterion | 1: QuickDock | 2: AcuDock Pro | 3: Suite | 4: ML Hybrid | 5: Scout |
|---|:---:|:---:|:---:|:---:|:---:|
| Speed to create | 5 | 3 | 1 | 2 | 3 |
| Docking accuracy | 2 | 4 | 4 | 5 | 3 |
| Ease of use | 4 | 5 | 2 | 2 | 3 |
| Extensibility | 2 | 3 | 5 | 3 | 4 |
| Colab compatibility | 5 | 4 | 3 | 2 | 4 |
| Educational value | 4 | 2 | 3 | 4 | 3 |
| Novelty | 1 | 2 | 3 | 5 | 5 |
| Scalability | 2 | 2 | 3 | 2 | 5 |
| **TOTAL** | **25** | **25** | **24** | **25** | **30** |

**Top 3 built:** Scout (30), QuickDock (25), AcuDock Pro (25)

---

## File Manifest

| File | Status | Description |
|------|--------|-------------|
| `LICENSE` | Complete | MIT license |
| `CLAUDE.md` | Complete | This tracking file |
| `compass_artifact_wf-*.md` | Research | Complete Colab development guide (File 1) |
| `deep-research-report (7).md` | Research | 7 prototype workflows + software comparison (File 2) |
| `deep-research-report (8).md` | Research | 8 AI ideas + implementation roadmaps (File 3) |
| `requirements.txt` | Complete | Shared Python dependencies |
| `AcuDock_QuickDock.ipynb` | Complete | Approach 1: Single linear Vina pipeline |
| `AcuDock_Pro.ipynb` | Complete | Approach 2: Interactive widgets + Gnina CNN |
| `acudock_utils.py` | Complete | Shared utilities for AcuDock Pro |
| `AcuDock_Scout.ipynb` | Complete | Approach 5: Active learning virtual screening |
| `acudock_surrogate.py` | Complete | ML surrogate model for Scout |
| `acudock_screening.py` | Complete | Batch docking manager for Scout |

---

## Dependencies

```
# Core docking
vina>=1.2.5
meeko>=0.5.0
rdkit>=2023.9.1

# Protein preparation
pdbfixer>=1.9
openmm>=8.0.0

# Visualization
py3Dmol>=2.0.0
prolif>=2.0.0

# Data handling
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0

# Cheminformatics
openbabel-wheel>=3.1.1

# ML (Scout notebook)
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Protein retrieval
prody>=2.4.0

# Interactive (Pro notebook)
ipywidgets>=8.0.0
```

---

## Useful Commands (Colab Quick Reference)

```python
# Install all dependencies
!pip install -q vina meeko rdkit prody py3Dmol prolif openbabel-wheel pdbfixer pandas scikit-learn matplotlib seaborn

# Fetch protein
from prody import fetchPDB
fetchPDB('1HSG', folder='/content/')

# Quick Vina dock
from vina import Vina
v = Vina(sf_name='vina')
v.set_receptor('/content/receptor.pdbqt')
v.set_ligand_from_file('/content/ligand.pdbqt')
v.compute_vina_maps(center=[x, y, z], box_size=[20, 20, 20])
v.dock(exhaustiveness=32, n_poses=20)
v.write_poses('/content/poses.pdbqt', n_poses=5)

# 3D visualization
import py3Dmol
view = py3Dmol.view(width=800, height=600)
view.addModel(open('/content/receptor.pdb').read(), 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()
```
