# Molecular docking programs: a complete guide for Colab development

Molecular docking predicts how small molecules bind to protein targets—the computational cornerstone of modern drug discovery. **Docking programs solve the problem of evaluating millions of compounds against a protein without synthesizing a single molecule**, outputting ranked binding poses and affinity estimates in kcal/mol. This report covers the full landscape of docking tools, algorithms, emerging AI methods, and practical implementation details needed to build a Google Colab notebook for compound testing. The field is undergoing a paradigm shift: deep learning methods like DiffDock and co-folding models like AlphaFold3 are challenging traditional physics-based approaches, yet classical tools like AutoDock Vina and Glide remain superior for physically valid predictions. For a Colab implementation, the optimal stack is **Vina (Python API) + Meeko + RDKit + PDBFixer + py3Dmol**, with Gnina for CNN-enhanced scoring and DiffDock as an ML alternative.

## How docking works: search, score, rank

Molecular docking combines two core components: a **search algorithm** that explores how a ligand can orient within a protein binding site, and a **scoring function** that estimates binding strength for each pose. A ligand with *n* rotatable bonds has approximately 3^*n* conformations, each requiring evaluation across 6 degrees of freedom (3 translational + 3 rotational), making exhaustive search intractable.

**Search algorithms** fall into three families. Systematic methods include exhaustive grid search (computationally prohibitive) and anchor-and-grow incremental construction (used by DOCK6 and Glide), which places a rigid fragment first, then grows flexible portions. Stochastic methods dominate modern docking: genetic algorithms (GOLD, AutoDock4), iterated local search with BFGS optimization (AutoDock Vina), Monte Carlo with Metropolis acceptance (SwissDock), simulated annealing (LeDock), and ant colony optimization (PLANTS). Glide uses a unique hierarchical funnel—rough shape matching, OPLS-AA grid minimization, then Monte Carlo refinement—achieving thoroughness without brute force.

**Scoring functions** determine prediction quality. Force-field-based functions (DOCK6's AMBER Score, Glide's OPLS-AA) calculate van der Waals and electrostatic energies from molecular mechanics but neglect solvation and entropy. Empirical functions (Vina, GlideScore, ChemPLP) are regression-calibrated against experimental binding data, summing weighted terms for hydrogen bonding, hydrophobic contact, and rotatable bond penalties. Knowledge-based functions (GOLD ASP, DrugScore) derive statistical potentials from atom-pair frequencies in the PDB. ML-based scoring (Gnina CNN, RF-Score, OnionNet) trains neural networks or random forests on structural features, achieving **Pearson correlations of R ≈ 0.81–0.83** versus R ≈ 0.60–0.68 for classical functions on CASF-2016 benchmarks.

**What scores mean in practice**: Vina reports estimated ΔG in kcal/mol (more negative = stronger binding). A score of **−7 kcal/mol ≈ K_d of ~7 μM**; **−10 kcal/mol ≈ ~50 nM**. Glide GScore follows the same convention. GOLD scores are dimensionless fitness values where higher is better. Gnina returns three values: Vina affinity, CNNscore (0–1 probability of good pose), and CNNaffinity (predicted pK). All docking scores carry inherent error of ~2 kcal/mol at best and cannot reliably rank-order compounds with similar affinities.

## Receptor and ligand preparation drive result quality

Preparation steps are arguably more important than the docking algorithm itself. **Receptor preparation** begins with importing a PDB structure (resolution <2.5 Å preferred), adding hydrogens, and assigning protonation states at pH 7.4 using tools like PROPKA or PDBFixer. Key residues—His (HID/HIE/HIP), Asp, Glu, Cys—require careful protonation assignment because errors propagate directly into incorrect scores. Crystallographic waters are typically removed except structurally important bridging waters (GOLD and Glide XP can retain these). Missing atoms and loops must be filled, alternate conformations resolved, and the structure energy-minimized to remove steric clashes (Schrödinger's Protein Prep Wizard minimizes to a maximum RMSD of 0.3 Å).

**Ligand preparation** generates 3D coordinates from SMILES (RDKit's ETKDG algorithm), adds explicit hydrogens, enumerates protonation states and tautomers at physiological pH, assigns partial charges (Gasteiger for AutoDock, MMFF for Glide), and converts to the required format (PDBQT for Vina/AutoDock, MOL2 for PLANTS/rDock, SDF for Gnina). Meeko handles the SMILES-to-PDBQT pipeline for Vina workflows.

**Rigid versus flexible docking** represents a critical tradeoff. The most common approach—flexible ligand, rigid receptor—explores ligand torsions while freezing the protein. Partial receptor flexibility (side-chain rotations for key residues) is supported by GOLD, Vina, and Gnina. Induced Fit Docking (Schrödinger IFD: Glide + Prime) captures larger conformational changes. Ensemble docking across multiple receptor conformations from MD or crystal structures offers a practical compromise for full flexibility.

**Blind docking versus targeted docking**: Blind docking searches the entire protein surface, useful when the binding site is unknown. SwissDock excels here (54.5% top-1 success on 251 complexes), and QuickVina-W was purpose-built for wide search boxes. Targeted docking places a grid box around a known or predicted binding site and is standard for most applications, offering much higher accuracy and speed.

## The current docking toolkit: 10 programs compared

The landscape spans open-source academic tools, commercial industry standards, and emerging ML-powered programs. Here is a technical profile of each.

**AutoDock Vina** remains the most widely used free docking tool. Its iterated local search combines random perturbations with BFGS local optimization and an empirical scoring function trained on PDBbind. It achieves **49% top-scored pose success** (RMSD <2 Å), runs ~100× faster than AutoDock4, and has an excellent Python API (`pip install vina`). GPU-accelerated variants include Vina-GPU 2.1 (65× speedup, 342% improvement in enrichment) and Uni-Dock (>1,000× speedup for batch screening). Vina is Apache 2.0 licensed and Colab-ready.

**AutoDock4** uses a Lamarckian genetic algorithm with a semi-empirical free energy scoring function requiring pre-computed AutoGrid maps. Slower (37.4% success rate) but offers maximum parameter control. AutoDock-GPU provides ~56× CUDA acceleration. Open source.

**Glide** (Schrödinger) is the industry gold standard. Its hierarchical search funnel operates in three modes: HTVS (~2 sec/compound), SP (~10 sec, 53.8% success), and XP (~2 min, 57.8% success, with explicit water desolvation penalties). GlideScore XP correlates with experiment at ~2.26 kcal/mol RMSD. Commercial, expensive, but available on Schrödinger Cloud.

**GOLD** uses a genetic algorithm with four scoring functions (ChemPLP, GoldScore, ChemScore, ASP). It handles protein side-chain flexibility via CSD rotamer libraries, water molecules, metal ions, and extensive constraints. **59.8% top-scored success**—highest among commercial programs. ChemPLP achieves ~78% for drug-like compounds. Commercial (CCDC).

**Gnina** forks Vina's search algorithm but integrates 3D convolutional neural networks for scoring. CNN rescoring boosts redocking success from **58% (Vina) to 73%** and cross-docking from 27% to 37%. It won CACHE Challenge #1 (a real-world drug discovery benchmark). Gnina 1.3 (2025) migrated to PyTorch, added knowledge-distilled CNNs for faster HTVS, and introduced covalent docking. Open source, GPU-capable, and **Colab-ready** with official notebooks.

**DOCK6** pioneered anchor-and-grow incremental construction. It offers AMBER Score with an MD engine for receptor flexibility and de novo design (DOCK_DN). Free academic license from UCSF.

**rDock/RxDock** performs cavity-based docking with genetic algorithm + simplex minimization. Designed for massive parallelization—used for SARS-CoV-2 screening via VirtualFlow. Open source (LGPL), excellent for RNA-ligand docking.

**SwissDock** (2024 version) replaced EADock DSS with a dual engine combining Attracting Cavities and AutoDock Vina. Free web-based service with blind docking capability. Best for quick jobs when binding site is unknown.

**LeDock** combines simulated annealing with a genetic algorithm and knowledge-based scoring. Achieves **57.4% top-scored success**—highest among free programs in the Wang et al. 2016 benchmark—while running fast. Free for academic use, closed source.

**PLANTS** uniquely uses ant colony optimization. ChemPLP scoring achieves up to 87% success on the Astex diverse set. **Best performer for metalloprotein targets (80% success)** and nucleic acid docking. VirtualFlow Ants enables deployment on up to 128,000 vCPUs on Google Cloud. Free for academic use.

| Program | Algorithm | Success Rate | License | Colab |
|---------|-----------|-------------|---------|-------|
| AutoDock Vina | Iterated local search | 49% | Open source | ✅ |
| Gnina | Vina + CNN | 73% (redocking) | Open source | ✅ (GPU) |
| Glide SP/XP | Hierarchical funnel | 54–58% | Commercial | ❌ |
| GOLD | Genetic algorithm | 60% | Commercial | ❌ |
| LeDock | SA + GA | 57% | Free academic | ✅ |
| PLANTS | Ant colony | 87% (Astex) | Free academic | ✅ |
| rDock/RxDock | GA + simplex | 50% | Open source | ✅ |
| DOCK6 | Anchor-and-grow | 73% (Astex) | Free academic | Possible |
| SwissDock | Attracting Cavities + Vina | 55% (blind) | Free web | Web only |

## Deep learning is reshaping docking (2022–2025)

**DiffDock** (Corso et al., ICLR 2023) reframed docking as a generative modeling problem. A diffusion model operates over the product space of translational (ℝ³), rotational (SO(3)), and torsional (∏SO(2)) degrees of freedom. The reverse diffusion process uses an equivariant graph neural network to iteratively denoise randomly sampled initial poses, while a separate confidence model ranks outputs. DiffDock reported **38% top-1 success** on PDBbind blind docking—but this advantage shrinks when binding sites are known: Surflex-Dock (68%), Glide (67%), and Gnina (58%) all outperform DiffDock (45%) in that setting. DiffDock-L (ICLR 2024) improved generalization to novel pockets.

**The PoseBusters reality check** (Buttenschoen et al., Chemical Science 2024) revealed that DL methods frequently produce physically implausible poses—steric clashes, distorted bond lengths, chirality inversions. **Over 50% of DL-generated poses fail stringent validity checks.** When requiring both RMSD <2 Å AND physical validity, classical methods (GOLD, Vina) outperform most DL approaches. Uni-Mol Docking V2 (May 2024) was a breakthrough, achieving **77% on PoseBusters with 75%+ passing all quality checks**—the first ML method to match traditional tools on physical plausibility.

**Co-folding models** represent the biggest paradigm shift. AlphaFold3 (Nature 2024), Chai-1, Boltz-1/2, and Protenix predict full protein-ligand complex structures from sequence + ligand, bypassing docking entirely. AF3 achieves ~73% PB-valid on PoseBusters; Chai-1 reaches 77% (81% with constraints). However, the "Runs N' Poses" benchmark (2025, 2,600 post-cutoff complexes) showed **all co-folding methods largely memorize training data**, with accuracy declining sharply on dissimilar complexes. None recapitulate conformational rearrangements like peptide flips or loop openings.

**ML-enhanced scoring** offers a pragmatic middle ground. Gnina's CNN rescoring improves Vina's virtual screening performance on 89/117 DUD-E + LIT-PCBA targets. RF-Score-VS achieves **55.6% hit rate in the top 1%** versus 16.2% for Vina. OnionNet-SFCT combines rotation-invariant CNN features with Vina scoring via AdaBoost random forest. The general pattern: ML scoring functions reach R ≈ 0.81–0.83 on CASF-2016, a **~35% improvement** over classical functions, but some studies suggest they learn dataset biases rather than generalizable physics.

**Other emerging methods include**: MM/GBSA rescoring (62.2% success for identifying native-like poses, Spearman correlations of 0.63–0.66); ensemble docking across multiple receptor conformations; covalent docking (CovDock, GOLD, HCovDock at 70.5% success, Gnina 1.3); generative de novo design models (DiffSBDD, TargetDiff, Pocket2Mol, PILOT); physics-informed networks (PIGNet2); QM/MM docking for metal-binding complexes; and active learning for ultra-large screening where **94.8% of top-50,000 ligands can be identified after docking only 2.4%** of a 100M library.

## Building the Colab notebook: a practical workflow

The recommended technology stack for a Colab docking notebook centers on **AutoDock Vina's Python API** supplemented by RDKit, Meeko, PDBFixer, and py3Dmol. Here is the complete workflow with specific implementation details.

**Installation** requires a single cell:
```
pip install vina meeko rdkit-pypi prody py3Dmol prolif openbabel-wheel
conda install -c conda-forge pdbfixer  # or pip with OpenMM
```

**Step 1 — Fetch protein**: Download from PDB using `urllib` (`https://files.rcsb.org/download/{pdb_id}.pdb`), ProDy's `fetchPDB()`, or BioPython's `PDBList`. Accept user upload via `google.colab.files.upload()`.

**Step 2 — Prepare protein**: PDBFixer handles the critical preparation: `findMissingResidues()` → `findNonstandardResidues()` → `replaceNonstandardResidues()` → `removeHeterogens(keepWater=False)` → `findMissingAtoms()` → `addMissingAtoms()` → `addMissingHydrogens(pH=7.4)`. Methods must be called in this exact order. Convert to PDBQT using Meeko's `mk_prepare_receptor.py`.

**Step 3 — Prepare ligand**: RDKit converts SMILES to 3D: `MolFromSmiles()` → `AddHs()` → `EmbedMolecule(AllChem.ETKDGv3())` → `MMFFOptimizeMolecule()`. Meeko converts to PDBQT: `MoleculePreparation().prepare(mol)` → `PDBQTWriterLegacy.write_string(setup)`. Note: the older `write_pdbqt_string()` method is deprecated.

**Step 4 — Define search box**: From a co-crystallized ligand centroid (extract coordinates with RDKit, add 5–10 Å padding); from known residue coordinates; from cavity detection (`fpocket -f protein.pdb`); or use the entire protein for blind docking with a large box.

**Step 5 — Run docking**: The Vina Python API is clean and direct:
```python
from vina import Vina
v = Vina(sf_name='vina')
v.set_receptor('receptor.pdbqt')
v.set_ligand_from_string(pdbqt_string)
v.compute_vina_maps(center=[x, y, z], box_size=[20, 20, 20])
v.dock(exhaustiveness=32, n_poses=20)
v.write_poses('results.pdbqt', n_poses=10)
```
Key parameters: `exhaustiveness` (default 8; use 32 for publication quality), `n_poses`, `energy_range` (kcal/mol window from best pose).

**Step 6 — Parse and visualize**: Meeko's `PDBQTMolecule` and `RDKitMolCreate` convert docked PDBQT poses back to RDKit molecules for analysis. py3Dmol renders interactive 3D views:
```python
view = py3Dmol.view(width=800, height=600)
view.addModel(protein_pdb, 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.addModel(ligand_sdf, 'sdf')
view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
view.zoomTo(); view.show()
```
ProLIF generates interaction fingerprints (H-bonds, hydrophobic, π-stacking) as pandas DataFrames. PLIP provides detailed interaction profiling.

**Step 7 — Batch screening**: Loop over a SMILES list, prepare each compound, dock, and collect scores into a DataFrame sorted by binding energy. For larger libraries (>1,000 compounds), consider Uni-Dock (>1,000× speedup on GPU) or Gnina with CNN rescoring.

**GPU acceleration**: Colab's free tier provides NVIDIA T4 (16 GB VRAM); Pro offers T4 or A100. Gnina's pre-built binary (`wget` from GitHub releases) runs CNN inference on GPU with the `--cnn rescore` flag (default, minimal overhead). DiffDock requires PyTorch + CUDA + ESM (T4 minimum). Uni-Dock achieves **~37,000 molecules/GPU-hour** on V100.

**Existing Colab notebooks** provide starting points: the official Vina examples at `autodock-vina.readthedocs.io/en/latest/colab_examples.html`; Gnina's official notebook; DiffDock community notebooks on GitHub and HuggingFace; the Jupyter_Dock repository (Vina, LeDock, SMINA, fpocket); LABODOCK's basic docking notebook; and Cloud-Bind's Uni-Dock + MD workflow.

## Benchmarks and databases every developer should know

**PDBbind** (v2020) provides the field's standard training and test data: ~19,500 complexes with experimental binding affinities (K_d, K_i, IC50). The refined set (~5,316 complexes, resolution <2.5 Å) trains scoring functions; the core set (285 complexes, 65 protein families) serves as the CASF test set. PDBbind ceased free availability after v2020, and concerns about training-set leakage led to LP-PDBBind as a leak-proof alternative.

**DUD-E** benchmarks virtual screening: 102 targets, ~22,432 actives, ~1.38M property-matched decoys. However, **analog bias** allows 2D fingerprint methods to achieve AUC >0.9 using ligand features alone, without protein information. LIT-PCBA (15 targets with real screening data) and DUD-Z (43 targets with reduced biases) are increasingly preferred.

**CASF-2016** evaluates four dimensions: scoring power (affinity correlation: best classical R ≈ 0.68, ML R ≈ 0.83), ranking power (same-target ligand ranking), docking power (pose prediction: Vina 90.2%, IGModel 95.3%), and screening power (active/decoy discrimination—the weakest dimension for all programs).

**ZINC-22** has grown to **~54.9 billion enumerated 2D molecules** (~5.9 billion with 3D structures), sourced primarily from Enamine REAL Space (34B), Enamine REAL Database (5B), and WuXi GalaXi (2.5B). Molecules are organized in tranches by physicochemical properties and available in ready-to-dock formats (PDBQT, MOL2, SDF). **ChEMBL** (v34) contains ~2.4M compounds with bioactivity data, queryable via Python API for retrieving target-specific IC50/Ki/Kd values. **PubChem** provides >116M compounds with bioassay data.

For validation, the field increasingly demands dual metrics: RMSD <2 Å **and** PoseBusters physical validity (bond lengths, stereochemistry, steric clashes). Protein-Ligand Interaction Fingerprint (PLIF) recovery is emerging as a third metric, revealing that even good-RMSD poses can miss critical interactions.

## Conclusion: building toward a production-ready notebook

The optimal Colab docking notebook should implement a **three-tier architecture**. The first tier uses Vina's Python API for fast, reliable docking with empirical scoring—handling the majority of use cases with ~49–90% pose prediction accuracy depending on the benchmark. The second tier adds Gnina for CNN-enhanced rescoring, boosting enrichment by 2–3× over Vina alone with minimal additional code (a single binary download). The third tier offers DiffDock as an ML alternative for blind docking or AlphaFold-predicted structures where traditional methods struggle.

The most important implementation insight is that **preparation quality matters more than algorithm choice**. PDBFixer's protonation and atom-repair pipeline, combined with Meeko's PDBQT conversion, handles the error-prone steps that most commonly cause docking failures. For visualization, py3Dmol renders interactive protein-ligand complexes directly in Colab cells. ProLIF converts docking results into analyzable interaction fingerprints.

Three practical recommendations stand out. First, always validate the docking protocol by redocking a co-crystallized ligand (target RMSD <2 Å) before screening novel compounds. Second, for libraries exceeding a few hundred compounds, switch to Uni-Dock on Colab's GPU for >1,000× throughput gains. Third, consensus scoring across 2–3 programs (Vina + Gnina + one other) reduces false positives more effectively than any single scoring improvement. The field's trajectory is clear: hybrid pipelines combining fast traditional docking with ML rescoring currently outperform either approach alone.