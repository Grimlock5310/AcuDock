# AcuDock - Molecular Docking Made Easy

AcuDock is a collection of Google Colab notebooks that predict how drug-like molecules bind to protein targets. Think of it as a virtual lab bench where you can test thousands of potential drug compounds against a disease target -- without synthesizing a single molecule.

**No installation required.** Everything runs in your web browser through Google Colab (free).

**Interactive widget interface** -- no code editing needed. Each notebook provides built-in controls (sliders, dropdowns, text fields) where you fill in values and click buttons. The code is hidden by default so you can focus on the science. Optional **GPU-accelerated docking** via Uni-Dock for 1000x+ speedup on large screening runs.

---

## Table of Contents

1. [What Is Molecular Docking?](#what-is-molecular-docking)
2. [Before You Start](#before-you-start)
3. [Which Notebook Should I Use?](#which-notebook-should-i-use)
4. [QuickDock - Getting Started](#quickdock---getting-started)
5. [AcuDock Pro - Full-Featured Docking](#acudock-pro---full-featured-docking)
6. [AcuDock Scout - Large-Scale Screening](#acudock-scout---large-scale-screening)
7. [Key Concepts Explained](#key-concepts-explained)
8. [How to Find Your Protein Target](#how-to-find-your-protein-target)
9. [How to Get Molecule SMILES](#how-to-get-molecule-smiles)
10. [Understanding Your Results](#understanding-your-results)
11. [Troubleshooting](#troubleshooting)
12. [File Reference](#file-reference)
13. [Testing and Validation](#testing-and-validation)
14. [FAQ](#faq)

---

## What Is Molecular Docking?

Imagine you have a lock (a protein involved in a disease) and you want to find the right key (a drug molecule) that fits into it. Molecular docking is a computer simulation that:

1. Takes a 3D model of the protein (the lock)
2. Takes a 3D model of a small molecule (a potential key)
3. Tries thousands of orientations to find the best fit
4. Scores each fit to predict how tightly the molecule binds

**Why is this useful?** Testing a real drug molecule in a lab costs thousands of dollars and takes weeks. Docking lets you screen millions of molecules in hours on a computer, narrowing down which ones are worth actually testing.

**What you get:** A ranked list of molecules with binding scores (in kcal/mol). More negative scores mean tighter predicted binding. A score of -10 kcal/mol suggests strong binding (nanomolar range), while -5 kcal/mol suggests weak or no meaningful binding.

---

## Before You Start

### What You Need

- A **Google account** (for Google Colab access)
- A **web browser** (Chrome recommended)
- Your **target protein's PDB ID** (a 4-character code like `1HSG` -- see [How to Find Your Protein Target](#how-to-find-your-protein-target))
- Your **molecule(s) as SMILES strings** (a text code that describes a molecule's structure -- see [How to Get Molecule SMILES](#how-to-get-molecule-smiles))

### What You Do NOT Need

- Any software installation
- Programming experience -- all three notebooks are controlled entirely through interactive widgets
- A powerful computer (everything runs on Google's servers)
- A paid account (the free tier of Colab is sufficient)

### Quick Start (5 minutes)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook > GitHub**
3. Paste this repository's URL and select `AcuDock_QuickDock.ipynb`
4. Click **Runtime > Run all** to run every cell at once
5. The first cell installs dependencies and the runtime will **automatically restart** -- this is normal
6. After restart, click **Runtime > Run all** again (the install cell will detect packages are already present and skip itself)
7. An interactive widget panel will appear -- fill in your PDB ID and SMILES, then click "Run Docking"

---

## Which Notebook Should I Use?

| If you want to... | Use this notebook | Difficulty | Time |
|---|---|---|---|
| Try docking for the first time | **QuickDock** | Beginner | ~10 min |
| Dock one molecule against one protein | **QuickDock** or **Pro** | Beginner | ~10 min |
| Test a handful of molecules (<50) | **QuickDock** or **Pro** (Batch tab) | Beginner | ~15 min |
| Get the most accurate scoring | **AcuDock Pro** (with Gnina CNN) | Intermediate | ~15 min |
| Test one molecule against multiple proteins | **QuickDock** or **Pro** (Multi-Protein tab) | Beginner | ~15 min |
| Screen a large library (100+) efficiently | **AcuDock Scout** | Intermediate | ~30-60 min |

All three notebooks hide code by default and present a clean widget interface. You never need to edit code unless you want to.

---

## QuickDock - Getting Started

**File:** `AcuDock_QuickDock.ipynb`

QuickDock is the simplest way to get started with molecular docking. It provides an interactive tabbed interface with three modes: single docking, batch screening, and multi-protein docking.

### How to Use QuickDock

#### Step 1: Open and install

- Open `AcuDock_QuickDock.ipynb` in [Google Colab](https://colab.research.google.com/)
- Click **Runtime > Run all** -- the first cell installs dependencies (~2-3 min)
- The runtime will **auto-restart** after installation -- this is normal

#### Step 2: Launch the interface

- After restart, click **Runtime > Run all** again
- A tabbed widget panel will appear with three tabs: **Single Dock**, **Batch**, and **Multi-Protein**

#### Step 3: Dock your molecule

In the **Single Dock** tab:
- **PDB ID**: Enter your protein's 4-character code (default: `1HSG` for HIV-1 protease)
- **Ligand SMILES**: Enter your molecule's SMILES string (default: Indinavir, a known HIV drug)
- **Ligand Name**: A label for your molecule
- **Engine**: Choose `Vina (CPU)` or `Uni-Dock (GPU)` from the dropdown
- Adjust **Exhaustiveness** and **Box Size** sliders if needed (defaults work well)
- Click **"Run Docking"** and wait for results

#### Step 4: Read your results

Results appear in the right panel:
- **Score table**: Each binding pose ranked by energy (most negative = best)
- **3D viewer**: Interactive visualization -- click and drag to rotate, scroll to zoom
- **CSV download**: Link to download results for Excel or other tools

#### Step 5: Batch screening (optional)

Switch to the **Batch** tab to test multiple molecules at once. Enter compounds as `Name,SMILES` pairs (one per line) or use the built-in example set.

#### Step 6: Multi-protein docking (optional)

Switch to the **Multi-Protein** tab to test one molecule against several protein targets. Enter PDB IDs one per line. You can optionally specify active site residues for each target (e.g., `1HSG:23,24,25,26`).

---

## AcuDock Pro - Full-Featured Docking

**Files:** `AcuDock_Pro.ipynb` + `acudock_utils.py`

AcuDock Pro adds **Gnina CNN rescoring** on top of Vina. Gnina uses a neural network trained on thousands of known protein-drug complexes to re-evaluate each binding pose, improving the success rate of identifying the correct pose from about 58% to about 73%.

### How to Use AcuDock Pro

#### Step 1: Open and install

- Open `AcuDock_Pro.ipynb` in Google Colab
- Click **Runtime > Run all** -- installation takes ~2-3 minutes
- Runtime auto-restarts; click **Runtime > Run all** again

#### Step 2: Use the widget panel

The interface has three tabs: **Single Dock**, **Batch**, and **Multi-Protein**. The Single Dock tab includes:

| Control | What It Does | Default |
|---------|-------------|---------|
| **PDB ID** | Your protein's 4-character code | `1HSG` |
| **Ligand SMILES** | Your molecule in SMILES format | Indinavir |
| **Ligand Name** | A label for your molecule | `Indinavir` |
| **Exhaustiveness** | How thorough the search is (8-128) | 32 |
| **Num Poses** | How many binding poses to generate (5-50) | 20 |
| **Box Size** | Search area around the binding site, in Angstroms (15-40) | 20 |
| **Scoring** | `Vina`, `Vina + Gnina CNN`, or `Consensus` | Vina |
| **Engine** | `Vina (CPU)` or `Uni-Dock (GPU)` | Vina (CPU) |

#### Step 3: Choose a scoring mode

- **Vina**: Fast, reliable baseline scoring. Works on CPU. Good for most uses.
- **Vina + Gnina CNN**: Docks with Vina, then re-evaluates each pose with a neural network for more accurate ranking. Requires a GPU runtime.
- **Consensus**: Combines Vina and Gnina scores using z-score weighting for the most robust ranking.

To use Gnina, make sure your Colab runtime has a GPU:
1. Go to **Runtime > Change runtime type**
2. Select **T4 GPU** under Hardware accelerator
3. Click **Save**

#### Step 4: Run and review

Click **"Run Docking"** to start. Results include a score table, 3D visualization of the best pose, and a downloadable CSV.

---

## AcuDock Scout - Large-Scale Screening

**Files:** `AcuDock_Scout.ipynb` + `acudock_surrogate.py` + `acudock_screening.py`

AcuDock Scout is designed for screening large compound libraries (hundreds to thousands of molecules). Instead of docking every single molecule (which could take days), it uses **active learning** -- a machine learning technique that intelligently picks which molecules to dock next, finding over 90% of the top hits while only docking a fraction of the library.

### How Active Learning Works

1. **Start**: Randomly pick a small batch of molecules and dock them (the "bootstrap")
2. **Learn**: Train a fast ML model to predict docking scores based on molecular structure
3. **Select**: The model identifies the most promising un-docked molecules -- balancing compounds it predicts will score well with compounds it is uncertain about (to avoid missing surprises)
4. **Dock**: Only dock those selected molecules
5. **Repeat**: Retrain the model with the new results, select the next batch, and so on

Think of it like a talent scout who watches a few auditions, learns what "good" looks like, and gets increasingly better at picking who to audition next -- skipping the obvious misses.

### How to Use AcuDock Scout

#### Step 1: Open and install

- Open `AcuDock_Scout.ipynb` in Google Colab
- Click **Runtime > Run all** -- installation takes ~2-3 minutes
- Runtime auto-restarts; click **Runtime > Run all** again

#### Step 2: Configure your campaign

The widget panel includes controls for the protein target and the active learning parameters:

| Parameter | Default | What It Means |
|-----------|---------|---------------|
| **PDB ID** | `1HSG` | Your protein target |
| **Active Site Residues** | `23,24,25,...` | Amino acid positions that define the binding pocket (optional) |
| **Bootstrap Size** | 100 | How many molecules to dock randomly at the start to seed the ML model |
| **Batch Size** | 50 | How many molecules to dock per learning cycle |
| **AL Cycles** | 3 | How many learn-select-dock cycles to run |
| **UCB Beta** | 1.5 | Controls the explore/exploit tradeoff -- higher values explore more diverse chemistry, lower values focus on predicted top scorers |
| **Exhaustiveness** | 8 | Lower than Pro/QuickDock defaults because Scout docks many molecules |
| **Engine** | Vina (CPU) | Choose `Uni-Dock (GPU)` for faster screening on GPU runtimes |

#### Step 3: Choose a compound library

Type one of the following in the **Library** field:

- **`demo`** (default): ~500 compounds derived from 33 common drugs (Aspirin, Ibuprofen, Caffeine, etc.) with structural variations. Good for testing.
- **`acyl-thiourea`**: ~500 acyl thiourea compounds (R-C(=O)-NH-C(=S)-NH-R') built from 24 acyl groups and 20 amine substituents, including metal-chelating variants. Useful for studying thiourea-based inhibitors.
- **Custom**: Paste `Name,SMILES` pairs (one per line) directly into the text box, or upload a CSV file with `SMILES` and `Name` columns.

#### Step 4: Run the campaign

Click **"Start Campaign"**. The notebook will:

1. Prepare the protein target
2. **Cycle 0 (Bootstrap)**: Dock a random sample to seed the ML model
3. **Cycles 1-N**: Select the most promising molecules, dock them, retrain the model
4. Generate results including convergence plots, score distributions, and a ranked hit list

#### Step 5: Interpret the results

- **Convergence plot**: Shows how quickly the best score was found versus how much of the library was docked. Ideally you find top hits after docking less than 10-20% of the library.
- **Score distribution**: Compares randomly selected (bootstrap) molecules versus ML-selected molecules. The ML-selected batch should have much better (more negative) scores.
- **Surrogate model performance**: Shows how well the ML model predicts docking scores (R-squared and RMSE over cycles). Higher R-squared means the model is learning effectively.
- **Top 20 hits table**: Your best molecules ranked by binding score, with molecular properties (weight, LogP, hydrogen bond donors/acceptors).
- **3D viewer**: Interactive visualization of the top hit bound to the protein.

---

## Key Concepts Explained

### Proteins and PDB IDs

Proteins are large molecules in your body that perform specific functions. In diseases, certain proteins malfunction or help pathogens (like viruses) replicate. Drug molecules work by binding to these proteins and blocking their activity.

Every experimentally-determined protein structure is stored in the **Protein Data Bank (PDB)** at [rcsb.org](https://www.rcsb.org/). Each structure has a unique 4-character code called a **PDB ID**. For example:
- `1HSG` = HIV-1 protease (an HIV drug target)
- `6LU7` = SARS-CoV-2 main protease (a COVID-19 drug target)
- `4LDE` = CDK2 kinase (a cancer drug target)

### SMILES Strings

SMILES (Simplified Molecular Input Line Entry System) is a way to write a molecule's structure as text. For example:
- Water: `O`
- Ethanol: `CCO`
- Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
- Caffeine: `Cn1c(=O)c2c(ncn2C)n(C)c1=O`

You do not need to write SMILES by hand. See [How to Get Molecule SMILES](#how-to-get-molecule-smiles) for easy ways to look them up.

### Binding Energy (kcal/mol)

The docking score is measured in kilocalories per mole (kcal/mol). It estimates how much energy is released when the molecule binds to the protein. **More negative = stronger binding.**

| Score Range | Binding Strength | Approximate Potency | What It Means |
|------------|-----------------|--------------------| --------------|
| -5 to -6 | Very weak | Millimolar (mM) | Probably not useful as a drug |
| -6 to -7 | Weak-moderate | High micromolar | Marginal activity |
| -7 to -8 | Moderate | Low micromolar (uM) | Interesting -- worth investigating |
| -8 to -9 | Good | Sub-micromolar | Promising lead compound |
| -9 to -10 | Strong | Nanomolar (nM) | Strong drug candidate |
| < -10 | Very strong | Sub-nanomolar | Excellent, but verify (could be an artifact) |

**Important:** These are predictions with an error margin of about +/- 2 kcal/mol. A score of -8 could really be anywhere from -6 to -10. Always validate top hits with experimental testing.

### RMSD (Root Mean Square Deviation)

RMSD measures how far apart two molecular poses are, in Angstroms (A). It is used to validate docking accuracy:
- **RMSD < 2.0 A**: The docking successfully reproduced the known binding pose (good)
- **RMSD > 2.0 A**: The docking failed to find the correct pose (review your setup)

### Exhaustiveness

This controls how thoroughly Vina searches for binding poses. Higher values explore more orientations and are more likely to find the best pose, but take proportionally longer.

| Value | Use Case | Time per Molecule |
|-------|----------|-------------------|
| 8 | Quick screening of many molecules | ~10-30 seconds |
| 32 | Standard docking (recommended default) | ~1-3 minutes |
| 64 | Thorough docking for top candidates | ~3-8 minutes |
| 128 | Publication-quality results | ~10-20 minutes |

### Vina vs. Uni-Dock

Both engines use the same scoring function and produce comparable results. The difference is speed:

- **Vina (CPU)**: Runs on any Colab runtime. Docks one molecule at a time. Best for single molecules or small batches.
- **Uni-Dock (GPU)**: Runs on GPU runtimes (T4 or A100). Docks many molecules simultaneously on the GPU, achieving 1000x+ speedup for large batches. Best for Scout campaigns or large batch screens. Falls back to Vina automatically if no GPU is available.

To enable GPU: go to **Runtime > Change runtime type** and select **T4 GPU**.

---

## How to Find Your Protein Target

### Option 1: Search the Protein Data Bank

1. Go to [rcsb.org](https://www.rcsb.org/)
2. Type your protein name in the search bar (e.g., "HIV protease" or "EGFR kinase")
3. Browse results and find a structure with:
   - A **co-crystallized ligand** (shown as a small molecule in the structure) -- this helps validate your docking setup
   - Good **resolution** (lower is better; aim for < 2.5 A)
4. Note the **PDB ID** (the 4-character code like `1HSG`)

### Option 2: Common Drug Targets to Try

| Disease Area | Protein | PDB ID | Notes |
|-------------|---------|--------|-------|
| HIV/AIDS | HIV-1 protease | `1HSG` | Default in all notebooks. Classic benchmark. |
| COVID-19 | Main protease (Mpro) | `6LU7` | SARS-CoV-2 drug target |
| Cancer | CDK2 kinase | `4LDE` | Cell cycle regulation |
| Cancer | EGFR kinase | `1M17` | Lung cancer target |
| Diabetes | DPP-4 | `2ONC` | Type 2 diabetes |
| Inflammation | COX-2 | `3LN1` | Target of ibuprofen/naproxen |
| Bacterial infection | DHFR | `1DLS` | Antibiotic target |

### Option 3: Upload Your Own Structure

If you have a PDB file from AlphaFold or another source, you can upload it directly to Colab (use the file browser on the left side) and modify the notebook to load from the file path instead of a PDB ID.

---

## How to Get Molecule SMILES

### Option 1: Search PubChem (Easiest)

1. Go to [pubchem.ncbi.nlm.nih.gov](https://pubchem.ncbi.nlm.nih.gov/)
2. Search for your molecule by name (e.g., "ibuprofen")
3. Click on the result
4. Find the **Canonical SMILES** field in the "Names and Identifiers" section
5. Copy the SMILES string

### Option 2: Search ChEMBL

1. Go to [ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl/)
2. Search by compound name, target, or disease
3. Click on a compound to see its SMILES

### Option 3: Draw and Convert

1. Go to an online structure editor like [MolView](https://molview.org/)
2. Draw your molecule using the drawing tools
3. The SMILES will be displayed automatically

### Option 4: Common Drug SMILES for Testing

| Drug | SMILES |
|------|--------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Ibuprofen | `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` |
| Caffeine | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |
| Acetaminophen | `CC(=O)Nc1ccc(O)cc1` |
| Naproxen | `COc1ccc2cc([C@H](C)C(=O)O)ccc2c1` |
| Metformin | `CN(C)C(=N)NC(=N)N` |
| Penicillin G | `CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@@H]1C(=O)O` |

---

## Understanding Your Results

### What to Look For

1. **Best score**: The most negative score in your results. Scores below -7 kcal/mol are generally considered interesting.

2. **Pose consistency**: If the top 3-5 poses are clustered in the same location on the protein, the docking is more reliable. If they are scattered, the result is uncertain.

3. **Interactions**: In the 3D viewer, look for:
   - **Hydrogen bonds** (dashed lines) between the molecule and protein residues
   - **Hydrophobic contacts** with non-polar protein residues
   - Contact with **known important residues** in the binding site

4. **Molecular properties**: Good drug candidates typically follow "Lipinski's Rule of Five":
   - Molecular weight < 500
   - LogP < 5 (a measure of how "greasy" a molecule is)
   - Hydrogen bond donors < 5
   - Hydrogen bond acceptors < 10

### What NOT to Conclude

- A good docking score does **not** mean the molecule is a drug. It means it might bind to the target.
- Docking has an error margin of about +/- 2 kcal/mol. Do not over-interpret small score differences.
- These are **predictions**, not measurements. The only way to confirm binding is through laboratory experiments.

### Exporting and Sharing Results

All notebooks save results as CSV files that you can:
- Download to your computer (each results panel includes a download link)
- Open in Excel, Google Sheets, or any spreadsheet program
- Share with collaborators for further analysis

---

## Troubleshooting

### "ModuleNotFoundError" or "ImportError"

This is the most common issue. Packages like `vina` and `rdkit` contain compiled C code that Python can only load after a runtime restart. The install cell **automatically restarts the runtime** after installation. After the restart:

1. **Skip the install cell** (it already ran successfully)
2. Run from the **imports cell** (the second code cell) onward

If you still see the error:
1. Go to **Runtime > Restart session** manually
2. **Skip** the install cell
3. Run from the imports cell onward

Do **not** re-run the install cell after restart -- the packages are already installed.

### "SMILES parsing error" or "Invalid SMILES"

Your SMILES string has a typo or uses unsupported syntax. Try:
1. Paste your SMILES into [PubChem](https://pubchem.ncbi.nlm.nih.gov/) to verify it is valid
2. Make sure there are no extra spaces or line breaks in the SMILES
3. Use the **Canonical SMILES** (not the Isomeric SMILES) if available

### "RuntimeError" during docking

Common causes:
- **Box too small**: Increase the Box Size slider to 25 or 30
- **Box in wrong location**: Check that the active site residue numbers are correct for your protein
- **Protein not found**: Verify the PDB ID exists at [rcsb.org](https://www.rcsb.org/)

### py3Dmol visualization not showing

- Make sure you are running in Google Colab (not a local Jupyter notebook)
- Try restarting the runtime and running all cells again
- The visualization requires JavaScript, so it will not render in exported PDF/HTML

### "Session crashed" or "Out of memory"

- Reduce exhaustiveness or the number of poses
- For Scout: reduce bootstrap size and batch size
- Go to **Runtime > Change runtime type** and select a GPU runtime (T4 GPU has more memory)

### Gnina not found (AcuDock Pro)

Gnina needs to be downloaded separately. The notebook should do this automatically, but if it fails:
1. Run this in a code cell:
   ```
   !wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O /usr/local/bin/gnina
   !chmod +x /usr/local/bin/gnina
   ```
2. Make sure your runtime has a GPU enabled

### Docking takes too long

- Lower exhaustiveness to 8 for screening, 32 for individual molecules
- For Scout: reduce the number of cycles or batch size
- Switch to Uni-Dock (GPU) in the Engine dropdown for batch or multi-protein docking

---

## File Reference

| File | Purpose | When You Need It |
|------|---------|-----------------|
| `AcuDock_QuickDock.ipynb` | Single docking, batch screening, multi-protein | Getting started, general use |
| `AcuDock_Pro.ipynb` | Docking with Gnina CNN rescoring | When you need more accurate pose ranking |
| `AcuDock_Scout.ipynb` | Active learning virtual screening | Screening large compound libraries (100+) |
| `acudock_utils.py` | Shared docking and visualization functions | Loaded automatically by QuickDock and Pro |
| `acudock_surrogate.py` | ML surrogate model for active learning | Loaded automatically by Scout |
| `acudock_screening.py` | Batch docking manager | Loaded automatically by Scout |
| `requirements.txt` | Python package list | Reference only (notebooks install packages themselves) |
| `CLAUDE.md` | Project tracking and technical notes | Developers and contributors |

---

## Testing and Validation

Before trusting results on novel molecules, validate your docking setup using known protein-ligand structures.

### Quick Validation Test

Every notebook includes a built-in test case: **HIV-1 protease (1HSG) + Indinavir**. This is a well-studied system where the crystal structure of the protein-drug complex is known. Run the default settings first to make sure everything works.

**What "works" means:**
- All cells run without errors
- Docking produces scores in the -5 to -12 kcal/mol range
- The 3D visualization shows the ligand inside the protein's binding pocket
- For the 1HSG + Indinavir test, the top pose should have a score around -9 to -11 kcal/mol

### Redocking Validation (Recommended Before Real Experiments)

The gold standard test: take a protein that was crystallized with a drug molecule, remove the drug, and see if docking puts it back in the right place.

1. Choose a PDB structure that has a **co-crystallized ligand** (most entries on rcsb.org show this)
2. Note the ligand's identity and find its SMILES
3. Run docking with that protein and ligand
4. Check the **RMSD** values in the results table
5. **RMSD < 2.0 Angstroms** = your setup is validated and reliable

If the RMSD is above 2.0 A, check:
- Is the search box centered on the correct binding site?
- Is the box large enough to contain the binding pocket?
- Are the active site residue numbers correct for your protein?

### Benchmark Datasets

For rigorous testing, these public datasets are the standard in the field:

| Dataset | What It Contains | Where to Get It | Best For |
|---------|-----------------|-----------------|----------|
| PDBBind | ~19,500 protein-ligand complexes with real binding data | [pdbbind.org.cn](http://www.pdbbind.org.cn/) | Validating score accuracy |
| DUD-E | 102 targets with known binders and decoys | [dude.docking.org](https://dude.docking.org/) | Testing virtual screening |
| CASF-2016 | 285 standardized test cases | Part of PDBBind | Benchmarking against published methods |

---

## FAQ

**Q: Is this free?**
A: Yes. Google Colab's free tier provides enough compute for all three notebooks. AcuDock itself is MIT licensed (free and open source).

**Q: Do I need to know Python?**
A: No. All three notebooks use interactive widgets (sliders, text fields, buttons) and the code is hidden by default. You never need to read or edit code.

**Q: How accurate is this?**
A: AutoDock Vina (the primary docking engine) has a 90.2% docking power score on the CASF benchmark. However, the predicted binding energies have an error margin of about +/- 2 kcal/mol. This is a computational prediction tool, not a replacement for lab experiments.

**Q: Can I use this for a real drug discovery project?**
A: AcuDock is suitable for early-stage virtual screening to identify candidate molecules worth testing in the lab. It is used by academic researchers, students, and early-stage biotech teams. For clinical drug development, results should always be validated experimentally.

**Q: How long does docking take?**
A: A single molecule takes about 10 seconds to 3 minutes depending on exhaustiveness. Batch screening of 50 molecules at exhaustiveness 8 takes about 10-15 minutes. AcuDock Scout can process a 500-molecule demo library in about 30 minutes on CPU, or significantly faster with Uni-Dock GPU.

**Q: What if my Colab session disconnects?**
A: Google Colab sessions last 12-24 hours (free tier). If disconnected, you need to re-run the notebook from the beginning. To save progress, download your results CSV before the session ends. For long-running Scout campaigns, consider Google Colab Pro for longer session limits.

**Q: Can I dock against a protein that is not in the PDB?**
A: Yes. If you have a PDB file from AlphaFold, homology modeling, or another source, upload it to Colab and modify the protein preparation cell to load from the file path instead of fetching by PDB ID.

**Q: What is the difference between Vina and Gnina?**
A: Vina uses a mathematical formula (called a scoring function) to estimate binding energy. Gnina adds a neural network (trained on thousands of known protein-drug complexes) that re-evaluates each pose. Gnina improves the success rate of identifying the correct binding pose from about 58% to about 73%. Gnina requires a GPU runtime.

**Q: What is the difference between Vina and Uni-Dock?**
A: They use the same scoring function and produce equivalent results. Uni-Dock runs on the GPU and docks many molecules simultaneously, making it 1000x+ faster for large batches. For a single molecule there is no speed advantage. All notebooks let you choose between them via the Engine dropdown, and will automatically fall back to Vina if Uni-Dock is not available.

**Q: Can I run this on my own computer instead of Colab?**
A: Yes, but you will need to install Python and all the dependencies listed in `requirements.txt`. Google Colab is recommended because it handles all of this automatically.

---

## License

AcuDock is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it.
