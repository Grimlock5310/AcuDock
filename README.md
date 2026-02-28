# AcuDock - Molecular Docking Made Easy

AcuDock is a collection of Google Colab notebooks that predict how drug-like molecules bind to protein targets. Think of it as a virtual lab bench where you can test thousands of potential drug compounds against a disease target -- without synthesizing a single molecule.

**No installation required.** Everything runs in your web browser through Google Colab (free).

---

## Table of Contents

1. [What Is Molecular Docking?](#what-is-molecular-docking)
2. [Before You Start](#before-you-start)
3. [Which Notebook Should I Use?](#which-notebook-should-i-use)
4. [QuickDock - Getting Started (Beginners)](#quickdock---getting-started-beginners)
5. [AcuDock Pro - Interactive Mode (Intermediate)](#acudock-pro---interactive-mode-intermediate)
6. [AcuDock Scout - Large-Scale Screening (Advanced)](#acudock-scout---large-scale-screening-advanced)
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
- Programming experience (especially for AcuDock Pro)
- A powerful computer (everything runs on Google's servers)
- A paid account (the free tier of Colab is sufficient)

### Quick Start (5 minutes)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook > GitHub**
3. Paste this repository's URL and select `AcuDock_QuickDock.ipynb`
4. Click **Runtime > Run all** from the menu bar
5. Wait ~5 minutes for installation and docking to complete
6. Scroll down to see your results and 3D visualization

---

## Which Notebook Should I Use?

| If you want to... | Use this notebook | Difficulty | Time |
|---|---|---|---|
| Try docking for the first time | **QuickDock** | Beginner | ~10 min |
| Dock without editing any code | **AcuDock Pro** | Intermediate | ~10 min |
| Test a handful of molecules (<50) | **QuickDock** or **Pro** | Beginner | ~15 min |
| Screen a large library (100+) | **AcuDock Scout** | Advanced | ~30-60 min |
| Get the most accurate results | **AcuDock Pro** (with Gnina CNN) | Intermediate | ~15 min |
| Learn how docking works | **QuickDock** | Beginner | ~10 min |

---

## QuickDock - Getting Started (Beginners)

**File:** `AcuDock_QuickDock.ipynb`

QuickDock is a step-by-step notebook that walks you through the entire docking process. Each step has an explanation followed by the code that runs it.

### How to Use QuickDock

#### Step 1: Open the notebook

- Go to [Google Colab](https://colab.research.google.com/)
- Upload `AcuDock_QuickDock.ipynb` via **File > Upload notebook**, or open it from GitHub

#### Step 2: Install packages (automatic)

- The first code cell installs all required software
- This takes about 2-3 minutes -- you only need to do this once per session
- Click the play button on the cell, or use **Runtime > Run all** to run everything

#### Step 3: Set your target

Find the **Configuration** cell and change these values to match your experiment:

```python
# Target protein PDB ID (example: HIV-1 protease)
PDB_ID = '1HSG'

# Ligand SMILES (example: Indinavir - HIV protease inhibitor)
LIGAND_SMILES = 'CC(C)(C)NC(=O)...'
LIGAND_NAME = 'Indinavir'
```

- **PDB_ID**: The 4-character code for your protein from the [Protein Data Bank](https://www.rcsb.org/). The default `1HSG` is HIV-1 protease, a classic test target.
- **LIGAND_SMILES**: The SMILES string for your molecule (see [How to Get Molecule SMILES](#how-to-get-molecule-smiles)).
- **LIGAND_NAME**: Any name you want to give your molecule (for labeling results).

#### Step 4: Adjust docking parameters (optional)

```python
EXHAUSTIVENESS = 32      # How thorough the search is (8=quick, 32=standard, 64=very thorough)
N_POSES = 20             # How many binding poses to generate
BOX_SIZE = [20, 20, 20]  # Size of the search area in Angstroms
```

- **EXHAUSTIVENESS**: Higher values find better results but take longer. Use `8` for quick tests, `32` for standard runs, `64` for publication-quality.
- **N_POSES**: Number of different binding orientations to report. 20 is a good default.
- **BOX_SIZE**: The 3D box (in Angstroms) that defines where the program searches for binding. `[20, 20, 20]` works for most binding sites. Increase to `[25, 25, 25]` or `[30, 30, 30]` for larger binding pockets.

#### Step 5: Run all cells

Click **Runtime > Run all** from the menu. The notebook will:

1. Download your protein from the PDB
2. Clean and prepare the protein (add hydrogens, fix missing atoms)
3. Convert your SMILES into a 3D molecule
4. Run the docking simulation
5. Show a 3D visualization of the best binding pose
6. Display a table of all poses with scores

#### Step 6: Read your results

- **Score table**: Shows each pose ranked by binding energy (most negative = best)
- **3D viewer**: Interactive visualization -- click and drag to rotate, scroll to zoom
- **CSV export**: Download your results for use in Excel or other tools

#### Step 7: Batch screening (optional)

To test multiple molecules at once, edit the compound library in the **Batch Screening** section:

```python
COMPOUND_LIBRARY = [
    ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'),
    ('Ibuprofen', 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'),
    ('Your_Molecule', 'your_SMILES_string_here'),
]
```

Each entry is a pair of `('Name', 'SMILES')`. Add as many as you like.

---

## AcuDock Pro - Interactive Mode (Intermediate)

**Files:** `AcuDock_Pro.ipynb` + `acudock_utils.py`

AcuDock Pro gives you a point-and-click interface with sliders and dropdown menus. You do not need to edit any code. It also adds **Gnina CNN rescoring**, which uses a neural network to improve the accuracy of binding predictions.

### How to Use AcuDock Pro

#### Step 1: Open and install

- Open `AcuDock_Pro.ipynb` in Google Colab
- Run the first two cells to install packages and import libraries
- This takes about 2-3 minutes

#### Step 2: Use the widget panel

After running the widget cell, you will see interactive controls:

| Control | What It Does | Default |
|---------|-------------|---------|
| **PDB ID** | Text field for your protein code | `1HSG` |
| **SMILES** | Text area for your molecule | Indinavir |
| **Ligand Name** | Label for your molecule | `Indinavir` |
| **Exhaustiveness** | Slider (8-128) -- how thorough the search is | 32 |
| **Num Poses** | Slider (5-50) -- how many poses to return | 20 |
| **Box Size** | Slider (15-40 Angstroms) -- search area size | 20 |
| **Scoring Engine** | Dropdown: Vina / Vina + Gnina CNN / Gnina only | Vina |

Simply change the values in the widgets, then run the subsequent cells.

#### Step 3: Choose a scoring engine

- **Vina**: Fast, reliable baseline scoring. Works on CPU. Good for most uses.
- **Vina + Gnina CNN**: Docks with Vina, then re-evaluates each pose with a neural network. More accurate ranking (improves success rate from ~58% to ~73%). Requires GPU.
- **Gnina only**: Uses the neural network for everything. Best accuracy but slower.

To use Gnina, make sure your Colab runtime has a GPU:
1. Go to **Runtime > Change runtime type**
2. Select **T4 GPU** under Hardware accelerator
3. Click **Save**

#### Step 4: Run the cells in order

Each section runs automatically when you execute the cell:
- **Protein Preparation**: Downloads and cleans your protein
- **Ligand Preparation**: Converts your SMILES to 3D
- **Docking Execution**: Runs the docking (progress bar shown)
- **Results**: Score table, 3D visualization, interaction analysis

#### Step 5: Batch screening

The batch section at the bottom lets you screen multiple compounds. Edit the compound list or upload a CSV file, then run those cells.

---

## AcuDock Scout - Large-Scale Screening (Advanced)

**Files:** `AcuDock_Scout.ipynb` + `acudock_surrogate.py` + `acudock_screening.py`

AcuDock Scout is designed for screening large compound libraries (hundreds to millions of molecules). Instead of docking every single molecule (which could take days), it uses **active learning** -- a machine learning technique that intelligently picks which molecules to dock next, finding over 90% of the top hits while only docking a fraction of the library.

### How Active Learning Works (Simple Explanation)

1. **Start**: Randomly pick a small batch of molecules and dock them
2. **Learn**: Train a fast ML model to predict docking scores from molecular structure
3. **Select**: Use the ML model to find the most promising un-docked molecules
4. **Dock**: Only dock those promising molecules
5. **Repeat**: Retrain the model with the new results, select the next batch, and so on

This is like a librarian who, after reading a few books from each shelf, learns to predict which remaining books are worth reading -- without opening every single one.

### How to Use AcuDock Scout

#### Step 1: Open and install

- Open `AcuDock_Scout.ipynb` in Google Colab
- Run the first two cells for installation (~2-3 minutes)

#### Step 2: Configure your campaign

Edit the configuration cell:

```python
# Target protein
PDB_ID = '1HSG'

# Active learning parameters
BOOTSTRAP_SIZE = 100    # How many molecules to dock randomly at the start
BATCH_SIZE = 50         # How many molecules to dock per cycle
N_CYCLES = 3            # How many learning cycles to run
UCB_BETA = 1.5          # Exploration vs. exploitation (see below)
```

**Parameter guide:**

| Parameter | Demo Value | Real Screening Value | What It Means |
|-----------|-----------|---------------------|---------------|
| BOOTSTRAP_SIZE | 100 | 1,000-5,000 | More = better initial model, but takes longer |
| BATCH_SIZE | 50 | 500-2,000 | More per cycle = faster but less efficient |
| N_CYCLES | 3 | 5-20 | More cycles = better coverage |
| UCB_BETA | 1.5 | 0.5-3.0 | Low = focus on predicted good molecules. High = explore diverse molecules. 1.5 is a good balance. |

#### Step 3: Provide your compound library

The notebook includes a **demo library** of ~500 molecules for testing. For real screening, replace it with your own library:

**Option A: Edit the SMILES list in the notebook**
```python
my_library = [
    ('Compound_1', 'CCO'),
    ('Compound_2', 'c1ccccc1'),
    # ... add more
]
```

**Option B: Upload a CSV file**
Prepare a CSV file with columns `SMILES` and `Name` (or `ID`), then upload it in Colab using the file browser on the left sidebar. Update the library loading cell to point to your file.

#### Step 4: Run the campaign

Click **Runtime > Run all**. The notebook will:

1. Prepare the protein target
2. **Cycle 0**: Dock a random bootstrap sample
3. Train the ML surrogate model
4. **Cycles 1-N**: Select the most promising molecules, dock them, retrain
5. Show convergence plots, score distributions, and top hits

#### Step 5: Interpret the results

The notebook generates several analysis plots:

- **Convergence plot**: Shows what percentage of top hits were found versus how much of the library was docked. The goal is to hit 90%+ while docking less than 10-20%.
- **Score distribution**: Compares randomly selected molecules versus ML-selected molecules. The ML-selected batch should have much better (more negative) scores.
- **Top hits table**: Your best molecules, ranked by binding score with molecular properties.
- **Chemical diversity**: Shows how structurally diverse your top hits are (you want diverse hits, not many variations of the same molecule).
- **Feature importance**: Shows which molecular features the ML model uses to predict good binders.

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

3. **Interactions**: In the 3D viewer and interaction analysis, look for:
   - **Hydrogen bonds** (shown as dashed lines) between the molecule and protein residues
   - **Hydrophobic contacts** with non-polar protein residues
   - Contact with **known important residues** in the binding site

4. **Molecular properties**: Good drug candidates typically follow "Lipinski's Rule of Five":
   - Molecular weight < 500
   - LogP < 5 (not too greasy)
   - Hydrogen bond donors < 5
   - Hydrogen bond acceptors < 10

### What NOT to Conclude

- A good docking score does **not** mean the molecule is a drug. It means it might bind to the target.
- Docking has an error margin of about +/- 2 kcal/mol. Do not over-interpret small score differences.
- These are **predictions**, not measurements. The only way to confirm binding is through laboratory experiments.

### Exporting and Sharing Results

All notebooks save results as CSV files that you can:
- Download to your computer (the notebooks include download code)
- Open in Excel, Google Sheets, or any spreadsheet program
- Share with collaborators for further analysis

---

## Troubleshooting

### "ModuleNotFoundError" or "ImportError"

The installation cell did not complete successfully. Try:
1. Go to **Runtime > Restart runtime**
2. Run the installation cell again (the first code cell with `pip install`)
3. Then run the import cell

### "SMILES parsing error" or "Invalid SMILES"

Your SMILES string has a typo or uses unsupported syntax. Try:
1. Paste your SMILES into [PubChem](https://pubchem.ncbi.nlm.nih.gov/) to verify it is valid
2. Make sure there are no extra spaces or line breaks in the SMILES
3. Use the **Canonical SMILES** (not the Isomeric SMILES) if available

### "RuntimeError" during docking

Common causes:
- **Box too small**: Increase `BOX_SIZE` to `[25, 25, 25]` or `[30, 30, 30]`
- **Box in wrong location**: Check that `ACTIVE_SITE_RESIDUES` matches your protein
- **Protein not found**: Verify the PDB ID exists at [rcsb.org](https://www.rcsb.org/)

### py3Dmol visualization not showing

- Make sure you are running in Google Colab (not a local Jupyter notebook)
- Try restarting the runtime and running all cells again
- The visualization requires JavaScript, so it will not render in exported PDF/HTML

### "Session crashed" or "Out of memory"

- Reduce `N_POSES` or `EXHAUSTIVENESS`
- For Scout: reduce `BOOTSTRAP_SIZE` and `BATCH_SIZE`
- Go to **Runtime > Change runtime type** and select a GPU runtime (T4 GPU has more memory)

### Gnina not found (AcuDock Pro)

Gnina needs to be downloaded separately. The notebook should do this automatically, but if it fails:
1. Run this cell manually:
   ```
   !wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O /usr/local/bin/gnina
   !chmod +x /usr/local/bin/gnina
   ```
2. Make sure your runtime has a GPU enabled

### Docking takes too long

- Lower `EXHAUSTIVENESS` to `8` for screening, `32` for individual molecules
- For batch screening, use `exhaustiveness=8` (the notebooks already do this)
- For Scout: reduce the number of cycles or batch size

---

## File Reference

| File | Purpose | When You Need It |
|------|---------|-----------------|
| `AcuDock_QuickDock.ipynb` | Step-by-step docking tutorial | First time users, learning docking |
| `AcuDock_Pro.ipynb` | Interactive docking with widgets | Point-and-click docking, Gnina CNN |
| `AcuDock_Scout.ipynb` | Large-scale virtual screening | Screening 100+ molecules |
| `acudock_utils.py` | Shared functions for Pro | Loaded automatically by Pro notebook |
| `acudock_surrogate.py` | ML model for Scout | Loaded automatically by Scout notebook |
| `acudock_screening.py` | Batch docking for Scout | Loaded automatically by Scout notebook |
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
A: For QuickDock, you only need to change a few text values (the PDB ID and SMILES). For AcuDock Pro, you do not need to touch any code at all -- everything is controlled by widgets. For Scout, basic familiarity with Python helps but is not required if you just change the configuration values.

**Q: How accurate is this?**
A: AutoDock Vina (the docking engine used here) has a 90.2% docking power score on the CASF benchmark. However, the predicted binding energies have an error margin of about +/- 2 kcal/mol. This is a computational prediction tool, not a replacement for lab experiments.

**Q: Can I use this for a real drug discovery project?**
A: AcuDock is suitable for early-stage virtual screening to identify candidate molecules worth testing in the lab. It is used by academic researchers, students, and early-stage biotech teams. For clinical drug development, results should always be validated experimentally.

**Q: How long does docking take?**
A: A single molecule takes about 10 seconds to 3 minutes depending on the `EXHAUSTIVENESS` setting. Batch screening of 50 molecules at `exhaustiveness=8` takes about 10-15 minutes. AcuDock Scout can process a 500-molecule demo library in about 30 minutes.

**Q: What if my Colab session disconnects?**
A: Google Colab sessions last 12-24 hours (free tier). If disconnected, you need to re-run the notebook from the beginning. To save progress, download your results CSV before the session ends. For long-running Scout campaigns, consider Google Colab Pro for longer session limits.

**Q: Can I dock against a protein that is not in the PDB?**
A: Yes. If you have a PDB file from AlphaFold, homology modeling, or another source, upload it to Colab and modify the protein preparation cell to load from the file path instead of fetching by PDB ID.

**Q: What is the difference between Vina and Gnina?**
A: Vina uses a mathematical formula to estimate binding energy. Gnina adds a neural network (trained on thousands of known protein-drug complexes) that re-evaluates each pose. Gnina improves the success rate of identifying the correct binding pose from about 58% to about 73%. Gnina requires a GPU.

**Q: Can I run this on my own computer instead of Colab?**
A: Yes, but you will need to install Python and all the dependencies listed in `requirements.txt`. Google Colab is recommended because it handles all of this automatically.

---

## License

AcuDock is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it.
