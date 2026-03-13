"""
AcuDock Report - PDF report generation for molecular docking results.

Generates professional PDF reports for single docking runs, batch screening
campaigns, multi-protein docking, and Scout active learning campaigns.
Uses reportlab for layout and RDKit for 2D structure rendering.

Designed for use in Google Colab with AcuDock notebooks.
"""

import os
import math
import datetime

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        Image, PageBreak,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = letter  # 612 x 792 points

# Color scheme
COLOR_PRIMARY = colors.HexColor('#1565C0')      # dark blue
COLOR_ACCENT = colors.HexColor('#00897B')        # teal
COLOR_LIGHT_BG = colors.HexColor('#E3F2FD')      # light blue background
COLOR_WHITE = colors.white
COLOR_BLACK = colors.black
COLOR_GRAY = colors.HexColor('#757575')
COLOR_LIGHT_GRAY = colors.HexColor('#F5F5F5')

# Score colors
COLOR_SCORE_GREEN = colors.HexColor('#2E7D32')
COLOR_SCORE_ORANGE = colors.HexColor('#F57F17')
COLOR_SCORE_RED = colors.HexColor('#C62828')

# Thermodynamic constants for Kd estimation
R_KCAL = 1.987e-3    # kcal/(mol*K)
T_STANDARD = 298.15   # K (25 C)

# Header dimensions
HEADER_HEIGHT = 50

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _get_styles():
    """Return a dictionary of ParagraphStyles for the report."""
    styles = getSampleStyleSheet()

    custom = {
        'title': ParagraphStyle(
            'AcuTitle', parent=styles['Title'],
            fontSize=18, textColor=COLOR_PRIMARY, spaceAfter=6,
        ),
        'subtitle': ParagraphStyle(
            'AcuSubtitle', parent=styles['Heading2'],
            fontSize=13, textColor=COLOR_ACCENT, spaceAfter=4, spaceBefore=10,
        ),
        'heading': ParagraphStyle(
            'AcuHeading', parent=styles['Heading3'],
            fontSize=11, textColor=COLOR_PRIMARY, spaceAfter=4, spaceBefore=8,
        ),
        'body': ParagraphStyle(
            'AcuBody', parent=styles['Normal'],
            fontSize=9, leading=12,
        ),
        'small': ParagraphStyle(
            'AcuSmall', parent=styles['Normal'],
            fontSize=7, leading=9, textColor=COLOR_GRAY,
        ),
        'cell': ParagraphStyle(
            'AcuCell', parent=styles['Normal'],
            fontSize=8, leading=10,
        ),
        'cell_center': ParagraphStyle(
            'AcuCellCenter', parent=styles['Normal'],
            fontSize=8, leading=10, alignment=TA_CENTER,
        ),
        'cell_bold': ParagraphStyle(
            'AcuCellBold', parent=styles['Normal'],
            fontSize=8, leading=10, fontName='Helvetica-Bold',
        ),
    }
    return custom


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_nan(value):
    """Check if a value is NaN, handling non-float types gracefully.

    Args:
        value: Any value to check.

    Returns:
        True if value is NaN, False otherwise.
    """
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _score_color(score):
    """Return a reportlab color based on docking score value.

    Green for strong binders (< -7), orange for moderate (-5 to -7),
    red for weak (> -5).

    Args:
        score: Vina docking score in kcal/mol.

    Returns:
        reportlab color object.
    """
    if score is None:
        return COLOR_GRAY
    try:
        score = float(score)
    except (TypeError, ValueError):
        return COLOR_GRAY
    if score < -7.0:
        return COLOR_SCORE_GREEN
    elif score <= -5.0:
        return COLOR_SCORE_ORANGE
    else:
        return COLOR_SCORE_RED


def _score_interpretation(score):
    """Return a text interpretation of a Vina docking score.

    Score thresholds:
        < -10  : Excellent
        -8 to -10 : Strong
        -6 to -8  : Moderate
        -4 to -6  : Weak
        > -4   : Very Weak

    Args:
        score: Vina docking score in kcal/mol.

    Returns:
        String interpretation.
    """
    if score is None:
        return 'N/A'
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 'N/A'
    if score < -10.0:
        return 'Excellent'
    elif score < -8.0:
        return 'Strong'
    elif score < -6.0:
        return 'Moderate'
    elif score < -4.0:
        return 'Weak'
    else:
        return 'Very Weak'


def _estimate_kd(score):
    """Estimate dissociation constant (Kd) from Vina score.

    Uses the thermodynamic relationship:
        Kd = exp(dG / (R * T))
    where dG is the Vina score in kcal/mol, R = 1.987e-3 kcal/(mol*K),
    and T = 298.15 K.

    Args:
        score: Vina docking score in kcal/mol.

    Returns:
        Formatted string (e.g. '50 nM', '7.2 uM', '1.3 mM').
    """
    if score is None:
        return 'N/A'
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 'N/A'

    kd_M = math.exp(score / (R_KCAL * T_STANDARD))

    if kd_M < 1e-9:
        return f'{kd_M * 1e12:.1f} pM'
    elif kd_M < 1e-6:
        return f'{kd_M * 1e9:.1f} nM'
    elif kd_M < 1e-3:
        return f'{kd_M * 1e6:.1f} uM'
    else:
        return f'{kd_M * 1e3:.1f} mM'


def _render_2d_structure(smiles, output_path, size=(300, 200)):
    """Render a 2D molecular structure image from SMILES using RDKit.

    Args:
        smiles: SMILES string of the molecule.
        output_path: File path to save the PNG image.
        size: Tuple (width, height) in pixels.

    Returns:
        output_path on success, None on failure.
    """
    if not HAS_RDKIT:
        return None
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        img.save(output_path)
        return output_path
    except Exception:
        return None


def _build_header(canvas, doc):
    """Draw branded AcuDock header on each page.

    Renders a dark blue banner across the top with the AcuDock branding,
    a teal accent line, date stamp, and a page-numbered footer.

    Called via onFirstPage / onLaterPages callbacks of SimpleDocTemplate.

    Args:
        canvas: reportlab canvas object.
        doc: SimpleDocTemplate object.
    """
    canvas.saveState()

    # Blue header bar
    canvas.setFillColor(COLOR_PRIMARY)
    canvas.rect(0, PAGE_HEIGHT - HEADER_HEIGHT, PAGE_WIDTH, HEADER_HEIGHT,
                fill=1, stroke=0)

    # Teal accent line
    canvas.setFillColor(COLOR_ACCENT)
    canvas.rect(0, PAGE_HEIGHT - HEADER_HEIGHT - 3, PAGE_WIDTH, 3,
                fill=1, stroke=0)

    # Title text
    canvas.setFillColor(COLOR_WHITE)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(30, PAGE_HEIGHT - 33, 'AcuDock')

    # Subtitle
    canvas.setFont('Helvetica', 9)
    canvas.drawString(115, PAGE_HEIGHT - 31, 'Molecular Docking Report')

    # Date on the right
    canvas.setFont('Helvetica', 8)
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    canvas.drawRightString(PAGE_WIDTH - 30, PAGE_HEIGHT - 31, date_str)

    # Footer
    canvas.setFillColor(COLOR_GRAY)
    canvas.setFont('Helvetica', 7)
    canvas.drawString(30, 20, f'Generated by AcuDock | Page {doc.page}')
    canvas.drawRightString(PAGE_WIDTH - 30, 20, 'github.com/AcuDock')

    canvas.restoreState()


def _make_summary_box(text_lines, styles):
    """Create a colored summary box with key-value lines.

    Args:
        text_lines: List of (label, value) tuples.
        styles: Style dict from _get_styles().

    Returns:
        reportlab Table acting as a styled box.
    """
    data = []
    for label, value in text_lines:
        data.append([
            Paragraph(f'<b>{label}:</b>', styles['cell_bold']),
            Paragraph(str(value), styles['cell']),
        ])

    table = Table(data, colWidths=[130, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return table


def _make_properties_table(properties, styles):
    """Create a formatted properties Table from a properties dict.

    Args:
        properties: Dict with keys like MW, LogP, HBD, HBA, RotBonds,
                    TPSA, QED, SA_Score, PAINS_Count, LE, LLE.
        styles: Style dict from _get_styles().

    Returns:
        reportlab Table object, or None if no properties.
    """
    if not properties:
        return None

    # Property display order and labels
    prop_order = [
        ('MW', 'Mol. Weight'),
        ('LogP', 'LogP'),
        ('HBD', 'H-Bond Donors'),
        ('HBA', 'H-Bond Acceptors'),
        ('RotBonds', 'Rotatable Bonds'),
        ('TPSA', 'TPSA'),
        ('QED', 'QED Score'),
        ('SA_Score', 'SA Score'),
        ('PAINS_Count', 'PAINS Alerts'),
        ('LE', 'Ligand Efficiency'),
        ('LLE', 'Lipophilic LE'),
    ]

    data = [[
        Paragraph('<b>Property</b>', styles['cell_bold']),
        Paragraph('<b>Value</b>', styles['cell_center']),
    ]]

    for key, label in prop_order:
        val = properties.get(key)
        if val is None:
            continue
        if isinstance(val, float) and not _is_nan(val):
            val_str = f'{val:.2f}'
        elif _is_nan(val):
            continue
        else:
            val_str = str(val)
        data.append([
            Paragraph(label, styles['cell']),
            Paragraph(val_str, styles['cell_center']),
        ])

    if len(data) <= 1:
        return None

    table = Table(data, colWidths=[140, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_WHITE),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT_GRAY]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return table


def _find_column(df, candidates):
    """Find the first matching column name from a list of candidates.

    Args:
        df: pandas DataFrame.
        candidates: List of possible column names.

    Returns:
        Matching column name, or None if not found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_score_table(results_df, styles):
    """Create a formatted score Table from a results DataFrame.

    Expects columns: a score column (Score, Affinity, or Best_Score),
    and optionally RMSD_lb and RMSD_ub.

    Args:
        results_df: pandas DataFrame with docking results.
        styles: Style dict from _get_styles().

    Returns:
        reportlab Table object, or None if empty.
    """
    if not HAS_PANDAS or results_df is None or results_df.empty:
        return None

    score_col = _find_column(results_df,
                             ['Vina Score', 'Score', 'Affinity', 'Best_Score',
                              'Vina_Score', 'score', 'affinity', 'vina_score'])
    rmsd_lb_col = _find_column(results_df, ['RMSD_lb', 'rmsd_lb', 'RMSD_l.b.', 'RMSD l.b.'])
    rmsd_ub_col = _find_column(results_df, ['RMSD_ub', 'rmsd_ub', 'RMSD_u.b.', 'RMSD u.b.'])

    headers = ['Pose', 'Score (kcal/mol)', 'RMSD l.b.', 'RMSD u.b.',
               'Est. Kd', 'Quality']
    data = [[Paragraph(f'<b>{h}</b>', styles['cell_bold']) for h in headers]]

    for idx, (i, row) in enumerate(results_df.iterrows()):
        pose_num = idx + 1
        try:
            score_val = row[score_col] if score_col else None
        except (KeyError, TypeError):
            score_val = None
        try:
            rmsd_lb = row[rmsd_lb_col] if rmsd_lb_col else None
        except (KeyError, TypeError):
            rmsd_lb = None
        try:
            rmsd_ub = row[rmsd_ub_col] if rmsd_ub_col else None
        except (KeyError, TypeError):
            rmsd_ub = None

        score_str = f'{float(score_val):.2f}' if score_val is not None and not _is_nan(score_val) else 'N/A'
        rmsd_lb_str = f'{float(rmsd_lb):.2f}' if rmsd_lb is not None and not _is_nan(rmsd_lb) else '-'
        rmsd_ub_str = f'{float(rmsd_ub):.2f}' if rmsd_ub is not None and not _is_nan(rmsd_ub) else '-'
        kd_str = _estimate_kd(score_val)
        interp = _score_interpretation(score_val)

        data.append([
            Paragraph(str(pose_num), styles['cell_center']),
            Paragraph(score_str, styles['cell_center']),
            Paragraph(rmsd_lb_str, styles['cell_center']),
            Paragraph(rmsd_ub_str, styles['cell_center']),
            Paragraph(kd_str, styles['cell_center']),
            Paragraph(interp, styles['cell_center']),
        ])

    col_widths = [40, 90, 65, 65, 70, 65]
    table = Table(data, colWidths=col_widths)

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_WHITE),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT_GRAY]),
    ]

    # Color-code score cells
    if score_col:
        for row_idx in range(1, len(data)):
            score_val = results_df.iloc[row_idx - 1].get(score_col)
            if score_val is not None and not _is_nan(score_val):
                sc = _score_color(score_val)
                style_cmds.append(
                    ('TEXTCOLOR', (1, row_idx), (1, row_idx), sc))

    table.setStyle(TableStyle(style_cmds))
    return table


def _add_3d_views_grid(story, image_paths, styles):
    """Add 3D binding pose images to the story.

    Displays images stacked vertically at a larger size for clarity.
    Supports both dict (label -> path) and list inputs.

    Args:
        story: List of reportlab flowables to append to.
        image_paths: Dict or list of image paths.
        styles: Style dict from _get_styles().
    """
    if not image_paths:
        return

    story.append(Paragraph('3D Binding Pose Views', styles['subtitle']))

    # Normalize to list of (label, path) tuples
    if isinstance(image_paths, dict):
        label_map = {
            'overview': 'Protein Overview with Docked Ligand',
            'binding_site': 'Binding Site Closeup',
        }
        items = [(label_map.get(k, k.replace('_', ' ').title()), v)
                 for k, v in image_paths.items()]
    elif isinstance(image_paths, (list, tuple)):
        items = [(f'View {i+1}', p) for i, p in enumerate(image_paths)]
    else:
        return

    valid_items = [(label, path) for label, path in items
                   if path and os.path.isfile(str(path))]
    if not valid_items:
        story.append(Paragraph('(3D view images not available)', styles['small']))
        return

    for label, path in valid_items:
        try:
            img = Image(str(path), width=420, height=310)
            story.append(img)
        except Exception:
            story.append(Paragraph('(image error)', styles['small']))
        story.append(Spacer(1, 4))


def _make_interactions_table(interactions_df, styles):
    """Create a color-coded interaction fingerprint table.

    Args:
        interactions_df: DataFrame with interaction data. Common columns:
                         Residue, Type, Distance, Strength.
        styles: Style dict from _get_styles().

    Returns:
        reportlab Table object.
    """
    cols = list(interactions_df.columns)
    data = [[Paragraph(f'<b>{c}</b>', styles['cell_bold']) for c in cols]]

    # Interaction type colors for visual distinction
    type_colors = {
        'hydrogen_bond': colors.HexColor('#1B5E20'),
        'h_bond': colors.HexColor('#1B5E20'),
        'hydrophobic': colors.HexColor('#E65100'),
        'pi_stacking': colors.HexColor('#4A148C'),
        'pi_cation': colors.HexColor('#880E4F'),
        'salt_bridge': colors.HexColor('#B71C1C'),
        'ionic': colors.HexColor('#B71C1C'),
        'water_bridge': colors.HexColor('#01579B'),
        'halogen_bond': colors.HexColor('#F57F17'),
    }

    type_col_idx = None
    for i, c in enumerate(cols):
        if c.lower() in ('type', 'interaction_type', 'interaction'):
            type_col_idx = i
            break

    for _, row in interactions_df.iterrows():
        row_data = []
        for c in cols:
            val = row.get(c)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val_str = '-'
            elif isinstance(val, float):
                val_str = f'{val:.2f}'
            else:
                val_str = str(val)
            row_data.append(Paragraph(val_str, styles['cell_center']))
        data.append(row_data)

    n_cols = len(cols)
    col_width = min(90, int(500 / max(n_cols, 1)))
    table = Table(data, colWidths=[col_width] * n_cols)

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_WHITE),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT_GRAY]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]

    # Color-code interaction type cells
    if type_col_idx is not None:
        for row_idx in range(1, len(data)):
            itype = str(interactions_df.iloc[row_idx - 1].get(
                cols[type_col_idx], ''
            )).lower().replace(' ', '_')
            tc = type_colors.get(itype)
            if tc:
                style_cmds.append((
                    'TEXTCOLOR',
                    (type_col_idx, row_idx), (type_col_idx, row_idx),
                    tc,
                ))

    table.setStyle(TableStyle(style_cmds))
    return table


def _make_ranked_table(df, score_col, styles, max_rows=10, extra_cols=None):
    """Create a ranked hits table with score color-coding.

    Args:
        df: DataFrame sorted by score (best first).
        score_col: Name of the score column.
        styles: Style dict from _get_styles().
        max_rows: Maximum number of rows to show.
        extra_cols: Optional list of extra column names to include.

    Returns:
        reportlab Table object.
    """
    if extra_cols is None:
        extra_cols = []
    # Determine which extra columns are actually present
    avail_extras = [c for c in extra_cols if c in df.columns]

    headers = ['Rank', 'Name', 'Score', 'Est. Kd', 'Quality'] + avail_extras
    data = [[Paragraph(f'<b>{h}</b>', styles['cell_bold']) for h in headers]]

    top = df.head(max_rows)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        sc = row[score_col]
        row_data = [
            Paragraph(str(rank), styles['cell_center']),
            Paragraph(str(row.get('Name', ''))[:25], styles['cell']),
            Paragraph(f'{float(sc):.2f}', styles['cell_center']),
            Paragraph(_estimate_kd(sc), styles['cell_center']),
            Paragraph(_score_interpretation(sc), styles['cell_center']),
        ]
        for c in avail_extras:
            val = row.get(c)
            if val is not None and not _is_nan(val):
                val_str = f'{float(val):.1f}'
            else:
                val_str = '-'
            row_data.append(Paragraph(val_str, styles['cell_center']))
        data.append(row_data)

    n_extra = len(avail_extras)
    col_widths = [35, 120, 60, 65, 65] + [55] * n_extra
    table = Table(data, colWidths=col_widths)

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_WHITE),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT_GRAY]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    for row_idx in range(1, len(data)):
        sc = top.iloc[row_idx - 1][score_col]
        style_cmds.append(
            ('TEXTCOLOR', (2, row_idx), (2, row_idx), _score_color(sc)))
    table.setStyle(TableStyle(style_cmds))
    return table


def _basic_properties(smiles):
    """Compute basic molecular properties from SMILES (fallback).

    Used when acudock_utils.get_ligand_properties is not available.

    Args:
        smiles: SMILES string.

    Returns:
        Dict of basic properties.
    """
    if not HAS_RDKIT or not smiles:
        return {}
    try:
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
        }
    except Exception:
        return {}


def _get_tmp_dir(output_path):
    """Return a temporary directory based on the output path's directory.

    Args:
        output_path: Path to the output PDF file.

    Returns:
        Directory path string.
    """
    d = os.path.dirname(output_path)
    return d if d else '/tmp'


def _build_doc(output_path):
    """Create a SimpleDocTemplate with standard AcuDock margins.

    Args:
        output_path: Path for the output PDF file.

    Returns:
        SimpleDocTemplate instance.
    """
    return SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=HEADER_HEIGHT + 20,
        bottomMargin=40,
        leftMargin=40,
        rightMargin=40,
    )


# ---------------------------------------------------------------------------
# Public API: Single Dock PDF
# ---------------------------------------------------------------------------


def generate_single_dock_pdf(output_path, protein_id, ligand_name, smiles,
                             results_df, properties, image_paths=None,
                             interactions_df=None):
    """Generate a PDF report for a single docking run.

    Page 1 contains the summary box (protein, ligand, best score,
    interpretation), 2D structure image, molecular properties table,
    and pose score table. Page 2 (if data available) shows 3D binding
    pose views in a 2x3 grid and the interaction fingerprint table.

    Args:
        output_path: File path for the output PDF.
        protein_id: PDB ID of the target protein (e.g. '1HSG').
        ligand_name: Name or identifier of the ligand.
        smiles: SMILES string of the docked ligand.
        results_df: DataFrame with pose results. Expected columns include
                    a score column (Score, Affinity, or Best_Score) and
                    optionally RMSD_lb, RMSD_ub.
        properties: Dict of molecular properties (MW, LogP, HBD, HBA,
                    RotBonds, TPSA, QED, SA_Score, PAINS_Count, LE, LLE).
        image_paths: Optional dict or list of 3D view image paths for
                     6-axis views (front, back, top, bottom, left, right).
        interactions_df: Optional DataFrame of interaction fingerprints
                         with columns like Residue, Type, Distance.

    Returns:
        output_path on success.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        raise ImportError('reportlab is required for PDF generation. '
                          'Install with: pip install reportlab')

    styles = _get_styles()
    story = []
    tmp_dir = _get_tmp_dir(output_path)

    # Top spacer for header
    story.append(Spacer(1, 30))
    story.append(Paragraph('Single Docking Report', styles['title']))

    # Determine best score
    best_score = None
    if HAS_PANDAS and results_df is not None and not results_df.empty:
        score_col = _find_column(results_df,
                                 ['Vina Score', 'Score', 'Affinity', 'Best_Score',
                                  'Vina_Score', 'score', 'affinity', 'vina_score'])
        if score_col:
            best_score = results_df[score_col].min()

    # Summary box
    summary_lines = [
        ('Protein', str(protein_id)),
        ('Ligand', str(ligand_name)),
    ]
    if smiles:
        display_smi = smiles if len(smiles) <= 60 else smiles[:57] + '...'
        summary_lines.append(('SMILES', display_smi))
    if best_score is not None:
        summary_lines.append(('Best Score', f'{best_score:.2f} kcal/mol'))
        summary_lines.append(('Interpretation', _score_interpretation(best_score)))
        summary_lines.append(('Est. Kd', _estimate_kd(best_score)))
    story.append(_make_summary_box(summary_lines, styles))
    story.append(Spacer(1, 10))

    # 2D structure image (~300x200)
    if smiles:
        struct_path = os.path.join(tmp_dir, '_acudock_2d_struct.png')
        rendered = _render_2d_structure(smiles, struct_path, size=(300, 200))
        if rendered and os.path.isfile(rendered):
            story.append(Paragraph('2D Structure', styles['subtitle']))
            try:
                story.append(Image(rendered, width=250, height=167))
            except Exception:
                pass
            story.append(Spacer(1, 8))

    # Molecular properties table
    props_table = _make_properties_table(properties, styles)
    if props_table is not None:
        story.append(Paragraph('Molecular Properties', styles['subtitle']))
        story.append(props_table)
        story.append(Spacer(1, 10))

    # Score table for all poses — start on new page
    score_table = _make_score_table(results_df, styles)
    if score_table is not None:
        story.append(PageBreak())
        story.append(Spacer(1, 30))
        story.append(Paragraph('Docking Poses', styles['subtitle']))
        story.append(score_table)

    # Page 3: 3D views and interactions (if available)
    has_3d = False
    if image_paths:
        if isinstance(image_paths, dict):
            has_3d = any(v and os.path.isfile(str(v))
                         for v in image_paths.values())
        elif isinstance(image_paths, (list, tuple)):
            has_3d = any(p and os.path.isfile(str(p)) for p in image_paths)

    has_interactions = (HAS_PANDAS and interactions_df is not None
                        and not interactions_df.empty)

    if has_3d or has_interactions:
        story.append(PageBreak())
        story.append(Spacer(1, 30))

        if has_3d:
            _add_3d_views_grid(story, image_paths, styles)
            story.append(Spacer(1, 12))

        if has_interactions:
            story.append(Paragraph('Interaction Fingerprint', styles['subtitle']))
            story.append(_make_interactions_table(interactions_df, styles))

    # Build PDF
    doc = _build_doc(output_path)
    doc.build(story, onFirstPage=_build_header, onLaterPages=_build_header)
    return output_path


# ---------------------------------------------------------------------------
# Public API: Batch Screening PDF
# ---------------------------------------------------------------------------


def generate_batch_pdf(output_path, protein_id, batch_df, chart_path=None,
                       per_ligand_data=None):
    """Generate a PDF report for batch screening results.

    Page 1 contains the campaign summary, ranked bar chart, and top 10
    summary table. Subsequent pages show per-ligand details including
    name, SMILES, 2D structure, properties, and score.

    Args:
        output_path: File path for the output PDF.
        protein_id: PDB ID of the target protein.
        batch_df: DataFrame of all batch results. Expected columns:
                  Name, SMILES, Best_Score (or Score), and optionally
                  MW, LogP, QED, etc.
        chart_path: Optional path to a bar chart image of ranked scores.
        per_ligand_data: Optional list of dicts, each with keys:
                         name, smiles, properties, score, image_path.
                         Used for per-ligand detail pages.

    Returns:
        output_path on success.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        raise ImportError('reportlab is required. Install with: pip install reportlab')

    styles = _get_styles()
    story = []
    tmp_dir = _get_tmp_dir(output_path)

    story.append(Spacer(1, 30))
    story.append(Paragraph('Batch Screening Report', styles['title']))

    # Campaign summary
    n_compounds = len(batch_df) if HAS_PANDAS and batch_df is not None else 0
    best_score = None
    best_name = 'N/A'
    score_col = None

    if HAS_PANDAS and batch_df is not None and not batch_df.empty:
        score_col = _find_column(batch_df,
                                 ['Best_Score', 'Score', 'Affinity'])
        if score_col:
            valid = batch_df.dropna(subset=[score_col])
            if not valid.empty:
                best_idx = valid[score_col].idxmin()
                best_score = valid.loc[best_idx, score_col]
                if 'Name' in valid.columns:
                    best_name = str(valid.loc[best_idx, 'Name'])

    summary_lines = [
        ('Protein Target', str(protein_id)),
        ('Compounds Screened', str(n_compounds)),
    ]
    if best_score is not None:
        summary_lines.append(('Best Score', f'{best_score:.2f} kcal/mol'))
        summary_lines.append(('Best Compound', best_name))
        summary_lines.append(('Interpretation', _score_interpretation(best_score)))
    story.append(_make_summary_box(summary_lines, styles))
    story.append(Spacer(1, 10))

    # Bar chart
    if chart_path and os.path.isfile(str(chart_path)):
        story.append(Paragraph('Score Distribution', styles['subtitle']))
        try:
            story.append(Image(str(chart_path), width=480, height=280))
        except Exception:
            pass
        story.append(Spacer(1, 10))

    # Top 10 summary table
    if HAS_PANDAS and batch_df is not None and not batch_df.empty and score_col:
        story.append(Paragraph('Top 10 Compounds', styles['subtitle']))
        sorted_df = batch_df.dropna(subset=[score_col]).sort_values(
            score_col, ascending=True
        )
        table = _make_ranked_table(sorted_df, score_col, styles,
                                   max_rows=10,
                                   extra_cols=['MW', 'LogP', 'QED'])
        story.append(table)

    # Per-ligand detail pages
    if per_ligand_data:
        for lig_data in per_ligand_data:
            story.append(PageBreak())
            story.append(Spacer(1, 30))

            lig_name = lig_data.get('name', 'Unknown')
            lig_smiles = lig_data.get('smiles', '')
            lig_score = lig_data.get('score')
            lig_props = lig_data.get('properties', {})

            story.append(Paragraph(f'Ligand Detail: {lig_name}', styles['subtitle']))

            detail_lines = [('Name', lig_name)]
            if lig_smiles:
                display = lig_smiles if len(lig_smiles) <= 60 else lig_smiles[:57] + '...'
                detail_lines.append(('SMILES', display))
            if lig_score is not None:
                detail_lines.append(('Score', f'{float(lig_score):.2f} kcal/mol'))
                detail_lines.append(('Est. Kd', _estimate_kd(lig_score)))
            story.append(_make_summary_box(detail_lines, styles))
            story.append(Spacer(1, 8))

            # 2D structure
            if lig_smiles:
                safe_name = str(lig_name).replace('/', '_').replace(' ', '_')[:20]
                struct_path = os.path.join(tmp_dir, f'_acudock_2d_{safe_name}.png')
                rendered = _render_2d_structure(lig_smiles, struct_path)
                if rendered and os.path.isfile(rendered):
                    try:
                        story.append(Image(rendered, width=220, height=147))
                    except Exception:
                        pass
                    story.append(Spacer(1, 6))

            # Properties
            props_table = _make_properties_table(lig_props, styles)
            if props_table is not None:
                story.append(props_table)

    # Build PDF
    doc = _build_doc(output_path)
    doc.build(story, onFirstPage=_build_header, onLaterPages=_build_header)
    return output_path


# ---------------------------------------------------------------------------
# Public API: Multi-Protein Docking PDF
# ---------------------------------------------------------------------------


def generate_multi_protein_pdf(output_path, ligand_name, smiles, results_df,
                               chart_path=None, per_protein_data=None):
    """Generate a PDF report for multi-protein docking of a single ligand.

    Page 1 contains the ligand info (name, SMILES, 2D structure, properties),
    comparison bar chart, and summary table across all proteins. Subsequent
    pages show per-protein details including PDB ID, score, and 3D views.

    Args:
        output_path: File path for the output PDF.
        ligand_name: Name or identifier of the ligand.
        smiles: SMILES string of the ligand.
        results_df: DataFrame with columns like Protein/PDB_ID and
                    Score/Best_Score.
        chart_path: Optional path to a comparison bar chart image.
        per_protein_data: Optional list of dicts, each with keys:
                          protein_id, score, image_paths (dict of 3D views).

    Returns:
        output_path on success.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        raise ImportError('reportlab is required. Install with: pip install reportlab')

    styles = _get_styles()
    story = []
    tmp_dir = _get_tmp_dir(output_path)

    story.append(Spacer(1, 30))
    story.append(Paragraph('Multi-Protein Docking Report', styles['title']))

    # Ligand summary
    summary_lines = [('Ligand', str(ligand_name))]
    if smiles:
        display = smiles if len(smiles) <= 60 else smiles[:57] + '...'
        summary_lines.append(('SMILES', display))

    score_col = None
    prot_col = None
    if HAS_PANDAS and results_df is not None and not results_df.empty:
        n_proteins = len(results_df)
        summary_lines.append(('Proteins Tested', str(n_proteins)))

        score_col = _find_column(results_df,
                                 ['Score', 'Best_Score', 'Affinity'])
        prot_col = _find_column(results_df,
                                ['Protein', 'PDB_ID', 'protein_id', 'Target'])

        if score_col:
            valid = results_df.dropna(subset=[score_col])
            if not valid.empty:
                best_idx = valid[score_col].idxmin()
                best_score = valid.loc[best_idx, score_col]
                best_prot = str(valid.loc[best_idx, prot_col]) if prot_col else 'N/A'
                summary_lines.append(('Best Score', f'{best_score:.2f} kcal/mol'))
                summary_lines.append(('Best Target', best_prot))

    story.append(_make_summary_box(summary_lines, styles))
    story.append(Spacer(1, 10))

    # 2D structure
    if smiles:
        struct_path = os.path.join(tmp_dir, '_acudock_2d_multi.png')
        rendered = _render_2d_structure(smiles, struct_path, size=(300, 200))
        if rendered and os.path.isfile(rendered):
            story.append(Paragraph('2D Structure', styles['subtitle']))
            try:
                story.append(Image(rendered, width=250, height=167))
            except Exception:
                pass
            story.append(Spacer(1, 8))

    # Molecular properties
    if smiles:
        try:
            from acudock_utils import get_ligand_properties
            props = get_ligand_properties(smiles)
        except (ImportError, Exception):
            props = _basic_properties(smiles)
        props_table = _make_properties_table(props, styles)
        if props_table is not None:
            story.append(Paragraph('Molecular Properties', styles['subtitle']))
            story.append(props_table)
            story.append(Spacer(1, 10))

    # Comparison chart
    if chart_path and os.path.isfile(str(chart_path)):
        story.append(Paragraph('Cross-Target Comparison', styles['subtitle']))
        try:
            story.append(Image(str(chart_path), width=480, height=280))
        except Exception:
            pass
        story.append(Spacer(1, 10))

    # Summary table across proteins
    if (HAS_PANDAS and results_df is not None and not results_df.empty
            and prot_col and score_col):
        story.append(Paragraph('Results Summary', styles['subtitle']))

        headers = ['Protein', 'Score (kcal/mol)', 'Est. Kd', 'Quality']
        data = [[Paragraph(f'<b>{h}</b>', styles['cell_bold']) for h in headers]]
        sorted_df = results_df.dropna(subset=[score_col]).sort_values(
            score_col, ascending=True
        )
        for _, row in sorted_df.iterrows():
            sc = row[score_col]
            data.append([
                Paragraph(str(row[prot_col]), styles['cell']),
                Paragraph(f'{float(sc):.2f}', styles['cell_center']),
                Paragraph(_estimate_kd(sc), styles['cell_center']),
                Paragraph(_score_interpretation(sc), styles['cell_center']),
            ])

        table = Table(data, colWidths=[100, 110, 85, 85])
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_WHITE),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT_GRAY]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]
        for row_idx in range(1, len(data)):
            sc = sorted_df.iloc[row_idx - 1][score_col]
            style_cmds.append(
                ('TEXTCOLOR', (1, row_idx), (1, row_idx), _score_color(sc)))
        table.setStyle(TableStyle(style_cmds))
        story.append(table)

    # Per-protein detail pages
    if per_protein_data:
        for prot_data in per_protein_data:
            story.append(PageBreak())
            story.append(Spacer(1, 30))

            prot_id = prot_data.get('protein_id', 'Unknown')
            prot_score = prot_data.get('score')
            prot_images = prot_data.get('image_paths')

            story.append(Paragraph(f'Target: {prot_id}', styles['subtitle']))
            detail_lines = [('PDB ID', prot_id)]
            if prot_score is not None:
                detail_lines.append(('Score', f'{float(prot_score):.2f} kcal/mol'))
                detail_lines.append(('Interpretation',
                                     _score_interpretation(prot_score)))
                detail_lines.append(('Est. Kd', _estimate_kd(prot_score)))
            story.append(_make_summary_box(detail_lines, styles))
            story.append(Spacer(1, 10))

            if prot_images:
                _add_3d_views_grid(story, prot_images, styles)

    # Build PDF
    doc = _build_doc(output_path)
    doc.build(story, onFirstPage=_build_header, onLaterPages=_build_header)
    return output_path


# ---------------------------------------------------------------------------
# Public API: Scout Active Learning PDF
# ---------------------------------------------------------------------------


def generate_scout_pdf(output_path, campaign_config, top_hits_df,
                       plots_dict=None, best_hit_data=None):
    """Generate a PDF report for a Scout active learning campaign.

    Page 1 shows campaign parameters and convergence/distribution plots.
    Page 2 has the top 20 hits table and 2D structures of the top 5.
    Page 3 (if best_hit_data provided) shows the best hit detail with
    3D views and interaction fingerprint.

    Args:
        output_path: File path for the output PDF.
        campaign_config: Dict with campaign parameters, e.g.:
                         protein_id, library_size, n_cycles, batch_size,
                         exhaustiveness, beta, total_docked, total_time_s.
        top_hits_df: DataFrame of top hits. Expected columns:
                     Name, SMILES, Best_Score, and optionally MW, LogP, etc.
        plots_dict: Optional dict of plot image paths, e.g.:
                    convergence, distribution, enrichment, uncertainty.
        best_hit_data: Optional dict for the best hit detail page:
                       name, smiles, score, properties, image_paths,
                       interactions_df.

    Returns:
        output_path on success.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        raise ImportError('reportlab is required. Install with: pip install reportlab')

    styles = _get_styles()
    story = []
    tmp_dir = _get_tmp_dir(output_path)
    config = campaign_config or {}

    # --- Page 1: Campaign overview ---
    story.append(Spacer(1, 30))
    story.append(Paragraph('Scout Active Learning Report', styles['title']))

    # Campaign parameters
    summary_lines = []
    if config.get('protein_id'):
        summary_lines.append(('Protein Target', str(config['protein_id'])))
    if config.get('library_size'):
        summary_lines.append(('Library Size', f"{config['library_size']:,}"))
    if config.get('total_docked'):
        summary_lines.append(('Compounds Docked',
                              f"{config['total_docked']:,}"))
    if config.get('n_cycles'):
        summary_lines.append(('AL Cycles', str(config['n_cycles'])))
    if config.get('batch_size'):
        summary_lines.append(('Batch Size', str(config['batch_size'])))
    if config.get('exhaustiveness'):
        summary_lines.append(('Exhaustiveness', str(config['exhaustiveness'])))
    if config.get('beta') is not None:
        summary_lines.append(('UCB Beta', str(config['beta'])))
    if config.get('total_time_s'):
        mins = config['total_time_s'] / 60.0
        summary_lines.append(('Total Time', f'{mins:.1f} min'))

    # Best score from hits
    best_score = None
    score_col = None
    if HAS_PANDAS and top_hits_df is not None and not top_hits_df.empty:
        score_col = _find_column(top_hits_df,
                                 ['Best_Score', 'Score', 'Affinity'])
        if score_col:
            valid = top_hits_df.dropna(subset=[score_col])
            if not valid.empty:
                best_score = valid[score_col].min()
                summary_lines.append(('Best Score',
                                      f'{best_score:.2f} kcal/mol'))
                summary_lines.append(('Interpretation',
                                      _score_interpretation(best_score)))

    if summary_lines:
        story.append(_make_summary_box(summary_lines, styles))
        story.append(Spacer(1, 10))

    # Convergence and distribution plots
    if plots_dict:
        for plot_key, plot_label in [
            ('convergence', 'Convergence Plot'),
            ('distribution', 'Score Distribution'),
            ('enrichment', 'Enrichment Curve'),
            ('uncertainty', 'Uncertainty vs Cycle'),
        ]:
            plot_path = plots_dict.get(plot_key)
            if plot_path and os.path.isfile(str(plot_path)):
                story.append(Paragraph(plot_label, styles['heading']))
                try:
                    story.append(Image(str(plot_path), width=440, height=260))
                except Exception:
                    pass
                story.append(Spacer(1, 6))

    # --- Page 2: Top hits table + 2D structures ---
    story.append(PageBreak())
    story.append(Spacer(1, 30))
    story.append(Paragraph('Top Hits', styles['title']))

    if (HAS_PANDAS and top_hits_df is not None
            and not top_hits_df.empty and score_col):
        sorted_df = top_hits_df.dropna(subset=[score_col]).sort_values(
            score_col, ascending=True
        )
        table = _make_ranked_table(sorted_df, score_col, styles,
                                   max_rows=20,
                                   extra_cols=['MW', 'LogP', 'QED'])
        story.append(table)
        story.append(Spacer(1, 12))

        # 2D structures of top 5
        top5 = sorted_df.head(5)
        if 'SMILES' in top5.columns:
            story.append(Paragraph('Top 5 Structures', styles['subtitle']))
            struct_items = []
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                smi = row.get('SMILES')
                name = str(row.get('Name', f'Hit {rank}'))[:20]
                if not smi:
                    continue
                img_path = os.path.join(
                    tmp_dir, f'_acudock_scout_top{rank}.png')
                rendered = _render_2d_structure(smi, img_path, size=(200, 150))
                if not rendered or not os.path.isfile(rendered):
                    continue

                cell = []
                try:
                    cell.append(Image(rendered, width=130, height=97))
                except Exception:
                    cell.append(
                        Paragraph('(error)', styles['small']))
                sc = row[score_col]
                cell.append(Paragraph(
                    f'{name}<br/>{float(sc):.2f} kcal/mol',
                    styles['cell_center'],
                ))
                inner = Table([[c] for c in cell], colWidths=[140])
                inner.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                struct_items.append(inner)

            if struct_items:
                rows = []
                for i in range(0, len(struct_items), 3):
                    row = struct_items[i:i + 3]
                    while len(row) < 3:
                        row.append('')
                    rows.append(row)
                grid = Table(rows, colWidths=[150] * 3)
                grid.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(grid)

    # --- Page 3: Best hit detail ---
    if best_hit_data:
        story.append(PageBreak())
        story.append(Spacer(1, 30))

        hit_name = best_hit_data.get('name', 'Best Hit')
        hit_smiles = best_hit_data.get('smiles', '')
        hit_score = best_hit_data.get('score')
        hit_props = best_hit_data.get('properties', {})
        hit_images = best_hit_data.get('image_paths')
        hit_interactions = best_hit_data.get('interactions_df')

        story.append(Paragraph(f'Best Hit: {hit_name}', styles['title']))

        detail_lines = [('Name', hit_name)]
        if hit_smiles:
            display = (hit_smiles if len(hit_smiles) <= 60
                       else hit_smiles[:57] + '...')
            detail_lines.append(('SMILES', display))
        if hit_score is not None:
            detail_lines.append(('Score',
                                 f'{float(hit_score):.2f} kcal/mol'))
            detail_lines.append(('Interpretation',
                                 _score_interpretation(hit_score)))
            detail_lines.append(('Est. Kd', _estimate_kd(hit_score)))
        story.append(_make_summary_box(detail_lines, styles))
        story.append(Spacer(1, 8))

        # 2D structure
        if hit_smiles:
            struct_path = os.path.join(tmp_dir, '_acudock_scout_best.png')
            rendered = _render_2d_structure(hit_smiles, struct_path)
            if rendered and os.path.isfile(rendered):
                try:
                    story.append(Image(rendered, width=220, height=147))
                except Exception:
                    pass
                story.append(Spacer(1, 6))

        # Properties
        props_table = _make_properties_table(hit_props, styles)
        if props_table is not None:
            story.append(Paragraph('Molecular Properties', styles['subtitle']))
            story.append(props_table)
            story.append(Spacer(1, 8))

        # 3D views
        if hit_images:
            _add_3d_views_grid(story, hit_images, styles)
            story.append(Spacer(1, 8))

        # Interactions
        if (HAS_PANDAS and hit_interactions is not None
                and not hit_interactions.empty):
            story.append(Paragraph('Interaction Fingerprint',
                                   styles['subtitle']))
            story.append(
                _make_interactions_table(hit_interactions, styles))

    # Build PDF
    doc = _build_doc(output_path)
    doc.build(story, onFirstPage=_build_header, onLaterPages=_build_header)
    return output_path
