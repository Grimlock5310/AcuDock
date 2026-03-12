"""
AcuDock Report Generator - PDF report generation for docking results.

Generates branded PDF reports for single docking, batch screening,
multi-protein docking, and active-learning scout campaigns using ReportLab.

Designed for use in Google Colab with AcuDock notebooks.
"""

import os
import math
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    Image, PageBreak, KeepTogether, HRFlowable,
)
from reportlab.lib.colors import HexColor

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

PRIMARY = HexColor("#1565C0")
ACCENT = HexColor("#00897B")
SCORE_GREEN = HexColor("#2ecc71")
SCORE_ORANGE = HexColor("#f39c12")
SCORE_RED = HexColor("#e74c3c")
TEXT_COLOR = HexColor("#212121")
LIGHT_GRAY = HexColor("#F5F5F5")
WHITE = colors.white
BLACK = colors.black

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_styles = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    "AcuTitle", parent=_styles["Heading1"],
    fontSize=20, textColor=PRIMARY, spaceAfter=4,
)
STYLE_H2 = ParagraphStyle(
    "AcuH2", parent=_styles["Heading2"],
    fontSize=14, textColor=PRIMARY, spaceBefore=12, spaceAfter=6,
)
STYLE_H3 = ParagraphStyle(
    "AcuH3", parent=_styles["Heading3"],
    fontSize=11, textColor=ACCENT, spaceBefore=8, spaceAfter=4,
)
STYLE_BODY = ParagraphStyle(
    "AcuBody", parent=_styles["Normal"],
    fontSize=9, textColor=TEXT_COLOR, leading=12,
)
STYLE_SMALL = ParagraphStyle(
    "AcuSmall", parent=_styles["Normal"],
    fontSize=7.5, textColor=TEXT_COLOR, leading=10,
)
STYLE_CENTER = ParagraphStyle(
    "AcuCenter", parent=STYLE_BODY, alignment=TA_CENTER,
)
STYLE_BOLD = ParagraphStyle(
    "AcuBold", parent=STYLE_BODY,
    fontName="Helvetica-Bold",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _score_color(score):
    """Return green/orange/red color based on docking score (kcal/mol)."""
    if score is None:
        return LIGHT_GRAY
    try:
        s = float(score)
    except (ValueError, TypeError):
        return LIGHT_GRAY
    if s <= -8.0:
        return SCORE_GREEN
    elif s <= -6.0:
        return SCORE_ORANGE
    else:
        return SCORE_RED


def _interpret_score(score):
    """Return human-readable interpretation of a Vina score."""
    try:
        s = float(score)
    except (ValueError, TypeError):
        return "N/A"
    if s <= -10:
        return "Excellent (sub-50 nM)"
    elif s <= -8:
        return "Strong (sub-uM)"
    elif s <= -6:
        return "Moderate (low-uM)"
    else:
        return "Weak (>100 uM)"


def _safe_image(path, width=None, height=None):
    """Return a ReportLab Image if the file exists, else a placeholder Paragraph."""
    if path and os.path.exists(path):
        kwargs = {}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        return Image(path, **kwargs)
    return Paragraph("<i>[Image not available]</i>", STYLE_SMALL)


def _make_header(canvas, doc, title, subtitle=""):
    """Draw branded AcuDock header with title, subtitle, and date."""
    canvas.saveState()
    # Blue header band
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, letter[1] - 60, letter[0], 60, fill=True, stroke=False)
    # Title text
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(40, letter[1] - 35, f"AcuDock  |  {title}")
    # Subtitle + date
    canvas.setFont("Helvetica", 9)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    right_text = f"{subtitle}    {date_str}" if subtitle else date_str
    canvas.drawRightString(letter[0] - 40, letter[1] - 35, right_text)
    # Accent bar
    canvas.setFillColor(ACCENT)
    canvas.rect(0, letter[1] - 63, letter[0], 3, fill=True, stroke=False)
    # Footer with page number
    canvas.setFillColor(TEXT_COLOR)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(letter[0] / 2, 20, f"Page {doc.page}")
    canvas.restoreState()


def _make_properties_table(props_dict):
    """Create a formatted two-column Table from a properties dict."""
    if not props_dict:
        return Paragraph("<i>No properties available.</i>", STYLE_BODY)

    display_order = [
        "MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "HeavyAtoms",
        "QED", "SA_Score", "PAINS_Count", "LE", "LLE",
    ]
    label_map = {
        "MW": "Mol. Weight",
        "LogP": "LogP",
        "HBD": "H-Bond Donors",
        "HBA": "H-Bond Acceptors",
        "TPSA": "TPSA",
        "RotBonds": "Rotatable Bonds",
        "HeavyAtoms": "Heavy Atoms",
        "QED": "QED",
        "SA_Score": "SA Score",
        "PAINS_Count": "PAINS Alerts",
        "LE": "Ligand Efficiency",
        "LLE": "Lipophilic LE",
    }

    rows = [
        [Paragraph("<b>Property</b>", STYLE_SMALL),
         Paragraph("<b>Value</b>", STYLE_SMALL)],
    ]
    for key in display_order:
        if key in props_dict:
            val = props_dict[key]
            if isinstance(val, float):
                val = f"{val:.2f}"
            rows.append([
                Paragraph(label_map.get(key, key), STYLE_SMALL),
                Paragraph(str(val), STYLE_SMALL),
            ])
    # Include any extra keys not in display_order
    for key, val in props_dict.items():
        if key not in display_order:
            if isinstance(val, float):
                val = f"{val:.2f}"
            rows.append([
                Paragraph(str(key), STYLE_SMALL),
                Paragraph(str(val), STYLE_SMALL),
            ])

    t = Table(rows, colWidths=[1.8 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _make_score_table(results_df):
    """Create a color-coded Table from a docking results DataFrame."""
    if results_df is None or results_df.empty:
        return Paragraph("<i>No docking results available.</i>", STYLE_BODY)

    df = results_df.copy()
    cols = list(df.columns)
    header = [Paragraph(f"<b>{c}</b>", STYLE_SMALL) for c in cols]
    data = [header]
    row_colors = []

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                val = f"{val:.2f}"
            cells.append(Paragraph(str(val), STYLE_SMALL))
        data.append(cells)
        # Determine row color from score column
        score_val = None
        for score_col in ["Score", "Best_Score", "Score (kcal/mol)", "score"]:
            if score_col in cols:
                score_val = row[score_col]
                break
        row_colors.append(_score_color(score_val))

    n_cols = len(cols)
    col_width = min(1.2 * inch, 6.5 * inch / max(n_cols, 1))
    t = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for i, rc in enumerate(row_colors):
        style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1),
                           colors.Color(rc.red, rc.green, rc.blue, alpha=0.15)))

    t.setStyle(TableStyle(style_cmds))
    return t


def _add_3d_views_grid(story, image_paths):
    """Add a 3-column x 2-row grid of 3D view images to the story."""
    if not image_paths:
        story.append(Paragraph("<i>No 3D view images available.</i>", STYLE_BODY))
        return

    valid = [p for p in image_paths if p and os.path.exists(p)]
    if not valid:
        story.append(Paragraph("<i>No 3D view images available.</i>", STYLE_BODY))
        return

    img_w = 2.2 * inch
    img_h = 1.8 * inch
    cells = []
    for p in valid[:6]:
        cells.append(Image(p, width=img_w, height=img_h))
    # Pad to fill grid
    while len(cells) < 6:
        cells.append(Paragraph("", STYLE_SMALL))

    grid_data = [cells[:3], cells[3:6]]
    t = Table(grid_data, colWidths=[img_w + 6] * 3)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


def _lipinski_check(props):
    """Return (pass/fail string, details) for Lipinski Rule of Five."""
    if not props:
        return "N/A", "No properties available"
    violations = 0
    details = []
    checks = [
        ("MW", 500, "MW <= 500"),
        ("LogP", 5, "LogP <= 5"),
        ("HBD", 5, "HBD <= 5"),
        ("HBA", 10, "HBA <= 10"),
    ]
    for key, limit, label in checks:
        val = props.get(key)
        if val is not None:
            try:
                if float(val) > limit:
                    violations += 1
                    details.append(f"{label}: FAIL ({val})")
                else:
                    details.append(f"{label}: PASS ({val})")
            except (ValueError, TypeError):
                details.append(f"{label}: N/A")
        else:
            details.append(f"{label}: N/A")
    status = "PASS" if violations <= 1 else "FAIL"
    return status, "; ".join(details)


def _build_summary_box(lines):
    """Build a light-gray summary box with key-value lines."""
    rows = []
    for key, val in lines:
        rows.append([
            Paragraph(f"<b>{key}:</b>", STYLE_SMALL),
            Paragraph(str(val), STYLE_SMALL),
        ])
    t = Table(rows, colWidths=[1.8 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _try_generate_2d_image(smiles, output_path):
    """Attempt to generate a 2D structure PNG from SMILES using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(400, 300))
        img.save(output_path)
        return output_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report 1: Single Docking
# ---------------------------------------------------------------------------


def generate_single_dock_pdf(
    output_path,
    pdb_id,
    ligand_name,
    smiles,
    results_df,
    properties,
    view_images=None,
    interactions_df=None,
    ligand_2d_path=None,
):
    """Generate a PDF report for a single ligand docking run.

    Args:
        output_path: Path for the output PDF file.
        pdb_id: PDB ID of the protein target.
        ligand_name: Name/identifier of the ligand.
        smiles: SMILES string of the ligand.
        results_df: DataFrame with docking poses and scores.
        properties: Dict with molecular properties (MW, LogP, etc.).
        view_images: List of up to 6 image paths for 3D views.
        interactions_df: Optional DataFrame of interaction fingerprints.
        ligand_2d_path: Optional path to 2D structure PNG.
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=80, bottomMargin=40,
        leftMargin=40, rightMargin=40,
    )

    def header_footer(canvas, doc):
        _make_header(canvas, doc, "Single Docking Report", pdb_id)

    story = []

    # --- Page 1: Summary ---
    best_score = None
    if results_df is not None and not results_df.empty:
        for col in ["Score", "Best_Score", "Score (kcal/mol)", "score"]:
            if col in results_df.columns:
                best_score = results_df[col].min()
                break

    summary_lines = [
        ("Protein Target", pdb_id or "N/A"),
        ("Ligand", ligand_name or "N/A"),
        ("SMILES", smiles or "N/A"),
        ("Best Score", f"{best_score:.2f} kcal/mol" if best_score is not None else "N/A"),
        ("Interpretation", _interpret_score(best_score)),
    ]
    story.append(_build_summary_box(summary_lines))
    story.append(Spacer(1, 12))

    # 2D structure and properties side by side
    left_items = []
    img_obj = None
    if ligand_2d_path and os.path.exists(ligand_2d_path):
        img_obj = Image(ligand_2d_path, width=3 * inch, height=2 * inch)
    elif smiles:
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "acudock_2d_tmp.png")
        result = _try_generate_2d_image(smiles, tmp_path)
        if result:
            img_obj = Image(tmp_path, width=3 * inch, height=2 * inch)

    if img_obj is None:
        img_obj = Paragraph("<i>[2D structure not available]</i>", STYLE_BODY)

    props_table = _make_properties_table(properties)

    side_by_side = Table(
        [[img_obj, props_table]],
        colWidths=[3.3 * inch, 3.5 * inch],
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 10))

    # Lipinski
    lip_status, lip_details = _lipinski_check(properties)
    lip_color = SCORE_GREEN if lip_status == "PASS" else SCORE_RED
    story.append(Paragraph(
        f'<b>Lipinski Rule of Five:</b> '
        f'<font color="{lip_color.hexval()}">{lip_status}</font>  '
        f'<font size="7">{lip_details}</font>',
        STYLE_BODY,
    ))

    # --- Page 2: Scores + 3D views ---
    story.append(PageBreak())
    story.append(Paragraph("Docking Scores (All Poses)", STYLE_H2))
    story.append(_make_score_table(results_df))
    story.append(Spacer(1, 16))

    if view_images:
        story.append(Paragraph("3D Binding Pose Views", STYLE_H2))
        _add_3d_views_grid(story, view_images)

    # --- Page 3: Interactions (optional) ---
    if interactions_df is not None and not interactions_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Protein-Ligand Interactions", STYLE_H2))
        story.append(_make_score_table(interactions_df))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output_path


# ---------------------------------------------------------------------------
# Report 2: Batch Screening
# ---------------------------------------------------------------------------


def generate_batch_pdf(
    output_path,
    pdb_id,
    batch_df,
    chart_path=None,
    per_ligand_images=None,
):
    """Generate a PDF report for a batch virtual screening campaign.

    Args:
        output_path: Path for the output PDF file.
        pdb_id: PDB ID of the protein target.
        batch_df: DataFrame with columns: Rank, Name, SMILES, Score (kcal/mol),
                  Est. Kd (uM), MW, LogP, QED, SA_Score, PAINS_Count, LE, Interpretation.
        chart_path: Optional path to a ranked bar chart image.
        per_ligand_images: Optional dict mapping ligand name to 2D structure PNG path.
    """
    per_ligand_images = per_ligand_images or {}
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=80, bottomMargin=40,
        leftMargin=40, rightMargin=40,
    )

    def header_footer(canvas, doc):
        _make_header(canvas, doc, "Batch Screening Report", pdb_id)

    story = []

    # Campaign summary
    n_total = len(batch_df) if batch_df is not None else 0
    n_strong = 0
    n_moderate = 0
    if batch_df is not None and not batch_df.empty:
        score_col = None
        for c in ["Score (kcal/mol)", "Score", "Best_Score"]:
            if c in batch_df.columns:
                score_col = c
                break
        if score_col:
            n_strong = int((batch_df[score_col] <= -8.0).sum())
            n_moderate = int(((batch_df[score_col] > -8.0) & (batch_df[score_col] <= -6.0)).sum())

    summary_lines = [
        ("Protein Target", pdb_id or "N/A"),
        ("Total Ligands", str(n_total)),
        ("Strong Hits (<= -8.0)", str(n_strong)),
        ("Moderate Hits (-8.0 to -6.0)", str(n_moderate)),
    ]
    story.append(_build_summary_box(summary_lines))
    story.append(Spacer(1, 12))

    # Bar chart
    if chart_path and os.path.exists(chart_path):
        story.append(Paragraph("Score Distribution", STYLE_H2))
        story.append(Image(chart_path, width=6.5 * inch, height=3.5 * inch))
        story.append(Spacer(1, 10))

    # Top 10 table
    story.append(Paragraph("Top 10 Hits", STYLE_H2))
    if batch_df is not None and not batch_df.empty:
        top10 = batch_df.head(10).copy()
        # Truncate SMILES for table display
        if "SMILES" in top10.columns:
            top10["SMILES"] = top10["SMILES"].apply(
                lambda s: (str(s)[:30] + "...") if len(str(s)) > 30 else str(s)
            )
        story.append(_make_score_table(top10))
    else:
        story.append(Paragraph("<i>No results available.</i>", STYLE_BODY))

    # Per-ligand detail pages (top 10)
    if batch_df is not None and not batch_df.empty:
        top10_full = batch_df.head(10)
        for _, row in top10_full.iterrows():
            story.append(PageBreak())
            name = row.get("Name", "Unknown")
            story.append(Paragraph(f"Ligand Detail: {name}", STYLE_H2))

            smiles_val = row.get("SMILES", "")
            detail_lines = [
                ("Name", name),
                ("SMILES", str(smiles_val)),
            ]
            for col in ["Score (kcal/mol)", "Est. Kd (uM)", "Interpretation"]:
                if col in row.index:
                    val = row[col]
                    detail_lines.append((col, f"{val:.2f}" if isinstance(val, float) else str(val)))
            story.append(_build_summary_box(detail_lines))
            story.append(Spacer(1, 8))

            # 2D image + properties side by side
            img_obj = None
            if name in per_ligand_images:
                img_path = per_ligand_images[name]
                if os.path.exists(img_path):
                    img_obj = Image(img_path, width=2.5 * inch, height=1.8 * inch)

            if img_obj is None and smiles_val:
                import tempfile as _tf
                tmp = os.path.join(_tf.gettempdir(), f"acudock_batch_{name[:20]}.png")
                if _try_generate_2d_image(str(smiles_val), tmp):
                    img_obj = Image(tmp, width=2.5 * inch, height=1.8 * inch)

            if img_obj is None:
                img_obj = Paragraph("<i>[2D structure not available]</i>", STYLE_SMALL)

            props = {}
            for key in ["MW", "LogP", "QED", "SA_Score", "PAINS_Count", "LE"]:
                if key in row.index:
                    props[key] = row[key]

            props_tbl = _make_properties_table(props)
            side = Table([[img_obj, props_tbl]], colWidths=[3 * inch, 3.5 * inch])
            side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(side)

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output_path


# ---------------------------------------------------------------------------
# Report 3: Multi-Protein Docking
# ---------------------------------------------------------------------------


def generate_multi_protein_pdf(
    output_path,
    ligand_name,
    smiles,
    results_df,
    chart_path=None,
    per_protein_views=None,
    ligand_2d_path=None,
):
    """Generate a PDF report for docking one ligand against multiple proteins.

    Args:
        output_path: Path for the output PDF file.
        ligand_name: Name/identifier of the ligand.
        smiles: SMILES string of the ligand.
        results_df: DataFrame with columns: PDB_ID, Best_Score, Est_Kd_uM,
                    Interpretation, Num_Poses, Prep_Time_s, Dock_Time_s, Engine, Error.
        chart_path: Optional path to a comparison bar chart image.
        per_protein_views: Optional dict mapping PDB_ID to list of 3D view image paths.
        ligand_2d_path: Optional path to 2D structure PNG.
    """
    per_protein_views = per_protein_views or {}
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=80, bottomMargin=40,
        leftMargin=40, rightMargin=40,
    )

    def header_footer(canvas, doc):
        _make_header(canvas, doc, "Multi-Protein Docking Report", ligand_name or "")

    story = []

    # Ligand info + 2D structure
    story.append(Paragraph("Ligand Information", STYLE_H2))

    img_obj = None
    if ligand_2d_path and os.path.exists(ligand_2d_path):
        img_obj = Image(ligand_2d_path, width=3 * inch, height=2 * inch)
    elif smiles:
        import tempfile as _tf
        tmp = os.path.join(_tf.gettempdir(), "acudock_multi_2d.png")
        if _try_generate_2d_image(smiles, tmp):
            img_obj = Image(tmp, width=3 * inch, height=2 * inch)

    info_lines = [
        ("Ligand", ligand_name or "N/A"),
        ("SMILES", smiles or "N/A"),
        ("Proteins Tested", str(len(results_df)) if results_df is not None else "0"),
    ]
    info_box = _build_summary_box(info_lines)

    if img_obj:
        side = Table([[img_obj, info_box]], colWidths=[3.3 * inch, 3.5 * inch])
        side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(side)
    else:
        story.append(info_box)
    story.append(Spacer(1, 12))

    # Comparison table
    story.append(Paragraph("Cross-Protein Comparison", STYLE_H2))
    if results_df is not None and not results_df.empty:
        display_df = results_df.copy()
        # Only show key columns if all are present
        show_cols = [c for c in [
            "PDB_ID", "Best_Score", "Est_Kd_uM", "Interpretation",
            "Num_Poses", "Dock_Time_s", "Engine", "Error",
        ] if c in display_df.columns]
        if show_cols:
            display_df = display_df[show_cols]
        story.append(_make_score_table(display_df))
    else:
        story.append(Paragraph("<i>No results available.</i>", STYLE_BODY))
    story.append(Spacer(1, 12))

    # Bar chart
    if chart_path and os.path.exists(chart_path):
        story.append(Paragraph("Score Comparison", STYLE_H2))
        story.append(Image(chart_path, width=6.5 * inch, height=3.5 * inch))

    # Per-protein detail pages
    if results_df is not None and not results_df.empty:
        # Show detail for at least the best hit, up to all proteins
        sorted_df = results_df.copy()
        if "Best_Score" in sorted_df.columns:
            sorted_df = sorted_df.sort_values("Best_Score", ascending=True)

        for _, row in sorted_df.iterrows():
            pdb_id = row.get("PDB_ID", "Unknown")
            story.append(PageBreak())
            story.append(Paragraph(f"Protein: {pdb_id}", STYLE_H2))

            detail_lines = []
            for col in ["PDB_ID", "Best_Score", "Est_Kd_uM", "Interpretation",
                        "Num_Poses", "Prep_Time_s", "Dock_Time_s", "Engine"]:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    if isinstance(val, float):
                        val = f"{val:.2f}"
                    detail_lines.append((col.replace("_", " "), str(val)))

            error_val = row.get("Error", "")
            if error_val and str(error_val) != "nan" and str(error_val) != "":
                detail_lines.append(("Error", str(error_val)))

            story.append(_build_summary_box(detail_lines))
            story.append(Spacer(1, 10))

            # 3D views for this protein
            if pdb_id in per_protein_views:
                story.append(Paragraph("3D Binding Pose Views", STYLE_H3))
                _add_3d_views_grid(story, per_protein_views[pdb_id])

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output_path


# ---------------------------------------------------------------------------
# Report 4: Scout (Active Learning)
# ---------------------------------------------------------------------------


def generate_scout_pdf(
    output_path,
    pdb_id,
    campaign_stats,
    top_hits_df,
    convergence_img=None,
    distribution_img=None,
    surrogate_img=None,
):
    """Generate a PDF report for an AcuDock Scout active-learning campaign.

    Args:
        output_path: Path for the output PDF file.
        pdb_id: PDB ID of the protein target.
        campaign_stats: Dict with campaign parameters (total_library, docked_count,
                        rounds, batch_size, surrogate_model, acquisition_fn, etc.).
        top_hits_df: DataFrame with columns: Name, SMILES, Best_Score, MW, LogP,
                     QED, SA_Score, PAINS_Count, LE.
        convergence_img: Optional path to convergence plot PNG.
        distribution_img: Optional path to score distribution PNG.
        surrogate_img: Optional path to surrogate model performance PNG.
    """
    campaign_stats = campaign_stats or {}
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=80, bottomMargin=40,
        leftMargin=40, rightMargin=40,
    )

    def header_footer(canvas, doc):
        _make_header(canvas, doc, "Scout Campaign Report", pdb_id)

    story = []

    # Campaign summary
    story.append(Paragraph("Campaign Summary", STYLE_H2))
    summary_lines = [
        ("Protein Target", campaign_stats.get("pdb_id", pdb_id or "N/A")),
        ("Library Size", str(campaign_stats.get("total_library", "N/A"))),
        ("Compounds Docked", str(campaign_stats.get("docked_count", "N/A"))),
        ("Rounds Completed", str(campaign_stats.get("rounds", "N/A"))),
        ("Batch Size", str(campaign_stats.get("batch_size", "N/A"))),
        ("Surrogate Model", str(campaign_stats.get("surrogate_model", "Random Forest"))),
        ("Acquisition Function", str(campaign_stats.get("acquisition_fn", "UCB"))),
    ]
    # Compute efficiency if data available
    total_lib = campaign_stats.get("total_library")
    docked = campaign_stats.get("docked_count")
    if total_lib and docked:
        try:
            pct = 100.0 * int(docked) / int(total_lib)
            summary_lines.append(("Library Explored", f"{pct:.1f}%"))
        except (ValueError, ZeroDivisionError):
            pass
    story.append(_build_summary_box(summary_lines))
    story.append(Spacer(1, 14))

    # Convergence plot
    if convergence_img and os.path.exists(convergence_img):
        story.append(Paragraph("Convergence Plot", STYLE_H2))
        story.append(Image(convergence_img, width=6.5 * inch, height=3 * inch))
        story.append(Spacer(1, 10))

    # Score distribution
    if distribution_img and os.path.exists(distribution_img):
        story.append(Paragraph("Score Distribution", STYLE_H2))
        story.append(Image(distribution_img, width=6.5 * inch, height=3 * inch))
        story.append(Spacer(1, 10))

    # Surrogate model performance
    if surrogate_img and os.path.exists(surrogate_img):
        story.append(Paragraph("Surrogate Model Performance", STYLE_H2))
        story.append(Image(surrogate_img, width=6.5 * inch, height=3 * inch))

    # --- Page 2: Top 20 hits table ---
    story.append(PageBreak())
    story.append(Paragraph("Top 20 Hits", STYLE_H2))
    if top_hits_df is not None and not top_hits_df.empty:
        top20 = top_hits_df.head(20).copy()
        if "SMILES" in top20.columns:
            top20["SMILES"] = top20["SMILES"].apply(
                lambda s: (str(s)[:25] + "...") if len(str(s)) > 25 else str(s)
            )
        story.append(_make_score_table(top20))
    else:
        story.append(Paragraph("<i>No hits available.</i>", STYLE_BODY))

    # --- Page 3: Top 5 2D structures grid ---
    story.append(PageBreak())
    story.append(Paragraph("Top 5 Hit Structures", STYLE_H2))

    if top_hits_df is not None and not top_hits_df.empty:
        top5 = top_hits_df.head(5)
        struct_cells_row1 = []
        struct_cells_row2 = []
        labels_row1 = []
        labels_row2 = []

        import tempfile as _tf
        tmp_dir = _tf.gettempdir()

        for i, (_, row) in enumerate(top5.iterrows()):
            smi = row.get("SMILES", "")
            name = row.get("Name", f"Hit_{i+1}")
            score = row.get("Best_Score", "")

            img_path = os.path.join(tmp_dir, f"acudock_scout_hit_{i}.png")
            generated = _try_generate_2d_image(str(smi), img_path)

            if generated:
                img_obj = Image(img_path, width=2 * inch, height=1.5 * inch)
            else:
                img_obj = Paragraph("<i>[Structure N/A]</i>", STYLE_SMALL)

            score_str = f"{float(score):.2f}" if score != "" else "N/A"
            label = Paragraph(
                f"<b>{name}</b><br/>{score_str} kcal/mol", STYLE_CENTER
            )

            if i < 3:
                struct_cells_row1.append(img_obj)
                labels_row1.append(label)
            else:
                struct_cells_row2.append(img_obj)
                labels_row2.append(label)

        # Pad row 2 if needed
        while len(struct_cells_row2) < 3:
            struct_cells_row2.append(Paragraph("", STYLE_SMALL))
            labels_row2.append(Paragraph("", STYLE_SMALL))

        cw = 2.2 * inch
        grid_data = [struct_cells_row1, labels_row1]
        if any(isinstance(c, Image) for c in struct_cells_row2):
            grid_data.extend([struct_cells_row2, labels_row2])

        t = Table(grid_data, colWidths=[cw] * 3)
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No hit structures available.</i>", STYLE_BODY))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output_path
