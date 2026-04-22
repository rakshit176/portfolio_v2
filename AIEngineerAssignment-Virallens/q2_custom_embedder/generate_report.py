"""
Q2 Custom Embedder Analysis Report - PDF Generator
Generates a professional PDF report with cover, TOC, results tables, and embedded plots.
"""
import os, sys, json
from pathlib import Path

# ━━ Paths ━━
PROJECT_DIR = Path("/home/z/my-project/q2_custom_embedder")
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
DOWNLOAD_DIR = Path("/home/z/my-project/download")
PDF_SKILL_DIR = Path("/home/z/my-project/skills/pdf")

sys.path.insert(0, str(PDF_SKILL_DIR / "scripts"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, CondPageBreak
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import SimpleDocTemplate
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import hashlib

# ━━ Color Palette ━━
ACCENT = colors.HexColor('#4f23d5')
TEXT_PRIMARY = colors.HexColor('#1c1b1a')
TEXT_MUTED = colors.HexColor('#8f8b83')
BG_SURFACE = colors.HexColor('#e7e4df')
BG_PAGE = colors.HexColor('#f3f2f0')
TABLE_HEADER_COLOR = ACCENT
TABLE_HEADER_TEXT = colors.white
TABLE_ROW_EVEN = colors.white
TABLE_ROW_ODD = BG_SURFACE

# ━━ Fonts ━━
pdfmetrics.registerFont(TTFont('TimesNewRoman', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('TimesNewRomanBold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Calibri', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
registerFontFamily('TimesNewRoman', normal='TimesNewRoman', bold='TimesNewRomanBold')
registerFontFamily('Calibri', normal='Calibri', bold='Calibri')

# ━━ Load evaluation data ━━
with open(RESULTS_DIR / "evaluation_results.json") as f:
    eval_data = json.load(f)
with open(RESULTS_DIR / "pipeline_summary.json") as f:
    summary = json.load(f)

# ━━ Page dimensions ━━
PAGE_W, PAGE_H = A4
LEFT_MARGIN = 1.0 * inch
RIGHT_MARGIN = 1.0 * inch
TOP_MARGIN = 0.8 * inch
BOTTOM_MARGIN = 0.8 * inch
AVAILABLE_W = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN

# ━━ Styles ━━
styles = getSampleStyleSheet()

cover_title_style = ParagraphStyle(
    name='CoverTitle', fontName='TimesNewRoman', fontSize=32, leading=42,
    alignment=TA_LEFT, textColor=TEXT_PRIMARY, spaceAfter=12,
)
cover_subtitle_style = ParagraphStyle(
    name='CoverSubtitle', fontName='Calibri', fontSize=16, leading=22,
    alignment=TA_LEFT, textColor=TEXT_MUTED, spaceAfter=6,
)
cover_meta_style = ParagraphStyle(
    name='CoverMeta', fontName='Calibri', fontSize=11, leading=16,
    alignment=TA_LEFT, textColor=TEXT_MUTED,
)

h1_style = ParagraphStyle(
    name='H1', fontName='TimesNewRoman', fontSize=20, leading=26,
    textColor=TEXT_PRIMARY, spaceBefore=18, spaceAfter=10,
)
h2_style = ParagraphStyle(
    name='H2', fontName='TimesNewRoman', fontSize=15, leading=20,
    textColor=ACCENT, spaceBefore=14, spaceAfter=8,
)
body_style = ParagraphStyle(
    name='Body', fontName='TimesNewRoman', fontSize=10.5, leading=17,
    alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY, spaceAfter=8,
    firstLineIndent=0,
)
caption_style = ParagraphStyle(
    name='Caption', fontName='Calibri', fontSize=9, leading=13,
    alignment=TA_CENTER, textColor=TEXT_MUTED, spaceBefore=4, spaceAfter=12,
)
header_cell_style = ParagraphStyle(
    name='HeaderCell', fontName='TimesNewRoman', fontSize=10,
    textColor=colors.white, alignment=TA_CENTER,
)
cell_style = ParagraphStyle(
    name='Cell', fontName='TimesNewRoman', fontSize=10,
    textColor=TEXT_PRIMARY, alignment=TA_CENTER,
)
cell_left_style = ParagraphStyle(
    name='CellLeft', fontName='TimesNewRoman', fontSize=10,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT,
)

# ━━ TOC Template ━━
class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))

def add_heading(text, style, level=0):
    key = 'h_%s' % hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph('<a name="%s"/>%s' % (key, text), style)
    p.bookmark_name = text
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p

H1_ORPHAN_THRESHOLD = (PAGE_H - TOP_MARGIN - BOTTOM_MARGIN) * 0.15

def add_major_section(text):
    return [
        CondPageBreak(H1_ORPHAN_THRESHOLD),
        add_heading(text, h1_style, level=0),
    ]

def make_table(data, col_widths):
    table = Table(data, colWidths=col_widths, hAlign='CENTER')
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), TABLE_HEADER_TEXT),
        ('GRID', (0, 0), (-1, -1), 0.5, TEXT_MUTED),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        bg = TABLE_ROW_EVEN if i % 2 == 1 else TABLE_ROW_ODD
        style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    table.setStyle(TableStyle(style_cmds))
    return table

def embed_plot(plot_path, caption_text, width=None):
    if not plot_path.exists():
        return [Paragraph(f'[Plot not found: {plot_path.name}]', body_style)]
    w = width or (AVAILABLE_W * 0.95)
    img = Image(str(plot_path), width=w, height=w * 0.55)
    img.hAlign = 'CENTER'
    cap = Paragraph(caption_text, caption_style)
    return [Spacer(1, 8), img, cap]

# ━━ Build PDF ━━
pdf_path = DOWNLOAD_DIR / "Q2_Custom_Embedder_Report.pdf"
pdf_path.parent.mkdir(parents=True, exist_ok=True)

doc = TocDocTemplate(
    str(pdf_path), pagesize=A4,
    leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
    topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
)

story = []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COVER PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(Spacer(1, 3.0 * inch))

# Accent line
cover_line_data = [['']]
cover_line = Table(cover_line_data, colWidths=[AVAILABLE_W * 0.35], rowHeights=[3])
cover_line.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), ACCENT),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
]))
story.append(cover_line)
story.append(Spacer(1, 24))

story.append(Paragraph('<b>Q2: Custom Embedder for<br/>Domain Documents</b>', cover_title_style))
story.append(Spacer(1, 12))
story.append(Paragraph(
    'Fine-Tuning Sentence Transformers on Legal Domain PDFs:<br/>'
    'Embedding Quality, Clustering, and Evaluation Analysis',
    cover_subtitle_style
))
story.append(Spacer(1, 36))

# Meta info
story.append(Paragraph('Pipeline: Preprocessing | Embedding | Clustering | Evaluation | Visualization', cover_meta_style))
story.append(Spacer(1, 6))
story.append(Paragraph(f'Documents Processed: {summary["documents"]}  |  Chunks Created: {summary["chunks"]}', cover_meta_style))
story.append(Spacer(1, 6))
story.append(Paragraph('Embedding Model: all-MiniLM-L6-v2 (384-dim)  |  Training: MNRL', cover_meta_style))

story.append(PageBreak())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLE OF CONTENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(Paragraph('<b>Table of Contents</b>', h1_style))
story.append(Spacer(1, 12))

toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle(name='TOC1', fontSize=12, leftIndent=20, fontName='TimesNewRoman',
                   spaceBefore=6, spaceAfter=4, leading=16),
    ParagraphStyle(name='TOC2', fontSize=10.5, leftIndent=40, fontName='Calibri',
                   spaceBefore=3, spaceAfter=2, leading=14, textColor=TEXT_MUTED),
]
story.append(toc)
story.append(PageBreak())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EXECUTIVE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>1. Executive Summary</b>'))

story.append(Paragraph(
    'This report presents a comprehensive analysis of a custom embedding model fine-tuned on legal domain '
    'documents, compared against a general-purpose baseline (all-MiniLM-L6-v2). The pipeline processes five '
    'legal PDF documents through text extraction, cleaning, and semantic chunking, producing 688 text chunks '
    'that serve as the corpus for both embedding generation and downstream evaluation. The fine-tuned model '
    'is trained using Multiple Negatives Ranking Loss (MNRL), which treats chunks from the same document as '
    'positive pairs and all other chunks in the batch as negatives, thereby encouraging the model to learn '
    'domain-specific semantic relationships.',
    body_style
))
story.append(Paragraph(
    'The evaluation framework employs multiple complementary metrics to assess embedding quality: within-document '
    'versus between-document cosine similarity, KMeans silhouette scores across a range of cluster counts, '
    'K-neighbor agreement (Jaccard overlap) between the baseline and custom embedding spaces, and Spearman/Pearson '
    'correlation of pairwise similarity matrices. These metrics collectively provide a multi-dimensional view of '
    'how fine-tuning affects the embedding geometry and its utility for downstream clustering tasks.',
    body_style
))

# Key metrics callout
baseline_sep = eval_data['summary']['baseline_separation_ratio']
custom_sep = eval_data['summary']['custom_separation_ratio']
improvement = eval_data['summary']['improvement_pct']
neighbor_agree = eval_data['summary']['neighbor_agreement']
spearman = eval_data['summary']['embedding_correlation']

metrics_data = [
    [Paragraph('<b>Metric</b>', header_cell_style),
     Paragraph('<b>Value</b>', header_cell_style)],
    [Paragraph('Baseline Separation Ratio', cell_left_style),
     Paragraph(f'{baseline_sep:.4f}', cell_style)],
    [Paragraph('Custom Separation Ratio', cell_left_style),
     Paragraph(f'{custom_sep:.4f}', cell_style)],
    [Paragraph('K-Neighbor Agreement (Jaccard)', cell_left_style),
     Paragraph(f'{neighbor_agree:.4f}', cell_style)],
    [Paragraph('Embedding Correlation (Spearman)', cell_left_style),
     Paragraph(f'{spearman:.4f}', cell_style)],
    [Paragraph('Best Baseline Silhouette', cell_left_style),
     Paragraph(f'{summary["baseline_best_silhouette"]:.4f}', cell_style)],
    [Paragraph('Best Custom Silhouette', cell_left_style),
     Paragraph(f'{summary["custom_best_silhouette"]:.4f}', cell_style)],
]
story.append(Spacer(1, 18))
t = make_table(metrics_data, [AVAILABLE_W * 0.65, AVAILABLE_W * 0.35])
story.append(t)
story.append(Paragraph('Table 1: Key evaluation metrics summary.', caption_style))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. METHODOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>2. Methodology</b>'))

story.append(Paragraph('<b>2.1 Preprocessing Pipeline</b>', h2_style))
story.append(Paragraph(
    'The preprocessing stage extracts raw text from legal PDF documents using PyMuPDF (fitz), applies '
    'domain-specific cleaning rules, and splits the cleaned text into semantically meaningful chunks with '
    'overlap. The cleaning process removes boilerplate headers/footers (e.g., "Page X of Y", case numbers), '
    'normalizes whitespace, strips non-printable characters, and removes orphan single-word uppercase lines '
    'that are likely headers or section markers. The chunking employs a RecursiveCharacterTextSplitter from '
    'LangChain with a chunk size of 512 characters, overlap of 64 characters, and minimum chunk length of '
    '50 characters to ensure each chunk carries sufficient semantic content for meaningful embedding comparison.',
    body_style
))

story.append(Paragraph('<b>2.2 Embedding Models</b>', h2_style))
story.append(Paragraph(
    'The baseline model is all-MiniLM-L6-v2, a compact 384-dimensional sentence embedding model from the '
    'sentence-transformers library, trained on over 1 billion sentence pairs. It serves as a strong general-purpose '
    'baseline. The custom model starts from the same pre-trained checkpoint and is fine-tuned on the legal corpus '
    'using Multiple Negatives Ranking Loss (MNRL). This loss function is particularly effective for fine-tuning '
    'sentence embeddings because it only requires positive pairs (chunks from the same document) and treats all '
    'other items in the batch as negatives, avoiding the need for explicit hard negative mining. The training '
    'uses a manual loop with AdamW optimizer (learning rate 2e-5, linear warmup over 10% of steps) to minimize '
    'memory consumption on CPU-based environments.',
    body_style
))

story.append(Paragraph('<b>2.3 Clustering</b>', h2_style))
story.append(Paragraph(
    'Both embedding spaces are clustered using KMeans with a sweep over k values from 3 to 8. The optimal k is '
    'selected by maximizing the mean silhouette coefficient, which measures how well each point fits within its '
    'assigned cluster relative to neighboring clusters. HDBSCAN, a density-based clustering algorithm, is also '
    'applied as a complementary approach that can discover clusters of arbitrary shape and automatically label '
    'noise points. Three internal validation metrics are computed for each KMeans configuration: silhouette score, '
    'Davies-Bouldin Index (lower is better), and Calinski-Harabasz Index (higher is better).',
    body_style
))

story.append(Paragraph('<b>2.4 Evaluation Framework</b>', h2_style))
story.append(Paragraph(
    'The evaluation framework is designed to assess embedding quality from four complementary perspectives. '
    'First, within-document versus between-document similarity measures whether chunks from the same source '
    'document are embedded closer together than chunks from different documents, expressed as a separation '
    'ratio (within / between). Second, K-neighbor agreement quantifies the overlap in top-k nearest neighbors '
    'between the baseline and custom embedding spaces using Jaccard similarity, revealing how much the fine-tuning '
    'alters the local neighborhood structure. Third, embedding correlation computes Spearman and Pearson correlations '
    'between pairwise similarity matrices of the two models, capturing global rank-order preservation. Fourth, '
    'clustering quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) assess the practical utility of the '
    'embeddings for downstream document organization tasks.',
    body_style
))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>3. Results</b>'))

story.append(Paragraph('<b>3.1 Embedding Quality Comparison</b>', h2_style))
story.append(Paragraph(
    'The following table presents a detailed comparison of embedding quality metrics between the baseline '
    'and custom models. The custom embedder achieves a notably higher within-document similarity (0.4170 vs. '
    '0.3631), indicating that fine-tuning successfully pulls chunks from the same document closer together '
    'in the embedding space. However, the between-document similarity also increases (0.3589 vs. 0.2923), '
    'which partially offsets the gains in absolute separation. The net result is a slight decrease in the '
    'separation ratio from 1.2423 to 1.1617 (-6.5%).',
    body_style
))

bs = eval_data['baseline_similarity']
cs = eval_data['custom_similarity']

sim_data = [
    [Paragraph('<b>Metric</b>', header_cell_style),
     Paragraph('<b>Baseline</b>', header_cell_style),
     Paragraph('<b>Custom</b>', header_cell_style),
     Paragraph('<b>Change</b>', header_cell_style)],
    [Paragraph('Within-Doc Similarity', cell_left_style),
     Paragraph(f'{bs["within_mean"]:.4f}', cell_style),
     Paragraph(f'{cs["within_mean"]:.4f}', cell_style),
     Paragraph(f'+{((cs["within_mean"]-bs["within_mean"])/bs["within_mean"]*100):.1f}%', cell_style)],
    [Paragraph('Within-Doc Std Dev', cell_left_style),
     Paragraph(f'{bs["within_std"]:.4f}', cell_style),
     Paragraph(f'{cs["within_std"]:.4f}', cell_style),
     Paragraph('-', cell_style)],
    [Paragraph('Between-Doc Similarity', cell_left_style),
     Paragraph(f'{bs["between_mean"]:.4f}', cell_style),
     Paragraph(f'{cs["between_mean"]:.4f}', cell_style),
     Paragraph(f'+{((cs["between_mean"]-bs["between_mean"])/bs["between_mean"]*100):.1f}%', cell_style)],
    [Paragraph('Between-Doc Std Dev', cell_left_style),
     Paragraph(f'{bs["between_std"]:.4f}', cell_style),
     Paragraph(f'{cs["between_std"]:.4f}', cell_style),
     Paragraph('-', cell_style)],
    [Paragraph('<b>Separation Ratio</b>', cell_left_style),
     Paragraph(f'<b>{bs["separation_ratio"]:.4f}</b>', cell_style),
     Paragraph(f'<b>{cs["separation_ratio"]:.4f}</b>', cell_style),
     Paragraph(f'<b>{improvement:+.1f}%</b>', cell_style)],
]
story.append(Spacer(1, 18))
t = make_table(sim_data, [AVAILABLE_W * 0.32, AVAILABLE_W * 0.22, AVAILABLE_W * 0.22, AVAILABLE_W * 0.24])
story.append(t)
story.append(Paragraph('Table 2: Within-document vs. between-document similarity comparison.', caption_style))

story.append(Paragraph('<b>3.2 Clustering Quality</b>', h2_style))
story.append(Paragraph(
    f'The KMeans clustering sweep found that the baseline model achieved its best silhouette score of '
    f'{summary["baseline_best_silhouette"]:.4f} while the custom model achieved {summary["custom_best_silhouette"]:.4f}. '
    f'The relatively low silhouette scores for both models (below 0.1) suggest that the legal documents in '
    f'this corpus share substantial topical overlap, making clean cluster separation inherently challenging. '
    f'The fine-tuned model shows a slight decrease in silhouette score, which is consistent with the observation '
    f'that fine-tuning increases between-document similarity, effectively making the embedding space more '
    f'compact overall.',
    body_style
))

story.append(Paragraph('<b>3.3 Neighbor Agreement Analysis</b>', h2_style))
na = eval_data['neighbor_agreement']
ec = eval_data['embedding_correlation']
story.append(Paragraph(
    f'The K-neighbor agreement analysis reveals a mean Jaccard overlap of {na["mean_jaccard"]:.4f} between '
    f'the top-5 neighbor sets of the baseline and custom embeddings. This moderate agreement indicates that '
    f'fine-tuning preserves roughly 46% of the local neighborhood structure while introducing meaningful '
    f'changes to the remaining 54%. The agreement rate (fraction of points with Jaccard > 0.3) is '
    f'{na["agreement_rate"]:.1%}, suggesting that most chunks retain at least partial neighborhood consistency.',
    body_style
))

neighbor_data = [
    [Paragraph('<b>Agreement Metric</b>', header_cell_style),
     Paragraph('<b>Value</b>', header_cell_style)],
    [Paragraph('Mean Jaccard Overlap (top-5)', cell_left_style),
     Paragraph(f'{na["mean_jaccard"]:.4f}', cell_style)],
    [Paragraph('Median Jaccard Overlap', cell_left_style),
     Paragraph(f'{na["median_jaccard"]:.4f}', cell_style)],
    [Paragraph('Agreement Rate (> 0.3)', cell_left_style),
     Paragraph(f'{na["agreement_rate"]:.1%}', cell_style)],
    [Paragraph('Spearman Correlation', cell_left_style),
     Paragraph(f'{ec["spearman_rho"]:.4f} (p={ec["spearman_p"]:.4f})', cell_style)],
    [Paragraph('Pearson Correlation', cell_left_style),
     Paragraph(f'{ec["pearson_r"]:.4f} (p={ec["pearson_p"]:.4f})', cell_style)],
]
story.append(Spacer(1, 18))
t = make_table(neighbor_data, [AVAILABLE_W * 0.55, AVAILABLE_W * 0.45])
story.append(t)
story.append(Paragraph('Table 3: Neighbor agreement and embedding correlation metrics.', caption_style))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. VISUALIZATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>4. Visualizations</b>'))

story.append(Paragraph(
    'The following visualizations provide intuitive representations of the embedding spaces, clustering '
    'behavior, and evaluation metrics. t-SNE projections reduce the 384-dimensional embeddings to two '
    'dimensions for visual inspection, with points colored by source document to reveal how well each '
    'model separates documents in the reduced space. The silhouette comparison chart shows scores across '
    'multiple cluster counts, while the similarity analysis and neighbor overlap plots provide additional '
    'perspectives on embedding quality.',
    body_style
))

story.append(Paragraph('<b>4.1 t-SNE Embedding Space Comparison</b>', h2_style))
story.extend(embed_plot(
    PLOTS_DIR / 'tsne_comparison.png',
    'Figure 1: t-SNE projections of baseline (left) and custom (right) embeddings, colored by source document.'
))

story.append(Paragraph('<b>4.2 Silhouette Score Comparison</b>', h2_style))
story.extend(embed_plot(
    PLOTS_DIR / 'silhouette_comparison.png',
    'Figure 2: Silhouette scores across cluster counts (k=2 to k=8) for baseline and custom embeddings.'
))

story.append(Paragraph('<b>4.3 Similarity Analysis</b>', h2_style))
story.extend(embed_plot(
    PLOTS_DIR / 'similarity_analysis.png',
    'Figure 3: Within-document vs. between-document similarity and separation ratio comparison.'
))

story.append(Paragraph('<b>4.4 Cluster Distribution</b>', h2_style))
story.extend(embed_plot(
    PLOTS_DIR / 'cluster_distribution.png',
    'Figure 4: Cluster size distribution and source composition per cluster (custom embedder, k=5).'
))

story.append(Paragraph('<b>4.5 Neighbor Overlap by Source Document</b>', h2_style))
story.extend(embed_plot(
    PLOTS_DIR / 'neighbor_overlap.png',
    'Figure 5: Average Jaccard overlap of top-5 neighbors between baseline and custom, grouped by source.'
))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ANALYSIS AND DISCUSSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>5. Analysis and Discussion</b>'))

story.append(Paragraph('<b>5.1 Key Observations</b>', h2_style))
story.append(Paragraph(
    'The experimental results reveal several important patterns about the behavior of domain-specific '
    'fine-tuning on sentence embeddings. First, the custom embedder successfully increases absolute '
    'within-document similarity by 14.8%, confirming that the MNRL training objective effectively learns '
    'to group semantically related legal text from the same source document. This is a meaningful improvement '
    'that directly enhances the utility of the embeddings for document retrieval and similarity search tasks.',
    body_style
))
story.append(Paragraph(
    'Second, the between-document similarity also increases substantially (+22.8%), which partially offsets '
    'the within-document gains. This phenomenon can be attributed to the nature of the legal corpus: all five '
    'documents share a common legal vocabulary, syntactic patterns, and domain-specific terminology. When the '
    'model fine-tunes on this corpus, it learns to represent legal language more uniformly, which inadvertently '
    'brings documents closer together even when they originate from different cases or legal proceedings. The net '
    'effect is a slight decrease in the separation ratio (-6.5%).',
    body_style
))
story.append(Paragraph(
    'Third, the high Spearman correlation (0.8642) between the baseline and custom similarity matrices '
    'indicates that fine-tuning preserves the global rank-order structure of the embedding space. This means '
    'that if document pair A is more similar than pair B in the baseline space, this relationship is largely '
    'maintained in the custom space. The moderate neighbor agreement (Jaccard 0.4638) further confirms that '
    'while the overall structure is preserved, meaningful local changes occur that could be beneficial for '
    'domain-specific tasks.',
    body_style
))

story.append(Paragraph('<b>5.2 Limitations and Considerations</b>', h2_style))
story.append(Paragraph(
    'Several factors may limit the effectiveness of the domain fine-tuning observed in this study. The corpus '
    'consists of only five legal documents with 688 total chunks, which is a relatively small training set '
    'for contrastive learning. The single training epoch further limits the model\'s capacity to learn complex '
    'domain patterns. Additionally, all documents belong to the legal domain and share significant vocabulary '
    'overlap, which makes inter-document separation inherently more difficult compared to a multi-domain corpus '
    'where documents originate from fundamentally different topical areas.',
    body_style
))
story.append(Paragraph(
    'The low absolute silhouette scores (0.07-0.08) for both models suggest that the legal documents in this '
    'corpus are not strongly separable in embedding space, regardless of the model used. This is a data property '
    'rather than a model deficiency, and it underscores the importance of selecting evaluation metrics that are '
    'appropriate for the specific clustering difficulty of the corpus.',
    body_style
))

story.append(Paragraph('<b>5.3 Recommendations</b>', h2_style))
story.append(Paragraph(
    'To improve domain-specific embedding quality, several strategies can be explored in future work. Increasing '
    'the training corpus size with more diverse legal documents (different case types, jurisdictions, and time '
    'periods) would provide the contrastive learning objective with harder negatives and more varied positive '
    'pairs. Training for additional epochs with learning rate scheduling and early stopping based on a validation '
    'set would allow the model to converge to a better optimum. Incorporating hard negative mining, where '
    'semantically similar but topically distinct chunks are explicitly paired as negatives, could improve the '
    'model\'s ability to distinguish between related but different legal concepts. Finally, experimenting with '
    'larger base models (e.g., all-mpnet-base-v2) or domain-adaptive pre-training (DAPT) on a large legal corpus '
    'before fine-tuning could yield embeddings with stronger domain-specific performance while maintaining '
    'general-purpose capabilities.',
    body_style
))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SYSTEM ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.extend(add_major_section('<b>6. System Architecture and Implementation</b>'))

story.append(Paragraph('<b>6.1 Pipeline Components</b>', h2_style))
story.append(Paragraph(
    'The Q2 pipeline is implemented as a modular Python project with clear separation of concerns across six '
    'core modules: preprocessing (text extraction, cleaning, and chunking), embedding (baseline and custom model '
    'wrappers), training (data generation and MNRL fine-tuning), clustering (KMeans and HDBSCAN wrappers), '
    'evaluation (similarity, neighbor agreement, and correlation metrics), and visualization (t-SNE, silhouette, '
    'and cluster distribution plots). Each module exposes a clean API with documented parameters, and the entire '
    'pipeline is orchestrated by a single entry point (main.py) that executes the stages sequentially with '
    'detailed logging and timing information.',
    body_style
))

story.append(Paragraph('<b>6.2 Testing Infrastructure</b>', h2_style))
story.append(Paragraph(
    'The project includes a comprehensive test suite organized into unit and integration tests. Unit tests cover '
    'each module independently using mocked dependencies (no real PDFs, no model downloads, no network activity). '
    'The test suite includes tests for text preprocessing (cleaning rules, chunking with metadata), configuration '
    'management (defaults, singleton behavior, path properties), embedding models (load/embed/save with mocked '
    'SentenceTransformer), clustering algorithms (KMeans metrics, HDBSCAN with mocked imports, cluster summaries), '
    'evaluation metrics (similarity distributions, neighbor agreement, correlation calculations), visualization '
    '(plot generation with mocked matplotlib/sklearn), and the training pipeline (pair generation, loss computation). '
    'Integration tests verify the end-to-end pipeline flow from preprocessing through evaluation with all external '
    'dependencies fully mocked.',
    body_style
))

# Config table
config_data = [
    [Paragraph('<b>Component</b>', header_cell_style),
     Paragraph('<b>Configuration</b>', header_cell_style)],
    [Paragraph('Chunk Size / Overlap', cell_left_style),
     Paragraph('512 chars / 64 chars', cell_style)],
    [Paragraph('Base Model', cell_left_style),
     Paragraph('all-MiniLM-L6-v2 (384-dim)', cell_style)],
    [Paragraph('Max Sequence Length', cell_left_style),
     Paragraph('256 tokens', cell_style)],
    [Paragraph('Training Loss', cell_left_style),
     Paragraph('MNRL (Multiple Negatives Ranking Loss)', cell_style)],
    [Paragraph('Learning Rate', cell_left_style),
     Paragraph('2e-5 with linear warmup (10%)', cell_style)],
    [Paragraph('Training Epochs', cell_left_style),
     Paragraph('1', cell_style)],
    [Paragraph('KMeans k Range', cell_left_style),
     Paragraph('[3, 4, 5, 6, 7, 8]', cell_style)],
    [Paragraph('HDBSCAN Min Cluster Size', cell_left_style),
     Paragraph('5', cell_style)],
    [Paragraph('Evaluation Top-K', cell_left_style),
     Paragraph('5', cell_style)],
    [Paragraph('t-SNE Perplexity', cell_left_style),
     Paragraph('30.0', cell_style)],
]
story.append(Spacer(1, 18))
t = make_table(config_data, [AVAILABLE_W * 0.45, AVAILABLE_W * 0.55])
story.append(t)
story.append(Paragraph('Table 4: Pipeline configuration parameters.', caption_style))

# ━━ Build ━━
doc.multiBuild(story)
print(f"PDF generated: {pdf_path}")
print(f"File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
