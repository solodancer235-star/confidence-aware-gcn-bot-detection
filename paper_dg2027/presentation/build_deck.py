"""Fill the IEEE SIST/DG presentation template with the paper's content."""
import copy

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x6D, 0xA4)
SKY = RGBColor(0x3A, 0xAF, 0xDF)
TINT = RGBColor(0xEA, 0xF3, 0xFA)
GRAY = RGBColor(0x59, 0x59, 0x59)
LGRAY = RGBColor(0xA6, 0xA6, 0xA6)
ORANGE = RGBColor(0xE8, 0x77, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Times New Roman"

prs = Presentation("work.pptx")
slides = list(prs.slides)


def set_runs(paragraph, text, **fmt):
    """Keep the first run's formatting, replace text, drop the other runs."""
    runs = paragraph.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for k, v in fmt.items():
        setattr(runs[0].font, k, v)


def header(slide, text):
    for sh in list(slide.shapes):
        if sh.is_placeholder and sh.placeholder_format.type == 1:  # title
            sh.width = Inches(8.0)
            para = sh.text_frame.paragraphs[0]
            set_runs(para, text)
            para.alignment = PP_ALIGN.LEFT
            sh.left = Inches(1.1)
        elif sh.name.startswith("Объект"):
            sh._element.getparent().remove(sh._element)


def tb(slide, x, y, w, h, paras, size=16, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, bullet=False, space=6, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        runs = item if isinstance(item, list) else [(item, {})]
        if bullet:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.25)))
            pPr.set("indent", str(-Inches(0.22)))
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
            pPr.append(bu)
        for text, f in runs:
            r = para.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(f.get("size", size))
            r.font.bold = f.get("bold", bold)
            r.font.italic = f.get("italic", False)
            r.font.color.rgb = f.get("color", color)
    return box


def card(slide, x, y, w, h, fill=TINT, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def badge(slide, x, y, text, fill=BLUE, d=0.5, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    r = para.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def style_chart(chart, legend=True, size=12):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = GRAY
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va = chart.value_axis
    va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    chart.category_axis.tick_labels.font.size = Pt(size)
    chart.category_axis.format.line.color.rgb = LGRAY


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- 1. title slide
s = slides[0]
for sh in s.shapes:
    if sh.name == "Прямоугольник 3":
        ps = sh.text_frame.paragraphs
        set_runs(ps[0], "2027 IEEE 4th International Student Conference on Digital Generation (DG)")
        set_runs(ps[1], "February 24-26, 2027")
    elif sh.name == "Subtitle 2" and "Authors" in sh.text_frame.text:
        sh.width = Inches(9.2)
        ps = sh.text_frame.paragraphs
        set_runs(ps[0], "Authors’ names: Kanalbay Kassymkhan, Lira La")
        r = ps[1].runs
        r[0].text = "University/company: Astana IT University; L.N. Gumilyov Eurasian National University"
        r[1].text = "City, Country: Astana, Kazakhstan"
        # make the second line a separate visual line (template uses a line break between runs)
    elif sh.name == "Subtitle 2" and "Paper ID" in sh.text_frame.text:
        ps = sh.text_frame.paragraphs
        set_runs(ps[0], "Paper ID-XXX")
        set_runs(ps[1], "Section: Computational Intelligence")
    elif sh.is_placeholder and sh.placeholder_format.type in (1, 3) and sh.name == "Title 1":
        sh.top = Inches(1.95)
        sh.height = Inches(1.3)
        tf = sh.text_frame
        tf.word_wrap = True
        para = tf.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT
        r = para.add_run()
        r.text = ("Confidence-Aware Graph Neural Networks for Social Bot Detection: "
                  "Message-Passing Reweighting and Similarity-Graph Construction")
        r.font.size = Pt(26)
        r.font.bold = True
        r.font.name = FONT
        r.font.color.rgb = NAVY
notes(s, "Good morning. I present our study of confidence-aware graph neural networks for social bot detection, "
         "and what actually makes graph models work on tabular account data.")

# ---------------------------------------------------------------- 2. motivation
s = slides[1]
header(s, "Motivation: Bots, Graphs and a Blind Spot of GCN")
tb(s, 0.5, 1.05, 5.2, 0.5, "Why graph models for bot detection?", size=20, bold=True)
tb(s, 0.5, 1.6, 5.2, 2.6, [
    "Bots inflate metrics and spread disinformation; they act in coordinated clusters",
    "Per-account classifiers miss this structure; GNNs propagate signal along edges",
    "A GCN weights every neighbor by degree only: a borderline account counts as much as a confident one",
    "Hypothesis of our earlier work: down-weight low-confidence neighbors",
], size=16, bullet=True, color=GRAY, space=8)
card(s, 0.5, 4.6, 5.2, 1.7, name="Prior result card")
tb(s, 0.7, 4.7, 4.9, 0.4, "Earlier single-split result (DG 2026)", size=15, bold=True)
tb(s, 0.7, 5.15, 2.3, 1.0, [[("+3.0", {"size": 32, "bold": True, "color": BLUE})], [("precision points", {"size": 13, "color": GRAY})]])
tb(s, 3.1, 5.15, 2.5, 1.0, [[("−7.6", {"size": 32, "bold": True, "color": ORANGE})], [("recall points", {"size": 13, "color": GRAY})]])
s.shapes.add_picture("fig1_synthetic_graph.png", Inches(6.0), Inches(1.15), width=Inches(3.6))
tb(s, 6.0, 4.15, 3.6, 0.9, "Synthetic social graph: humans (green) and two dense bot clusters (red)", size=12, color=GRAY)
tb(s, 6.0, 5.0, 3.6, 1.3, [[("Question of this paper: ", {"bold": True, "color": NAVY}),
                            ("is that gain real, and what really helps?", {"color": NAVY})]], size=16)
notes(s, "Bots are coordinated, so graph models are attractive. A GCN treats confident and borderline neighbours equally. "
         "Last year we proposed to reweight by confidence and saw a precision gain on one split. Today we test it properly.")

# ---------------------------------------------------------------- 3. method
s = slides[2]
header(s, "Method: Confidence-Aware Message Passing")
badge(s, 0.5, 1.1, "1")
tb(s, 1.1, 1.08, 8.3, 0.5, "Conf-GCN: weight neighbors by the model's own confidence", size=17, bold=True)
s.shapes.add_picture("eq_conf.png", Inches(1.1), Inches(1.65), height=Inches(0.62))
tb(s, 1.1, 2.35, 8.3, 0.7, "No extra parameters, no gradient through confidence, no labels used. For two classes c ∈ [0.5, 1], so weights differ at most 2×.",
   size=14, color=GRAY)
badge(s, 0.5, 3.15, "2")
tb(s, 1.1, 3.13, 8.3, 0.5, "AC-GCN / AC-SAGE: learnable exponent on the baseline weights", size=17, bold=True)
s.shapes.add_picture("eq_ac.png", Inches(1.1), Inches(3.7), height=Inches(0.72))
tb(s, 1.1, 4.45, 8.3, 0.5, "α = 0 gives exactly GCN or GraphSAGE; large α removes the 2× bound.", size=14, color=GRAY)
badge(s, 0.5, 5.1, "3")
tb(s, 1.1, 5.08, 8.3, 1.4, [
    [("Agreement weights: ", {"bold": True}), ("edge weight = probability both endpoints share a class (can switch edges off)", {})],
    [("Self-training: ", {"bold": True}), ("unlabeled nodes with confidence ≥ 0.9 become pseudo-labels", {})],
], size=15, color=GRAY, space=6)
notes(s, "Three ways to use confidence: as a neighbour weight, as a learnable redistribution of the baseline's weights, "
         "and as a filter for pseudo-labels.")

# ---------------------------------------------------------------- 4. research questions
s = slides[3]
header(s, "Research Questions")
rqs = [("RQ1", "Is the single-split precision gain real, and is it caused by the confidence signal?"),
       ("RQ2", "Do learnable exponents, agreement weights or self-training help?"),
       ("RQ3", "How much does the construction of the similarity graph matter?"),
       ("RQ4", "What changes when only a few hundred accounts are labeled?")]
for i, (k, q) in enumerate(rqs):
    x = 0.5 + (i % 2) * 4.55
    y = 1.2 + (i // 2) * 2.6
    card(s, x, y, 4.3, 2.3, name=f"{k} card")
    badge(s, x + 0.25, y + 0.3, k, d=0.85, size=15)
    tb(s, x + 1.25, y + 0.25, 2.9, 1.9, q, size=17, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Four questions structure the talk.")

# ---------------------------------------------------------------- 5. setup
s = slides[4]
header(s, "Experimental Setup")
cols = [("Data", ["Twitter human/bot dataset (Botometer + manual labels)", "Balanced subsample: 5000 accounts",
                  "Features: SBERT description (384) + 6 numeric + 4 binary"]),
        ("Three k-NN graphs (k = 10)", ["Description graph: homophily 0.62", "Metadata graph: homophily 0.78",
                                       "Feature graph (both): homophily 0.76"]),
        ("Protocol", ["5-fold CV × 5 seeds = 25 runs per model", "Paired Wilcoxon + Holm correction",
                      "Baselines: MLP, GCN, GraphSAGE, GAT", "Ablations, threshold-matched precision"])]
for i, (title, items) in enumerate(cols):
    x = 0.5 + i * 3.05
    card(s, x, 1.15, 2.85, 4.2, name=f"{title} card")
    tb(s, x + 0.15, 1.3, 2.6, 0.8, title, size=18, bold=True, color=BLUE)
    tb(s, x + 0.15, 2.15, 2.6, 3.1, items, size=15, color=GRAY, bullet=True, space=10)
tb(s, 0.5, 5.6, 9.1, 0.8, [[("All models: ", {"bold": True, "color": NAVY}), ("3 layers, hidden size 64, dropout 0.5, 100 epochs, identical splits; hyper-parameters selected on validation folds only", {})]], size=15, color=GRAY)
notes(s, "All models share splits and capacity: three layers, hidden size 64. Twenty-five runs per model let us test significance.")

# ---------------------------------------------------------------- 6. RQ1 results
s = slides[5]
header(s, "RQ1: The Precision Gain Does Not Replicate")
cd = CategoryChartData()
cd.categories = ["MLP", "GCN", "GraphSAGE", "GAT", "Conf-GCN"]
cd.add_series("Precision", (0.805, 0.731, 0.805, 0.744, 0.732))
cd.add_series("F1", (0.807, 0.740, 0.803, 0.734, 0.731))
cd.add_series("AUROC", (0.890, 0.819, 0.887, 0.819, 0.794))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.0), Inches(6.0), Inches(5.5), cd)
gf.name = "RQ1 chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Description graph, mean of 25 runs"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
ch.value_axis.minimum_scale = 0.6
ch.value_axis.maximum_scale = 0.95
ch.value_axis.tick_labels.number_format = "0.00"
ch.value_axis.tick_labels.number_format_is_linked = False
for ser, col in zip(ch.series, (SKY, BLUE, NAVY)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
ch.plots[0].gap_width = 60
tb(s, 6.6, 1.1, 3.0, 0.9, [[("0.732 vs 0.731", {"size": 26, "bold": True, "color": BLUE})],
                           [("precision, Conf-GCN vs GCN (p = 0.83)", {"size": 13, "color": GRAY})]])
tb(s, 6.6, 2.35, 3.0, 0.9, [[("−2.5", {"size": 26, "bold": True, "color": ORANGE})],
                           [("AUROC points vs GCN", {"size": 13, "color": GRAY})]])
tb(s, 6.6, 3.55, 3.0, 2.9, [
    "Earlier gain was split noise (run-to-run std ≈ 0.02)",
    "At matched recall a tuned GCN is 0.9 points more precise",
    "Feature-only MLP beats every GCN-type model on this graph",
], size=14, color=GRAY, bullet=True, space=8)
notes(s, "With 25 runs, Conf-GCN has the same precision as GCN and loses AUROC. And the plain MLP is the best model on the description graph.")

# ---------------------------------------------------------------- 7. ablations
s = slides[6]
header(s, "RQ1: Normalization, Not Confidence, Drives the Effect")
rows = [("Variant", "F1", "AUROC"),
        ("Conf-GCN (full)", "0.731", "0.794"),
        ("Uniform confidence (c ≡ 1)", "0.736", "0.795"),
        ("Shuffled confidence", "0.735", "0.795"),
        ("Inverted weighting", "0.733", "0.797"),
        ("α = 2 / α = 4", "0.728 / 0.727", "0.793 / 0.788"),
        ("Uniform + self-loop", "0.742", "0.817"),
        ("GCN (reference)", "0.740", "0.819")]
gt = s.shapes.add_table(len(rows), 3, Inches(0.5), Inches(1.15), Inches(5.6), Inches(4.6))
gt.name = "Ablation table"
t = gt.table
t.columns[0].width = Inches(3.0)
t.columns[1].width = Inches(1.3)
t.columns[2].width = Inches(1.3)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = t.cell(i, j)
        c.text = val
        para = c.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = para.runs[0]
        r.font.name = FONT
        r.font.size = Pt(15)
        r.font.bold = (i == 0) or (i == 6)
        r.font.color.rgb = WHITE if i == 0 else NAVY
        c.fill.solid()
        c.fill.fore_color.rgb = BLUE if i == 0 else (RGBColor(0xFD, 0xE9, 0xD9) if i == 6 else (TINT if i % 2 else WHITE))
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
tb(s, 0.5, 5.9, 5.6, 0.6, "Description graph, 25 runs. Highlighted: adding a self-loop recovers GCN performance.", size=12, color=GRAY)
card(s, 6.4, 1.15, 3.2, 2.3, name="Weight spread card")
tb(s, 6.55, 1.25, 2.9, 2.1, [[("6.5%", {"size": 40, "bold": True, "color": BLUE})],
                             [("average deviation of confidence weights from uniform aggregation", {"size": 14, "color": GRAY})]])
tb(s, 6.4, 3.7, 3.2, 2.8, [
    "Removing, shuffling or inverting confidence changes nothing",
    "The AUROC gap came from the missing self-loop",
], size=15, color=GRAY, bullet=True, space=10)
notes(s, "If confidence mattered, shuffling or inverting it should hurt. It does not. The weights stay within a few percent of uniform.")

# ---------------------------------------------------------------- 8. graph construction
s = slides[7]
header(s, "RQ3: Graph Construction Is the Decisive Choice")
cd = CategoryChartData()
cd.categories = ["GCN", "GraphSAGE", "GAT", "Conf-GCN"]
cd.add_series("Description graph (h = 0.62)", (0.740, 0.803, 0.734, 0.731))
cd.add_series("Feature graph (h = 0.76)", (0.820, 0.825, 0.814, 0.818))
cd.add_series("Metadata graph (h = 0.78)", (0.826, 0.840, 0.823, 0.824))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.0), Inches(6.3), Inches(5.5), cd)
gf.name = "Graph construction chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Test F1 by graph (25 runs); MLP without graph = 0.807"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
ch.value_axis.minimum_scale = 0.70
ch.value_axis.maximum_scale = 0.86
ch.value_axis.tick_labels.number_format = "0.00"
ch.value_axis.tick_labels.number_format_is_linked = False
for ser, col in zip(ch.series, (LGRAY, SKY, BLUE)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
ch.plots[0].gap_width = 60
tb(s, 6.9, 1.1, 2.8, 5.3, [
    [("Description graph: ", {"bold": True, "color": NAVY}), ("every GCN-type model is worse than the MLP", {})],
    [("Metadata graph: ", {"bold": True, "color": NAVY}), ("every GNN beats the MLP (p < 2·10⁻⁵)", {})],
    [("Density matters little: ", {"bold": True, "color": NAVY}), ("k = 5, 10, 20 changes F1 by < 1 point", {})],
], size=15, color=GRAY, bullet=True, space=12)
notes(s, "Changing only the graph reverses the conclusion. Homophily rises from 62 to 78 percent, and all graph models overtake the MLP.")

# ---------------------------------------------------------------- 9. best detector
s = slides[8]
header(s, "Best Detector: GraphSAGE on the Metadata Graph")
stats = [("0.840", "F1 score", BLUE), ("0.921", "AUROC", BLUE), ("+9.9", "F1 vs original GCN pipeline", ORANGE),
         ("+3.2", "F1 vs feature-only MLP", ORANGE)]
for i, (v, lab, col) in enumerate(stats):
    x = 0.5 + i * 2.3
    card(s, x, 1.3, 2.1, 2.4, name=f"Stat {lab}")
    tb(s, x + 0.1, 1.5, 1.9, 1.0, v, size=40, bold=True, color=col, align=PP_ALIGN.CENTER)
    tb(s, x + 0.1, 2.55, 1.9, 1.0, lab, size=14, color=GRAY, align=PP_ALIGN.CENTER)
tb(s, 0.5, 4.1, 9.0, 0.5, "Why it works", size=20, bold=True)
tb(s, 0.5, 4.65, 9.0, 1.9, [
    "Metadata neighbors share the class far more often (homophily 0.78 vs 0.62)",
    "GraphSAGE keeps a node's own features next to the aggregated neighborhood",
    "Precision and recall rise together: 0.842 / 0.838 vs 0.731 / 0.752 (p < 10⁻⁶)",
], size=16, color=GRAY, bullet=True, space=8)
notes(s, "The final recipe gives 0.84 F1 and 0.92 AUROC, almost ten points above our original pipeline.")

# ---------------------------------------------------------------- 10. extensions
s = slides[9]
header(s, "RQ2: Adaptive Confidence Models")
cd = CategoryChartData()
cd.categories = ["Layer 1", "Layer 2", "Layer 3"]
cd.add_series("AC-GCN", (0.69, 0.69, 0.61))
cd.add_series("AC-SAGE", (0.78, 0.77, 0.76))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.0), Inches(4.8), Inches(4.6), cd)
gf.name = "Learned exponent chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Learned exponent α (initialized at 1.0)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = 1.0
ch.value_axis.tick_labels.number_format = "0.0"
ch.value_axis.tick_labels.number_format_is_linked = False
pl = ch.plots[0]
pl.has_data_labels = True
pl.data_labels.number_format = "0.00"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
for ser, col in zip(ch.series, (SKY, BLUE)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
tb(s, 0.5, 5.75, 4.7, 0.7, "Given freedom to amplify confidence, the models reduce it", size=14, color=GRAY)
tb(s, 5.5, 1.1, 4.1, 5.4, [
    [("AC-GCN fixes the AUROC loss: ", {"bold": True, "color": NAVY}), ("0.819 vs 0.794 for Conf-GCN (= GCN)", {})],
    [("No significant gain: ", {"bold": True, "color": NAVY}), ("AC-GCN, AC-SAGE and agreement weights stay within 0.5 F1 of their baselines", {})],
    [("Reason: ", {"bold": True, "color": NAVY}), ("confidence predicts whether a neighbor is correct only moderately (AUROC 0.66)", {})],
], size=15, color=GRAY, bullet=True, space=12)
notes(s, "When the exponent is learned, it shrinks from one to about 0.6 to 0.8. The models choose to rely less on confidence.")

# ---------------------------------------------------------------- 11. few labels
s = slides[10]
header(s, "RQ4: Few Labels, Confidence as a Pseudo-Label Filter")
cd = CategoryChartData()
cd.categories = ["100", "250", "500", "3000"]
cd.add_series("MLP", (0.744, 0.761, 0.773, 0.807))
cd.add_series("GCN", (0.727, 0.762, 0.790, 0.820))
cd.add_series("GraphSAGE", (0.755, 0.776, 0.788, 0.825))
cd.add_series("GraphSAGE + self-training", (0.764, 0.781, 0.795, 0.824))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.0), Inches(6.2), Inches(5.5), cd)
gf.name = "Few labels chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Test F1 vs labeled accounts (feature graph)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
ch.value_axis.minimum_scale = 0.70
ch.value_axis.maximum_scale = 0.84
ch.value_axis.tick_labels.number_format = "0.00"
ch.value_axis.tick_labels.number_format_is_linked = False
for ser, col, w in zip(ch.series, (LGRAY, SKY, BLUE, ORANGE), (2.0, 2.0, 2.0, 3.0)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(w)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
tb(s, 6.8, 1.1, 2.8, 1.2, [[("+2.0", {"size": 36, "bold": True, "color": ORANGE})],
                           [("F1 over MLP at 250 and 500 labels (p = 0.002)", {"size": 13, "color": GRAY})]])
tb(s, 6.8, 2.75, 2.8, 3.7, [
    "GraphSAGE + self-training is best at 100–500 labels",
    "Self-training adds 0.6 F1 to GraphSAGE at 250 labels (p = 0.014)",
    "The graph matters most when labels are scarce",
], size=14, color=GRAY, bullet=True, space=10)
notes(s, "Moderation teams rarely have thousands of labels. With a few hundred, the graph plus confidence-based self-training wins by two points.")

# ---------------------------------------------------------------- 12. conclusion
s = slides[11]
header(s, "Conclusions and Future Work")
items = [("1", "Conf-GCN's single-split precision gain was sampling noise; confidence weights stay near uniform"),
         ("2", "Graph construction is decisive: a metadata k-NN graph makes every GNN beat the MLP"),
         ("3", "Recipe: metadata graph + GraphSAGE → F1 0.840, AUROC 0.921 (+9.9 F1)"),
         ("4", "Few labels: confidence-based self-training gives the best detector")]
for i, (k, text) in enumerate(items):
    y = 1.15 + i * 0.95
    badge(s, 0.5, y, k)
    tb(s, 1.15, y - 0.02, 8.4, 0.85, text, size=17, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
card(s, 0.5, 5.05, 9.1, 1.45, name="Future work card")
tb(s, 0.7, 5.15, 8.7, 0.45, "Future work", size=17, bold=True, color=BLUE)
tb(s, 0.7, 5.6, 8.7, 0.85, "Real follow/mention graphs (TwiBot-20), calibrated confidence, and learning the graph jointly with the classifier",
   size=15, color=GRAY)
notes(s, "To summarise: build the right graph, keep the node's own features, and use confidence for pseudo-labels rather than for weighting.")

# ---------------------------------------------------------------- 13. thank you
s = slides[12]
header(s, "Thank You")
tb(s, 0.5, 2.0, 9.0, 1.0, "Thank you for your attention!", size=40, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
tb(s, 0.5, 3.1, 9.0, 0.6, "Questions are welcome", size=22, color=BLUE, align=PP_ALIGN.CENTER)
tb(s, 0.5, 4.2, 9.0, 1.5, [
    "Kanalbay Kassymkhan — solodancer235@gmail.com",
    "Lira La — Lira.La@astanait.edu.kz",
    "Code and results: github.com/solodancer235-star/confidence-aware-gcn-bot-detection",
], size=15, color=GRAY, align=PP_ALIGN.CENTER, space=4)
notes(s, "Thank you. I am happy to take questions.")

prs.save("DG2027_presentation.pptx")
print("saved")
