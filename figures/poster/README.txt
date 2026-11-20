LEAF workshop poster

Print file: LEAF_workshop_poster_61x85cm.pdf
PowerPoint: LEAF_workshop_poster_61x85cm.pptx
Size: 610 mm wide x 850 mm high, portrait, one page.
Print at 100% / actual size. PDF text and charts are vector with embedded Arial fonts.
Editable artwork: LEAF_workshop_poster_editable.svg (open in Inkscape or Illustrator).
Preview: LEAF_workshop_poster_preview.png (screen review, not the print master).
Rebuild current figure-based version: python figures/poster/build_poster_with_figures.py
Dependencies: reportlab, pymupdf, python-pptx; Windows Arial font files.
build_poster.py builds the earlier text-and-chart version; it will overwrite current
PDF/SVG/preview files, so use build_poster_with_figures.py for the current version.

The current poster uses original paper figure PDFs:
- Figure1_architecture.pdf: full paper Figure 1.
- fig2_styled_panel.pdf: selected panels B (performance) and C (proposal duplicates).
- fig5_lipid_structures.pdf: structure panel corresponding to paper Figure 5A.
- fig4_evidence_panel.pdf: selected panel B (reasoning engagement).
Selected panels are cropped at panel boundaries without changing plotted data.
PDF figures remain vector. PowerPoint uses 220-dpi images at their placed sizes;
all surrounding text and layout shapes are editable. The slide notes list sources.

Content source: C:\Users\hs494\Downloads\LEAF_biorxiv.pdf, read directly.
Primary comparisons: paper Table 1, random split, n=8.
Charged-query duplicate rates: Section 3.3 (not proposal-pool duplicate rates).
Strict novelty intervention: Table 3; confidence interval in Section 3.3.
Reasoning-mode and score-shuffle checks: Section 3.4, exploratory n=4.
Validated HeLa-only endpoint: Section 3.1.
Author affiliation markers follow the PDF author list (1 and 2).

Workshop service page inspected on 13 November 2026:
https://www.postersessiononline.eu/pr/poster2.asp?congreso=469631373
The page identifies the NeurIPS 2026 Sydney workshop service. Its template link
returned the same service page, so canvas dimensions follow the user's explicit
61 x 85 cm requirement. No poster was submitted or print order placed.
