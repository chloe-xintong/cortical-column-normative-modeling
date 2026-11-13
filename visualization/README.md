# Cortical-column method visualization

The scripts in this folder are read-only consumers of the existing column and
profile outputs. They do not modify the analysis pipeline. Coordinates in the
column NPZ and exported FreeSurfer surfaces are native FreeSurfer surface RAS
(tkReg RAS); load them with the same subject's `T1.mgz`, `white`, and `pial`.

`generate_method_figures.py` writes three independent figures plus the combined
mapping-to-column panel in SVG, PDF, and 300-dpi PNG. It requires a real column
NPZ and a matching real MD-profile NPZ and refuses invalid/non-21-point data.

`export_freeview_columns.py` writes tube surfaces and MD morph overlays for one
or a few template vertices, a separate colorbar, metadata, and directly runnable
Freeview commands. The main vertex should be the same vertex used for the profile
figure. Freeview option spelling can differ between releases; the generated
surface and morph files are standard FreeSurfer files even if a local Freeview
build requires adjusting an overlay display flag.
