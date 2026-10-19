# Option 1 supplementary QC

These notebooks summarize supplementary native-surface analyses that were
developed before the template-driven column pipeline became the production
method. They are retained as QC and supporting evidence, not as production
profile-generation code.

## Retained notebooks

- `boundary_disagreement_qc.ipynb`: checks whether label disagreement is
  concentrated near cortical region boundaries or occurs within regions.
- `collision_profile_qc.ipynb`: compares native columns that map to the same
  fsaverage6 vertex, including profile similarity, native spatial span,
  multiplicity, and label mixing.

The notebooks read previously generated external result tables. The former
calculation scripts and Slurm submission files were removed from this branch;
the archived method-comparison branch preserves them if rerunning the legacy
analyses becomes necessary.
