# ADDcode preprocessing

This workflow is for the S-series ADDecode DWI data without reverse-phase images. It does not run topup or eddy. It uses the existing ADD preprocessing steps and the common tensor-fitting stage.

Checkpoints are validated after preprocessing, before tensor fitting, and after metric generation.
