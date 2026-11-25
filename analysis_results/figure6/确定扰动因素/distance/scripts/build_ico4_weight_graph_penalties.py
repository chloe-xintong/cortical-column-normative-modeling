#!/usr/bin/env python3
"""Build normalized ico4 spatial and 21-depth chain edge operators from original topology."""
import json
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[5]; AGE=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction'; FS=Path('/home/apps/freesurfer/subjects')
def main():
 nodes=pd.read_csv(AGE/'features/plateau_feature_index.csv').sort_values('node_order').reset_index(drop=True); edges=[]; hemi_reports={}
 for hemi in ('lh','rh'):
  _,faces=fsio.read_geometry(FS/'fsaverage4/surf'/f'{hemi}.sphere')
  part=nodes[nodes.hemisphere.eq(hemi)]; lookup=dict(zip(part.ico_node_id.astype(int),part.node_order.astype(int)))
  raw=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]])); raw=np.sort(raw,axis=1); raw=np.unique(raw,axis=0)
  keep=np.array([(int(a) in lookup and int(b) in lookup) for a,b in raw]); valid=raw[keep]
  mapped=np.array([(lookup[int(a)],lookup[int(b)]) for a,b in valid],dtype=np.int32); edges.append(mapped)
  hemi_reports[hemi]={'n_original_vertices':int(faces.max()+1),'n_original_faces':len(faces),'n_original_edges':len(raw),'n_valid_nodes':len(part),'n_retained_edges':len(mapped),'n_removed_edges_touching_mask_gap':int(len(raw)-len(mapped))}
 edges=np.vstack(edges); n=len(nodes); spatial_mean_diag=2*len(edges)/n; depth_mean_diag=2*20/21
 np.savez_compressed(AGE/'features/ico4_weight_graph_edges.npz',spatial_u=edges[:,0],spatial_v=edges[:,1],n_nodes=np.int32(n),n_depths=np.int32(21),spatial_B_scale=np.float64(1/np.sqrt(spatial_mean_diag)),depth_B_scale=np.float64(1/np.sqrt(depth_mean_diag)))
 report={'status':'complete','topology_source':str((FS/'fsaverage4/surf').resolve()),'topology_rule':'original ico4 faces; unique undirected edges; retain only edges whose two endpoints are in formal analysis mask; no retriangulation','hemispheres':hemi_reports,'n_bilateral_nodes':n,'n_bilateral_spatial_edges':len(edges),'spatial_penalty_mean_diagonal_before_normalization':spatial_mean_diag,'spatial_penalty_mean_diagonal_after_normalization':1.0,'depth_edges_per_column':20,'depth_chain_wraparound':False,'depth_penalty_mean_diagonal_before_normalization':depth_mean_diag,'depth_penalty_mean_diagonal_after_normalization':1.0,'feature_order':'node_order major, depth 0..20 minor'}
 (AGE/'features/ico4_weight_graph_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
