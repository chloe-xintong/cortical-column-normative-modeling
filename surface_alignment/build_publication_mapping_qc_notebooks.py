"""Build publication-style single-panel fsaverage QC notebooks."""
import json
from pathlib import Path

STYLE = """import matplotlib as mpl\nimport matplotlib.pyplot as plt\nfrom matplotlib import font_manager\nfrom IPython.display import display\nfrom pathlib import Path\n_available_fonts={f.name for f in font_manager.fontManager.ttflist}\n_selected_font=('Arial' if 'Arial' in _available_fonts else 'Liberation Sans' if 'Liberation Sans' in _available_fonts else 'DejaVu Sans')\nprint('Using font:', _selected_font)\nmpl.rcParams.update({'font.family':_selected_font,'font.size':7,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.8,'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,'figure.facecolor':'white','savefig.facecolor':'white','savefig.dpi':1200,'pdf.fonttype':42,'svg.fonttype':'none'})\nOKABE={'blue':'#0072B2','orange':'#E69F00','purple':'#CC79A7'}\nWIDTH,HEIGHT=3.35,2.8\ndef save_figure(fig,stem):\n    png=QC_ROOT/f'{stem}.png'; pdf=QC_ROOT/f'{stem}.pdf'; svg=QC_ROOT/f'{stem}.svg'\n    fig.savefig(png,dpi=1200,bbox_inches='tight',pad_inches=0.03,facecolor='white')\n    fig.savefig(pdf,bbox_inches='tight',pad_inches=0.03,facecolor='white')\n    fig.savefig(svg,bbox_inches='tight',pad_inches=0.03,facecolor='white')\n    display(fig)\n    print(f'Saved: {png} ({png.stat().st_size/1024:.0f} KB)')\n    plt.close(fig)\n"""

def notebook(template):
    root=f'/mnt/newStor/paros/paros_WORK/xintong/results/{template}_mapping_qc'
    def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[s]}
    def md(s): return {'cell_type':'markdown','metadata':{},'source':[s]}
    load=STYLE+f"\nQC_ROOT=Path('{root}')\nregion_qc=pd.read_csv(QC_ROOT/'fsaverage_subject_region_mapping_qc.csv'); vertex_freq=pd.read_csv(QC_ROOT/'fsaverage_template_vertex_coverage_frequency.csv'); region_freq=pd.read_csv(QC_ROOT/'fsaverage_region_coverage_frequency_summary.csv'); failures=pd.read_csv(QC_ROOT/'fsaverage_mapping_failures.csv')\nprint(f'Subjects: {{region_qc.subject_id.nunique():,}}; rows: {{len(region_qc):,}}; failures: {{len(failures):,}}')\ncoverage_stats=region_qc.groupby('hemisphere').agg(coverage_pct=('regional_coverage_valid_pct','median'), theoretical_max_pct=('theoretical_max_coverage_after_cleaning_pct','median'), efficiency_pct=('coverage_efficiency_after_cleaning_pct','median')).reindex(['lh','rh'])\nprint('Median coverage statistics (%)')\ndisplay(coverage_stats.round(2))\nprint(f'Overall median coverage: {{region_qc.regional_coverage_valid_pct.median():.2f}}%')\nregion_accuracy_stats=region_qc.groupby(['hemisphere','region']).region_label_agreement_pct.agg(median='median',p25=lambda x:x.quantile(.25),p75=lambda x:x.quantile(.75)).reset_index().sort_values(['hemisphere','median'])\nprint('Region-label accuracy by cortical region (%)')\ndisplay(region_accuracy_stats.round(2))\n"
    cells=[md(f'# {template} native-column mapping QC\n\nPublication-style single-panel figures; source data are not modified.'),code("import numpy as np\nimport pandas as pd\n"+load),code("x=region_qc.mapping_distance_median_deg.dropna(); fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); ax.hist(x,bins=35,color=OKABE['blue'],edgecolor='white',linewidth=.4); ax.set_title('Mapping distance distribution',fontsize=8,pad=6); ax.set_xlabel('Median mapping distance (degrees)'); ax.set_ylabel('Subject–region count'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); save_figure(fig,'01_mapping_distance_distribution')"),code("x=region_qc.region_label_agreement_pct.dropna(); fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); ax.hist(x,bins=35,color=OKABE['purple'],edgecolor='white',linewidth=.4); ax.set_title('Target region-label agreement',fontsize=8,pad=6); ax.set_xlabel('Target region-label agreement (%)'); ax.set_ylabel('Subject–region count'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); save_figure(fig,'02_region_label_agreement_distribution')"),code("for hemi in ('lh','rh'):\n    plot=region_qc[region_qc.hemisphere.eq(hemi)].pivot(index='subject_id',columns='region',values='regional_coverage_valid_pct'); plot=plot.loc[plot.median(axis=1).sort_values().index,plot.median(axis=0).sort_values().index]; fig,ax=plt.subplots(figsize=(WIDTH,3.2)); im=ax.imshow(plot.to_numpy(),aspect='auto',cmap='viridis',vmin=0,vmax=100); ax.set_title(f'Subject × region coverage ({hemi})',fontsize=8,pad=6); ax.set_ylabel('Subject'); ax.set_xlabel(''); ax.set_xticks(range(len(plot.columns))); ax.set_xticklabels(plot.columns,rotation=90,fontsize=4.5); ax.set_yticks([]); cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('Coverage (%)',fontsize=8); cb.ax.tick_params(labelsize=7); ax.spines[:].set_visible(False); fig.subplots_adjust(bottom=.30,top=.88); save_figure(fig,f'03_subject_region_coverage_{hemi}')\n    ") ,code("for hemi in ('lh','rh'):\n    plot=region_qc[region_qc.hemisphere.eq(hemi)].pivot(index='subject_id',columns='region',values='region_label_agreement_pct'); plot=plot.loc[plot.median(axis=1).sort_values().index,plot.median(axis=0).sort_values().index]; fig,ax=plt.subplots(figsize=(WIDTH,3.2)); im=ax.imshow(plot.to_numpy(),aspect='auto',cmap='viridis',vmin=0,vmax=100); ax.set_title(f'Subject × region label agreement ({hemi})',fontsize=8,pad=6); ax.set_ylabel('Subject'); ax.set_xlabel(''); ax.set_xticks(range(len(plot.columns))); ax.set_xticklabels(plot.columns,rotation=90,fontsize=4.5); ax.set_yticks([]); cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('Agreement (%)',fontsize=8); cb.ax.tick_params(labelsize=7); ax.spines[:].set_visible(False); fig.subplots_adjust(bottom=.30,top=.88); save_figure(fig,f'04_subject_region_label_agreement_{hemi}')\n    ")]
    cells.extend([
      md(''),
      code("fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); vals=[region_qc.collision_vertex_rate_pct.dropna().to_numpy(),region_qc.colliding_column_rate_pct.dropna().to_numpy()]; ax.boxplot(vals,tick_labels=['Collision vertices','Colliding columns'],showfliers=False,patch_artist=True,boxprops={'facecolor':'#D9EAF2','edgecolor':OKABE['blue'],'linewidth':.8},medianprops={'color':OKABE['orange'],'linewidth':1.2},whiskerprops={'color':OKABE['blue'],'linewidth':.8},capprops={'color':OKABE['blue'],'linewidth':.8}); rng=np.random.default_rng(0); [ax.scatter(np.full(len(v),i+1)+rng.uniform(-.08,.08,len(v)),v,s=2,alpha=.15,color=OKABE['blue'],linewidths=0) for i,v in enumerate(vals)]; ax.set_title('Within-region collision rates',fontsize=8,pad=6); ax.set_ylabel('Rate (%)'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); save_figure(fig,'05_collision_rates')"),
      code("""eff=region_qc.groupby('hemisphere',as_index=False).agg(theoretical=('theoretical_max_coverage_after_cleaning_pct','median'),actual=('regional_coverage_valid_pct','median'),efficiency=('coverage_efficiency_after_cleaning_pct','median')); labels=['Theoretical max.','Efficiency','Actual coverage']; x=np.arange(3); fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); width=.34; colors=[OKABE['blue'],OKABE['purple']];
for i,row in eff.iterrows():
    ax.scatter(x+(i-.5)*width,[row.theoretical,row.actual,row.efficiency],s=22,color=colors[i],label=row.hemisphere, zorder=3)
ax.set_title('Coverage ceiling and mapping efficiency',fontsize=8,pad=6); ax.set_ylabel('Percentage (%)'); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=6); ax.set_ylim(0,105); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); save_figure(fig,'06_coverage_ceiling_and_efficiency')""")
    ])
    cells.insert(2, code("""groups=[region_qc.loc[region_qc.hemisphere.eq(h),'region_label_agreement_pct'].dropna().to_numpy() for h in ('lh','rh')]
fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT))
parts=ax.violinplot(groups,positions=[1,2],showmeans=False,showmedians=False,showextrema=False)
for body,color in zip(parts['bodies'],[OKABE['blue'],OKABE['purple']]):
    body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.35)
for i,values in enumerate(groups,1):
    q1,med,q3=np.percentile(values,[25,50,75])
    ax.vlines(i,q1,q3,color=OKABE['blue'],linewidth=3)
    ax.scatter(i,med,color=OKABE['orange'],s=18,zorder=3)
ax.set_title('Overall region-label agreement',fontsize=8,pad=6)
ax.set_ylabel('Agreement (%)'); ax.set_xticks([1,2]); ax.set_xticklabels(['Left','Right'])
ax.set_ylim(0,100); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); save_figure(fig,'02b_overall_region_label_agreement')"""))
    cells.insert(3, code("""multiplicity_cols=['multiplicity_2_vertices','multiplicity_3_vertices','multiplicity_4_vertices','multiplicity_5_vertices','multiplicity_6plus_vertices']
if not set(multiplicity_cols).issubset(region_qc.columns):
    print('Conflict multiplicity 4/5/>=6 fields are not available in this QC table.')
else:
    multiplicity_labels=['2','3','4','5','>=6']
    multiplicity_counts=region_qc[multiplicity_cols].sum().astype(float)
    multiplicity_pct=100*multiplicity_counts/multiplicity_counts.sum()
    display(pd.DataFrame({'multiplicity':multiplicity_labels,'target_vertices':multiplicity_counts.astype(int).to_numpy(),'percentage':multiplicity_pct.to_numpy()}))
    fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT))
    bars=ax.bar(multiplicity_labels,multiplicity_pct.to_numpy(),color=[OKABE['blue'],OKABE['purple'],OKABE['orange'],'#56B4E9','#009E73'])
    ax.set_title('fsaverage6 conflict multiplicity',fontsize=8,pad=6)
    ax.set_xlabel('Native columns mapped to one target vertex'); ax.set_ylabel('Percentage of conflict vertices (%)'); ax.set_ylim(0, max(100, multiplicity_pct.max()*1.2))
    ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,multiplicity_pct.to_numpy()): ax.text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}%',ha='center',va='bottom',fontsize=7)
    fig.tight_layout(); save_figure(fig,'02c_conflict_multiplicity')"""))
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'cortical_trajectory_analysis','language':'python','name':'cortical_trajectory_analysis'}},'nbformat':4,'nbformat_minor':5}

for t in ('fsaverage','fsaverage6'):
    nb=notebook(t)
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell['source']=[s.replace("fig,ax=plt.subplots(figsize=(WIDTH,3.2))", "fig,ax=plt.subplots(figsize=(7.5,4.5))") for s in cell['source']]
            cell['source']=[s.replace("['Theoretical maximum','Actual coverage','Coverage efficiency']", "['Theoretical max.','Actual coverage','Efficiency']") for s in cell['source']]
            cell['source']=[s.replace("['Theoretical maximum coverage (%)','Actual coverage (%)','Coverage efficiency (%)']", "['Theoretical\\nmaximum','Actual\\ncoverage','Coverage\\nefficiency']") for s in cell['source']]
            cell['source']=[s.replace("[row.theoretical,row.actual,row.efficiency]", "[row.theoretical,row.efficiency,row.actual]") for s in cell['source']]
            cell['source']=[s.replace("['Theoretical maximum','Actual coverage','Coverage efficiency']", "['Theoretical max.','Efficiency','Actual coverage']") for s in cell['source']]
            cell['source']=[s.replace("fsaverage coverage and mapping efficiency", f"{t} coverage and mapping efficiency") for s in cell['source']]
            cell['source']=[s.replace("\n    display(fig)\n", "\n    mpl.rcParams['figure.dpi']=180\n    display(fig)\n") for s in cell['source']]
            if any("save_figure(fig,'05_collision_rates')" in s for s in cell['source']):
                cell['source']=[("fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); vals=[region_qc.collision_vertex_rate_pct.dropna(),region_qc.colliding_column_rate_pct.dropna()]; ax.boxplot(vals,tick_labels=['Collision vertices','Colliding columns'],showfliers=True,patch_artist=False,boxprops={'color':'#333333','linewidth':.8},medianprops={'color':OKABE['orange'],'linewidth':1.2},whiskerprops={'color':'#333333','linewidth':.8},capprops={'color':'#333333','linewidth':.8},flierprops={'marker':'o','markersize':2,'markerfacecolor':'none','markeredgecolor':'#111111','alpha':.65}); ax.set_title('Within-region collision rates',fontsize=8,pad=6); ax.set_ylabel('Rate (%)'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); save_figure(fig,'05_collision_rates')")]
            if any("save_figure(fig,'06_coverage_ceiling_and_efficiency')" in s for s in cell['source']):
                cell['source']=[("""overall_coverage=pd.DataFrame({'Theoretical maximum coverage (%)':[region_qc.theoretical_max_coverage_after_cleaning_pct.median()],'Actual coverage (%)':[region_qc.regional_coverage_valid_pct.median()],'Coverage efficiency (%)':[region_qc.coverage_efficiency_after_cleaning_pct.median()]})
fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); values=overall_coverage.iloc[0]; bars=ax.bar(np.arange(3),values.values,color=['#8ecae6','#219ebc','#52b788'],alpha=.9)
ax.set_title('fsaverage coverage and mapping efficiency',fontsize=8,pad=6); ax.set_ylabel('Percentage (%)'); ax.set_ylim(0,108); ax.set_xticks(np.arange(3)); ax.set_xticklabels(['Theoretical\\nmaximum','Actual\\ncoverage','Coverage\\nefficiency'],fontsize=6); ax.spines[['top','right']].set_visible(False)
for bar,value in zip(bars,values.values): ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}%',ha='center',va='bottom',fontsize=7)
fig.tight_layout(); save_figure(fig,'06_coverage_ceiling_and_efficiency')""")]
                cell['source']=[("""values=np.array([region_qc.theoretical_max_coverage_after_cleaning_pct.median(),region_qc.regional_coverage_valid_pct.median(),region_qc.coverage_efficiency_after_cleaning_pct.median()])
fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT)); bars=ax.bar(np.arange(3),values,color=['#8ecae6','#52b788','#219ebc'],alpha=.9)
ax.set_title('fsaverage coverage and mapping efficiency',fontsize=8,pad=6); ax.set_ylabel('Percentage (%)'); ax.set_ylim(0,108); ax.set_xticks(np.arange(3)); ax.set_xticklabels(['Theoretical\\nmaximum','Coverage\\nefficiency','Actual\\ncoverage'],fontsize=6); ax.spines[['top','right']].set_visible(False)
for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}%',ha='center',va='bottom',fontsize=7)
fig.tight_layout(); save_figure(fig,'06_coverage_ceiling_and_efficiency')""")]
    Path(f'surface_alignment/{t}_mapping_qc.ipynb').write_text(json.dumps(nb,indent=1))
    print(f'wrote {t} notebook')
