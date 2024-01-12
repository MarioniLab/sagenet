import sagenet as sg
from sagenet.utils import map2ref, glasso
import numpy as np
import squidpy as sq
import scanpy as sc
import torch
import anndata as ad

random.seed(1996)


adata_r1 = sg.MGA_data.seqFISH1_1()
adata_r2 = sg.MGA_data.seqFISH2_1()
adata_r3 = sg.MGA_data.seqFISH3_1()
adata_q1 = sg.MGA_data.seqFISH1_2()
adata_q2 = sg.MGA_data.seqFISH2_2()
adata_q3 = sg.MGA_data.seqFISH3_2()
adata_q  = sg.MGA_data.scRNAseq()


# Map everything to 1-1
glasso(adata_r1)
adata_r1.obsm['spatial'] = np.array(adata_r1.obs[['x','y']])
sq.gr.spatial_neighbors(adata_r1, coord_type="generic")
sc.tl.leiden(adata_r1, resolution=.01, random_state=0, key_added='leiden_0.01', adjacency=adata_r1.obsp["spatial_connectivities"])
sc.tl.leiden(adata_r1, resolution=.05, random_state=0, key_added='leiden_0.05', adjacency=adata_r1.obsp["spatial_connectivities"])
sc.tl.leiden(adata_r1, resolution=.1, random_state=0, key_added='leiden_0.1', adjacency=adata_r1.obsp["spatial_connectivities"])
sc.tl.leiden(adata_r1, resolution=.5, random_state=0, key_added='leiden_0.5', adjacency=adata_r1.obsp["spatial_connectivities"])
sc.tl.leiden(adata_r1, resolution=1, random_state=0, key_added='leiden_1', adjacency=adata_r1.obsp["spatial_connectivities"])

if torch.cuda.is_available():  
  dev = "cuda:0"  
else:  
  dev = "cpu"  
  
  
device = torch.device(dev)
print(device)

sg_obj = sg.sage.sage(device=device)
sg_obj.add_ref(adata_r1, comm_columns=['leiden_0.01', 'leiden_0.05', 'leiden_0.1', 'leiden_0.5', 'leiden_1'], tag='embryo1_2', epochs=20, verbose = True, classifier='GraphSAGE')


sg_obj.map_query(adata_r1, save_prob=True)
ind, conf = map2ref(adata_r1, adata_r1)
adata_r1.obsm['spatial_pred'] = adata_r1.obsm['spatial'][ind,:]
adata_r1.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_r2, save_prob=True)
ind, conf = map2ref(adata_r1, adata_r2)
adata_r2.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_r2.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_r3, save_prob=True)
ind, conf = map2ref(adata_r1, adata_r3)
adata_r3.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_r3.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_q1, save_prob=True)
ind, conf = map2ref(adata_r1, adata_q1)
adata_q1.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_q1.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_q2, save_prob=True)
ind, conf = map2ref(adata_r1, adata_q2)
adata_q2.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_q2.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_q3, save_prob=True)
ind, conf = map2ref(adata_r1, adata_q3)
adata_q3.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_q3.obs['conf'] = np.log(conf)
sg_obj.map_query(adata_q, save_prob=True)
ind, conf = map2ref(adata_r1, adata_q)
adata_q.obsm['spatial'] = adata_r1.obsm['spatial'][ind,:]
adata_q.obs['conf'] = np.log(conf)

adata_r1.obsm['spatial'] = adata_r1.obsm['spatial_pred']


ad_concat = ad.concat([adata_r1, adata_r2, adata_r3, adata_q1, adata_q2, adata_q3, adata_q], label='batch')

sc.pl.spatial(
    ad_concat,
    color='cell_type',
    palette=celltype_colours,# Color cells based on 'cell_type'
    # color_map=cell_type_color_map,  # Use the custom color map
    # library_id='r1_mapping',  # Use 'r1_mapping' coordinates
    title='all to r1 map',
    save='_ad_r1_all.pdf',
    spot_size=.1
)

sc.pl.spatial(
    ad_concat,
    color='conf',
    # palette=celltype_colours,# Color cells based on 'cell_type'
    # color_map=cell_type_color_map,  # Use the custom color map
    # library_id='r1_mapping',  # Use 'r1_mapping' coordinates
    title='all to r1 map',
    save='_ad_r1_all_conf.pdf',
    spot_size=.1
)