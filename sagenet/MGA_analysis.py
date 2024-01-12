import sagenet as sg
from sagenet.utils import map2ref, glasso
import numpy as np
import squidpy as sq
import scanpy as sc
import torch

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
ind, conf = map2ref(adata_r1, adata_r1, key='spatial')
adata_r1.obsm['spatial_pred'] = adata_r1.obsm['spatial'][ind,:]
map2ref(adata_r1, adata_r2, key='spatial')
map2ref(adata_r1, adata_r3, key='spatial')
map2ref(adata_r1, adata_r1, key='spatial')
sg_obj.map_query(adata_r1, save_prob=True)
sg_obj.map_query(adata_r1, save_prob=True)
sg_obj.map_query(adata_r1, save_prob=True)
sg_obj.map_query(adata_r1, save_prob=True)
sg_obj.map_query(adata_r1, save_prob=True)
sg_obj.map_query(adata_r1, save_prob=True)

