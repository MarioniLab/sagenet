SageNet: spatial reconstruction of dissociated single-cell data using graph neural networks
=========================================================================
.. raw:: html

**SageNet** is a robust and generalizable graph neural network approach that probabilistically maps a dissociated scRNAseq dataset to their hypothetical tissue of origin using one or more reference datasets aquired by spatially resolved transcriptomics techniques. It is compatible with both high-plex imaging (e.g., seqFISH, MERFISH, etc.) and spatial barcoding (e.g., 10X visium, Slide-seq, etc.) datasets as the spatial reference. 


.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://github.com/MarioniLab/sagenet/files/7663030/final.pdf"
             width="700px" alt="sagenet logo">
        </a>
    </p>

SageNet is implemented with `pytorch <https://pytorch.org/docs/stable/index.html>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ to be modular, fast, and scalable. Also, it uses ``anndata`` to be compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and `squidpy <https://squidpy.readthedocs.io/en/stable/>`_ for pre- and post-processing steps.

Installation
-------------------------------
You can get the latest development version of our toolkit from `Github <https://github.com/MarioniLab/sagenet>`_ using the following steps:

First, clone the repository using ``git``::

    git clone https://github.com/MarioniLab/sagenet

Then, ``cd`` to the sagenet folder and run the install command::

    cd sagenet
    python setup.py install #or pip install ` 


The dependency ``torch-geometric`` should be installed separately, corresponding the system specefities, look at `this link <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ for instructions. 


.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://user-images.githubusercontent.com/55977725/144909791-7b451f94-bcf4-4f2d-9f7e-6c1a692e6ffd.gif"
             width="400px" alt="activations logo">
        </a>
    </p>



Usage
-------------------------------
::

	import sagenet as sg
	import scanpy as sc
	import squidpy as sq
	import anndata as ad
	import random
	random.seed(10)
	

**#. Training phase:**

**Input:=**
- Expression matrix associated with the (spatial) reference dataset (an ``anndata`` object)

::

	adata_r = sg.datasets.seqFISH()


- gene-gene interaction network
		

::

	glasso(adata_r, [0.5, 0.75, 1])




- one or more partitionings of the spatial reference into distinct connected neighborhoods of cells or spots

::

	adata_r.obsm['spatial'] = np.array(adata_r.obs[['x','y']])
	sq.gr.spatial_neighbors(adata_r, coord_type="generic")
	sc.tl.leiden(adata_r, resolution=.01, random_state=0, key_added='leiden_0.01', adjacency=adata_r.obsp["spatial_connectivities"])
	sc.tl.leiden(adata_r, resolution=.05, random_state=0, key_added='leiden_0.05', adjacency=adata_r.obsp["spatial_connectivities"])
	sc.tl.leiden(adata_r, resolution=.1, random_state=0, key_added='leiden_0.1', adjacency=adata_r.obsp["spatial_connectivities"])
	sc.tl.leiden(adata_r, resolution=.5, random_state=0, key_added='leiden_0.5', adjacency=adata_r.obsp["spatial_connectivities"])
	sc.tl.leiden(adata_r, resolution=1, random_state=0, key_added='leiden_1', adjacency=adata_r.obsp["spatial_connectivities"])



**Training:** 
::


	sg_obj = sg.sage.sage(device=device)
	sg_obj.add_ref(adata_r, comm_columns=['leiden_0.01', 'leiden_0.05', 'leiden_0.1', 'leiden_0.5', 'leiden_1'], tag='seqFISH_ref', epochs=20, verbose = False)


	
**Output:**
- A set of pre-trained models (one for each partitioning)

::


	!mkdir models
	!mkdir models/seqFISH_ref
	sg_obj.save_model_as_folder('models/seqFISH_ref')	


- A concensus scoring of spatially informativity of each gene

::


	ind = np.argsort(-adata_r.var['seqFISH_ref_entropy'])[0:12]
	with rc_context({'figure.figsize': (4, 4)}):
		sc.pl.spatial(adata_r, color=list(adata_r.var_names[ind]), ncols=4, spot_size=0.03, legend_loc=None)




#. Mapping phase:

**Input:**

- Expression matrix associated with the (dissociated) query dataset (an ``anndata`` object)
::
	
	adata_q = sg.datasets.MGA()


**Mapping:**
::

	sg_obj.map_query(adata_q)


**Output:**

- The reconstructed cell-cell spatial distance matrix 
::

	adata_q.obsm['dist_map']


- A concensus scoring of mapability (uncertainity of mapping) of each cell to the references
::

	adata_q.obs
		

.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://github.com/MarioniLab/sagenet/files/7687712/umapeli-11.pdf"
             width="900px" alt="umap">
        </a>
    </p>
		

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/MarioniLab/sagenet/issues/new>`__ or reach us by `email <mailto:eheidari@student.ethz.ch>`_.


