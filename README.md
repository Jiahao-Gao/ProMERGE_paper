This repo contains the scripts to reproduce the analysis in the manuscript "Learning protein representations across contexts and modalities with ProMERGE".  

0. To run the notebooks, we recommand to [create a conda environment](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) with `python==3.11, pytorch==2.11.0, numpy, pandas, matplotlib, notebook, node2vec` and [cellmaps_pipeline](https://github.com/idekerlab/cellmaps_pipeline).
1. Follow `data/<subfolder>/readme.txt` to download the data from each source for the corresponding analysis.
2. Using analysis in `scripts/HEK293T_HCT116/` as an example:  
   [1] `1.1_input_raw_PPI_APMS.ipynb` and `1.1_input_raw_PPI_SECMS_EPIC.ipynb` parse the downloaded data to PPI. `1.2_input_PPI.ipynb` and `1.3_input_n2v_concat.ipynb` embed the PPI with node2vec as the initial embedding to the ProMERGE model.  
   (Note that for SEC-MS, we run EPIC which require its own python2 environment. Please see their [github](https://github.com/BaderLab/EPIC) for instruction.  
   [2] `2_coemb/` contain the scripts to run the ProMERGE and other baseline models. It is currently set up as utilizing the Slurm system in our server. Please configure to your server setting or simply run the shell lines directly.  
   [3] `3_PPI_from_emb.ipynb` and `3_run_cca.ipynb` to get the PPIs or specificity scores from each model's embeddings.  
3. For other folders under `scripts/`, they follow a similar three-step workflow, where the first step will differ to process the data accordingly.
4. Finally, use `scripts/figure/` and `scripts/other_analysis/` to generate the main and appendix figures.

   
