=======
Inputs
=======

The tool takes as input path to directories containing embedding file (where it looks for files named ``image_emd.tsv``
or ``ppi_emd.tsv``) or paths to specific embedding files. It requires two or more embedding files in TSV format.
If directories are ro-crates it uses the json files to retrieve the metadata.


-  ``embeddings files``
    A tab-separated file containing the embeddings. Each row corresponds to gene name and the subsequent columns
    contain the embedding vector.

.. code-block::

            1	2	3	4
    BPTF	-0.037030112	-0.139459819	0.417184144	0.386600941
    KAT2B	0.02969132	-0.139459819	-0.038685802	0.136547908
    PARP1	-0.037030112	-0.139459819	0.540370524	0.119614214
    MSL1	0.18169874	-0.139459819	-0.038685802	0.152157351
    KAT6B	-0.037030112	-0.139459819	0.308141887	0.257056117

