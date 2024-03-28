# Rebuttal

We are grateful to the reviewers for coming to our Rebuttal page. Here we have compiled new figures and tables which were relevant for the revision and requested by the reviewers. 
First, as requested by almost all reviewers, we have compiled a benchmarking study comparing Wormhole to current approaches for compuing Wasserstein distance and acceleration thereof. Briefly, we sampled different sized cohorts from each dataset (MNIST or ModelNet40) and measured the time required for Wormhole, Sinkhorn, Low Rank (LR) Sinkhorn and MetaOT to produce the pairwise distance matrix, on a fully utilized 80GB GPU.  
Other methods are more appropriate at tiny cohorts, as they do not require training a large, parametric model. However, even in cohorts with relatively few samples, Wormhole is superior. No method other than Wormhole can scale to complete datasets, requiring weeks of GPU time.

![alt text](RebuttalFigures/TimeComparisonAll.png?raw=true)

We note that several points in the figure above are projections. Pairwise computation for each cohort was done in batches, and we simply measured the time it took for Sinkhorn, LR Sinkhorn and Meta-OT to go through 100 batches and extrapolated for an estiamte of how long the entire sample would take.

We next extended our comparison to other Wasserstein Embedding methods to include DWE from ‘Learning Wasserstein Embeddings’, in addition to DiffusionEMD and Frechet based approximation. Despite the DWE code being defunct, we have completely updated it to python3 and JAX during the rebuttal period. We show that on low dimensional 2D and 3D point clouds, Wormhole produces finer embeddings.
Wormhole is also the only current OT based embedding method which can be applied to high-dimensional point-clouds. We further demonstrate this key feature by extending our manuscript to include two additional datasets: A scRNA-seq atlas of COVID patients and a spatial transcriptomics (seqFISH) dataset of mouse embryogenesis. Wormhole can produce accurate and OT preserving embeddings, while all other methods produce OOM errors.
Finally, we have also updated our correlation scores to include average and standard deviations of 10 random samples of 128 points, as requested by several reviewers.

![alt text](Tables/Table1.png?raw=true)

Not only relying on correlation with true Wasserstein distances, we also compute MSE, and show Wormhole consistently outperforms DWE. DiffusionEMD was not included in this comparison since its embeddings are only correlated with OT, rather than approach it.

![alt text](Tables/Table3.png?raw=true)

As requested by reviewers ‘3vyc’ and ‘Asp1’, we have ablated the impact of batch size on Wormhole training. As the figure below shows, the larger the size of the training batch, the better the test-set loss curves, as expected. This however comes at a cost, since computational complexity grows quadratically with batch size, as training requires computing the pairwise Wasserstein distance within each batch. All experiments in the paper used a batch size of 16, showing we can get great accuracy at remarkable speed.

![alt text](RebuttalFigures/BatchSizeAblation.png?raw=true)

Reviewer ‘Asp1’ also questioned whether there are differences between training and test set losses during Wormhole training. We are happy to find virtually no differences between them on the MNIST dataset.

![alt text](RebuttalFigures/LossCurves.png?raw=true)

Finally, reviewer  ‘Asp1’ raised concerns about the use of an MLP in obtaining class predictions from Wormhole embeddings. Since the embeddings are in 128 dimensions, we chose a deep model to classify them. For fairness, embeddings from DWE were treated in the same manner. Still, we find that even when an unideal linear classifier is used, test-set classification is still high and competitive with other methods. 

![alt text](Tables/Table4.png?raw=true)
