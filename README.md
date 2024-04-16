Robust Representation of Oil Wells' Intervals via Sparse Attention Mechanism
=====

Code for experiments from the article of the same name. Include framework `trans_oil_gas` for dataset of well-intervals generation and training and testing Transformer-based (with our proposed model Reguformer, the original Transformer, Performer, DropDim, and LRformer) Siamese and Triplet models. 

The focus of the paper is on the well logs; however, the superior quality of Reguformers was proved on testing them on three additional datasets: [Boston crime reports](https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston), [weather logs stations] (https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation), and [the stocks market](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231). 

Installation of `trans_oil_gas`:
-----
1. Clone this repository
2. Install all necessary libraries via command in terminal: `pip install Reguformer/`
3. Use our framework via importing modules with names started with `utils_*` from `trans_oil_gas` 

Reproducing experiments from the article
-----
To reproduce all our experiments from the article, run the experiments from the `notebooks` folder in the following order:
1. Run jupyter notebook `well_linking.ipynb`. It will train all models (the vanilla Transformer model, Reguformers with all regularization strategies, and Performer) with Siamese and Triplet loss functions. All the following notebooks use the best models obtained on this step (unless otherwise stated).
2. Conduct the experiments with the embeddings quality evaluation:
    * Run `emb_quality_classification.ipynb` for obtaining well-intervals embeddings and their classification on wells with downstream classifiers: XGBoost, One linear layer, and $3$-layered fully-connected neural network;
    * `emb_quality_clustering.ipynb` for obtaining well-intervals embeddings and their clustering on wells;  
    * `emb_quality_tsne.ipynb` for t-SNE compressition and visualization of embeddings of Reguformer with top queries, Reguformer with random queries and keys, and the vanilla Transformer;  
   and the experiment with GPU inferene time measure:
    * `inference_time.ipynb`, which measures GPU inferene time for Reguformer with different regularizations. 
3. Run the `robust.ipynb` for the experiments with models' (Reguformer with top queries, Reguformer with top keys, Reguformer with top queries and keys, Reguformer with random queries and keys, and the vanilla Transformer) robustness. Moreover, it also calculates the correlation coefficient between the vanilla Transformer attention scores and gradients. 
4. Conduct the experiment with the vanilla Transformer attention analysis via running the notebook `transformer_attention_analysis.ipynb`

Here we present the syntetic dataset includes $4$ wells.
In all the notebooks the values for train and test sample generaltion, the number of epochs, etc. used in the article are commented and smaller values for the demonstration are presented. In the original dataset $28$ wells are presented.   

License
-----
The project is distributed under [MIT License](https://github.com/roguLINA/Reguformer/blob/main/License.txt).
