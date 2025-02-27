
class settings:

    #HDBSCAN
    min_cluster_size=5
    metric='euclidean'
    cluster_selection_method='eom'
    prediction_data=True

    #UMAP
    n_neighbors=15
    n_components=5
    min_dist=0.0 
    metric='cosine'
