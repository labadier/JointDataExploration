
class settings:

    #HDBSCAN
    min_cluster_size=20 #TODO tune for dataset size
    hdbscan_metric='euclidean'
    cluster_selection_method='eom'
    prediction_data=True

    #UMAP
    n_neighbors=15
    n_components=5
    min_dist=0.0 
    umap_metric='cosine'

class pretraining_settings:
    
    episodes = 500
    episode_length = 1000
    # dataset_coverage_step = .75
    gamma = 0.99
    # actions_embedd_dim = 2
    buffer_size = 64

    lr_actor = 0.001
    lr_critic = 0.0001

    temperature = 5.0
    final_temperature = 0.1
    decay_rate = 0.99  # Decay per episode
