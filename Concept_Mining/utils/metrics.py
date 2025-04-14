from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(sentences, model, tokenizer, batch_size = 16):
    with torch.no_grad():

        output = None
        for i in range(0, len(sentences), batch_size):
            encoded_input = tokenizer(sentences[i: i + batch_size], padding=True, return_tensors='pt')

            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            output = sentence_embeddings if output is None else torch.cat([output, sentence_embeddings])

    return output

def multivariate_kullback_leibler(reference_distribution, new_distribution, epsilon=1e-5):
    
    # Estimate the mean and covariance matrix for the first distribution
    mu_1 = torch.mean(reference_distribution, axis=0, dtype=torch.float64)
    sigma_1 = torch.cov(reference_distribution.T) + epsilon * np.eye(mu_1.shape[0])

    # Estimate the mean and covariance matrix for the second distribution
    mu_2 = torch.mean(new_distribution, axis=0, dtype=torch.float64)
    sigma_2 = torch.cov(new_distribution.T) + epsilon * np.eye(mu_2.shape[0])

    k = mu_1.shape[0]

    # Compute the inverse and determinant of sigma_2
    sigma_2_inv = torch.linalg.inv(sigma_2)
    det_sigma_2 = torch.linalg.det(sigma_2)
    det_sigma_1 = torch.linalg.det(sigma_1)

    # Compute the KL divergence
    kl_divergence = 0.5 * (torch.trace(sigma_2_inv @ sigma_1) + 
                        (mu_2 - mu_1).T @ sigma_2_inv @ (mu_2 - mu_1) - 
                        k + 
                        torch.log(det_sigma_2 / det_sigma_1))

    return kl_divergence.item()

def kl_metric( y: list[str], y_hat: list[str], reference: list[str], epsilon = 1e-5) -> float:
	
    """
    
    Compute the KL divergence of two distribution in the representation space of a transformer model.
    Each concept is assumed to be a point in the embedding space, therefore we can consider the distribution
    of the predicted set of concepts and the distribution of the ground truth set of concepts. This gives us
    how much two concept sets are semantically different.
    Args:
        y (list[str]): The predicted set of concepts
        y_hat (list[str]): The ground truth set of concepts
        reference (list[str]): The reference set of concepts. Ussually y \cup y_hat \subseteq reference to estimate correctl
                                the PCA required to project the embeddings and compute the KL divergence.
        epsilon (float): Small value to avoid division by zero

    Returns:
        kl (float): The KL divergence between the predicted and ground truth set of concepts
    """
     
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    
    y_embedd = get_embedding(y, model, tokenizer)
    y_hat_embedd = get_embedding(y_hat, model, tokenizer)
    reference_embedd = get_embedding(reference, model, tokenizer)

    pca = PCA(n_components=11, random_state=42)
    pca.fit(reference_embedd)

    y_embedd_proj = torch.tensor(pca.transform(y_embedd))
    y_hat_embedd_proj = torch.tensor(pca.transform(y_hat_embedd))

    kl = multivariate_kullback_leibler(y_embedd_proj, y_hat_embedd_proj, epsilon)
    return kl


def get_matchings(y: list[str], y_hat: list[str]) -> float:
    """
    Compute the matchings between the predicted and ground truth set of concepts.
    Since this is syntactic rather than semantic, we can use the WordNet library to extend the concepts, 
    including the adjectives to finally compute the matchings.
    Args:
        y (list[str]): The predicted set of concepts
        y_hat (list[str]): The ground truth set of concepts
    Returns:
        matchings (float): The matchings between the predicted and ground truth set of concepts
    """

    nltk.download('wordnet')

    def extend_concepts(concepts):
        extended_concepts = []
        for concept in concepts:
            for ss in wn.synsets(concept):
                extended_concepts += ss.lemma_names()
        return list(set(extended_concepts))

    def count_matches(points, ground_truth):

        lemmatizer = WordNetLemmatizer()

        lematized_points = [lemmatizer.lemmatize(i) for i in points]
        lematized_ground_truth = [lemmatizer.lemmatize(i) for i in ground_truth]

        matches = sum([1 for i in lematized_points if i in lematized_ground_truth])

        return matches

    e_y_hat = extend_concepts(y_hat)
    e_y = extend_concepts(y)
    return count_matches(e_y_hat, e_y)/(len(e_y) + (not len(e_y)))
