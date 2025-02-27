import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

import pickle, nltk
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline
from nltk.corpus import stopwords as stpw
from nltk.stem import WordNetLemmatizer

from glob import glob
from tqdm import tqdm

from hdbscan import HDBSCAN
from umap import UMAP

from ConceptModeling import ConceptModel
from settings import settings

nltk.download('stopwords')
nltk.download('wordnet')


class ConceptMiner():

    def __init__(self, images, load_model: bool = False,
                  model_path: str = "Salesforce/blip-image-captioning-base"):
        
        self.images = images
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if load_model:
            self.model = model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.pipe = pipeline("image-to-text", model=model, tokenizer=self.processor.tokenizer,
                             image_processor = self.processor, device=self.device)

    def get_image_caption(self, get_encodings : bool = False,  batch_size : int = 8) -> tuple:

        stopwords = stpw.words('english')
        lemmatizer = WordNetLemmatizer()
        
        captions = []
        encodings = None
        for i in tqdm(range(0, len(self.images), batch_size)):

            batch_images = [Image.open(image) for image in self.images[i:i+batch_size]]
            batch_encodings = self.processor(batch_images, return_tensors="pt", padding=True).to(self.device)
            if get_encodings:
                encodings = self.model.vision_model(**batch_encodings).pooler_output.detach().cpu() if encodings is None else torch.cat([encodings, self.model.vision_model(**batch_encodings).pooler_output.detach().cpu() ])

            captions += [' '.join([lemmatizer.lemmatize(word) for word in i[0]['generated_text'].split() if word not in stopwords]) for i in self.pipe(images=batch_images, max_new_tokens=75)] 

        if get_encodings:
            return encodings, captions
        return captions

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data['encodings'], data['captions']
    
    def concept_modeling(self, encodings: np.ndarray, 
                         captions: list, 
                         batch_size: int = 8):
        
        hdbscan_model = HDBSCAN(min_cluster_size= settings.min_cluster_size,
                                metric= settings.hdbscan_metric,
                                cluster_selection_method= settings.cluster_selection_method,
                                prediction_data= settings.prediction_data)

        umap_model = UMAP(n_neighbors=settings.n_neighbors, n_components= settings.n_components,
                        min_dist= settings.min_dist, metric= settings.umap_metric)


        concept_model = ConceptModel(hdbscan_model=hdbscan_model, umap_model=umap_model)
        _ = concept_model.fit_transform(images = self.images, image_embeddings = encodings,
                                            docs = captions, #selected_nouns,
                                            n_topics_per_cluster = 30,
                                            text_image_matched = True,
                                            text_econder={'processor': self.processor, 'decoder': self.model.text_decoder, 
                                                            'batch_size': batch_size})
        
        generated_concepts = list(set([k.strip() for i in concept_model.topics.values() for k in i.split(',')]))
        return concept_model, generated_concepts
