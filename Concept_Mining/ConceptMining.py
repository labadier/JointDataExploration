import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

import pickle, nltk
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline
from nltk.corpus import stopwords as stpw

from glob import glob
from tqdm import tqdm

from hdbscan import HDBSCAN
from umap import UMAP

nltk.download('stopwords')


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
        
        self.hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                cluster_selection_method='eom', prediction_data=True)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


    def get_image_caption(self, get_encodings : bool = False,  batch_size : int = 8) -> tuple:

        stopwords = stpw.words('english')
        
        captions = []
        encodings = None
        for i in tqdm(range(0, len(self.images), batch_size)):

            batch_images = [Image.open(image) for image in self.images[i:i+batch_size]]
            batch_encodings = self.processor(batch_images, return_tensors="pt", padding=True).to(self.device)
            if get_encodings:
                encodings = self.model.vision_model(**batch_encodings).pooler_output.detach().cpu() if encodings is None else torch.cat([encodings, self.model.vision_model(**batch_encodings).pooler_output.detach().cpu() ])
            
            captions += [' '.join([word for word in i[0]['generated_text'].split() if word not in stopwords]) for i in self.pipe(images=batch_images, max_new_tokens=75)] 

        if get_encodings:
            return encodings, captions
        return captions

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data['encodings'], data['captions']
