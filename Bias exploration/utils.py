from tqdm import tqdm
import numpy as np
from PIL import Image
import os

import time
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "mps"
print("Model loaded")



def zero_shot(images, concepts, concetps_neg=None, batch_size = 16, model=None, processor=None, device=device):

	global_logits = None
	negative_global_logits = None

	times = []
	for i in tqdm(range(0, len(images), batch_size)):

		image = [Image.open(images[img]) for img in range(i, min(i + batch_size,  len(images)))]
		
		logits_per_image = None
		negative_logits_per_image = None
		
		for j in range(0, len(concepts), batch_size):

			inputs = processor(text=concepts[j:j + batch_size], images=image, return_tensors="pt", padding=True).to(device)
			outputs = model(**inputs)
			
			start = time.time()
			# print(inputs.keys())
			_ = model.text_model(inputs.input_ids, inputs.attention_mask)
			logits_per_image = outputs.logits_per_image.detach().cpu() if logits_per_image is None else torch.cat((logits_per_image, outputs.logits_per_image.detach().cpu()), dim=-1)
			times.append(time.time() - start)

			if concetps_neg is not None:
				inputs = processor(text=concetps_neg[j:j + batch_size], images=image, return_tensors="pt", padding=True).to(device)
				outputs = model(**inputs)
				negative_logits_per_image = outputs.logits_per_image.detach().cpu() if negative_logits_per_image is None else torch.cat((negative_logits_per_image, outputs.logits_per_image.detach().cpu()), dim=-1)

		# times.append(time.time() - start)
		# print(times)
		global_logits = logits_per_image if global_logits is None else torch.cat((global_logits, logits_per_image))
		if concetps_neg is not None:
			negative_global_logits = negative_logits_per_image if negative_global_logits is None else torch.cat((negative_global_logits, negative_logits_per_image))

	# contrastive_matrix = torch.stack([global_logits, negative_global_logits], dim=-1)
	# contrastive_matrix_softmax = contrastive_matrix.softmax(dim=-1)
	# prediction = contrastive_matrix_softmax[:, :, 0] > contrastive_matrix_softmax[:, :, 1]

	return global_logits, sum(times)#prediction, contrastive_matrix_softmax

def zero_shot_BLIP_QA(model, images, concepts, processor, batch_size = 16,
                      device=device):


    global_probs = None

    for img in tqdm(images):

        image = [Image.open(img)]
        
        probs_image = None
        for j in range(0, len(concepts), batch_size):

            inputs = processor(images=image, text=concepts[j:j + batch_size], return_tensors="pt", padding=True).to(device)

            outputs = model.generate(model, **inputs).logits # tweak in .env/lib/python3.12/site-packages/transformers/models/blip/modeling_blip.py

            probs = torch.concat([outputs.softmax(dim=-1)[:,0,model.yes_tokens].sum(dim=-1, keepdim=True),
                            outputs.softmax(dim=-1)[:,0,model.no_tokens].sum(dim=-1, keepdim=True)
                        ], dim=-1)
            
            probs_image = probs if probs_image is None else torch.cat((probs_image, probs), dim=0)


        global_probs = probs_image.unsqueeze(0) if global_probs is None else torch.cat((global_probs, probs_image.unsqueeze(0)), dim=0)
    
    prediction = global_probs[:, :, 0] > global_probs[:, :, 1] #yes larger than no
    
    return prediction, global_probs



def MINICPM_format_QA(image, concepts, model, tokenizer):

	msgs = [{'role': 'user', 'content': i} for i in concepts]

	for i in range(len(msgs)):
		if image is not None and isinstance(msgs[i]["content"], str):
			msgs[i]["content"] = [image, msgs[i]["content"]]

	images = []
	tgt_sizes = []
	for i, msg in enumerate(msgs):
		role = msg["role"]
		content = msg["content"]
		assert role in ["user", "assistant"]
		if i == 0:
			assert role == "user", "The role of first msg should be user"
		if isinstance(content, str):
			content = [content]
		cur_msgs = []
		cur_imgs = []
		for c in content:
			if isinstance(c, Image.Image):
				slice_images, image_placeholder = model.get_slice_image_placeholder(
								image, tokenizer
							)
				cur_msgs.append(image_placeholder)
				for slice_image in slice_images:
					slice_image = model.transform(slice_image)
					H, W = slice_image.shape[1:]
					cur_imgs.append(model.reshape_by_patch(slice_image))
					tgt_sizes.append(torch.Tensor([H // model.config.patch_size, W // model.config.patch_size]).type(torch.int32))
			elif isinstance(c, str):
				cur_msgs.append(c)
		msg["content"] = "\n".join(cur_msgs)
		images.append(cur_imgs)
	return msgs, images, tgt_sizes
    
def zero_shot_MINICPM_QA(model, images, concepts, tokenizer, batch_size = 1):

	global_probs = None

	for img in tqdm(images):

		image = Image.open(img)
		
		probs_image = None
		for j in range(0, len(concepts), batch_size):
			
			text = [f"Is this a {k}?" for k in concepts[j: j + batch_size]]
			msgs, images, tgt_sizes = MINICPM_format_QA(image, text, model, tokenizer=tokenizer)
			
			outputs = model.get_logits(model,
						msgs = msgs,
						images = images,
						tokenizer = tokenizer,
						tgt_sizes = tgt_sizes,
						).logits
			
			probs = torch.concat([outputs.softmax(dim=-1)[:,-1,model.yes_tokens].sum(dim=-1, keepdim=True),
							outputs.softmax(dim=-1)[:,-1,model.no_tokens].sum(dim=-1, keepdim=True)
						], dim=-1)
			
			probs_image = probs if probs_image is None else torch.cat((probs_image, probs), dim=0)


		global_probs = probs_image.unsqueeze(0) if global_probs is None else torch.cat((global_probs, probs_image.unsqueeze(0)), dim=0)

	prediction = global_probs[:, :, 0] > global_probs[:, :, 1] #yes larger than no
	return prediction, global_probs
