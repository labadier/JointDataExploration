import random; random.seed(42)
import numpy as np; np.random.seed(42)


def generate_episode(topic_groups, concept_instance, images, threshold = 10):

	"""
    Generates an episode consisting of batches of images explored by the user before introducing
	a concept, ensuring that concepts appear at least a specified number of times within the batches.

    Parameters:
    -----------
    topic_groups : dict
        A dictionary where keys represent topic labels and values are lists of associated concepts.
    concept_instance : dict
        A dictionary mapping each concept to a set of images representing that concept.
    images : list
        A list of all available images.
    threshold : int, optional
        The minimum number of times each concept should appear in the episode (default is 10).

    Returns:
    --------
    images_batches : list of sets
        A list where each element is a batch (set) of images corresponding to a step in the episode.
	topic : list
		The list of externalized concepts in the selected topic.
    """
	
	images_batches = []
	observed_images = set()
	threshold = 10

	topic = topic_groups[np.random.choice(list(topic_groups.keys()))]
	random.shuffle(topic)
	concept_appereance = {i:set() for i in topic}

	for step in topic:

		if len(concept_appereance[step]) < threshold:
			sample1 = set(random.sample(concept_instance[step], 
										threshold - len(concept_appereance[step])))
			
			sample = sample1 | set(random.sample(images, 
										len(images)//len(topic) - len(sample1))) 
		else:
			sample = set(random.sample(images, len(images)//len(topic))) 

		observed_images |= sample
		images_batches.append(sample)

		for key in concept_appereance:

			added = set(concept_instance[key]) & sample
			concept_appereance[key] |= added

	return images_batches, topic
