import minari
import torch


class EpisodesDataset(torch.utils.data.Dataset): 
    def __init__(self, cl_model = None, minari_dataset = "D4RL/pointmaze/large-v2", n_episodes=1, episodeData=None):
        """
        Dataset to store the z-representation of the states of their corresponding episodes. 

        Args: 
            cl_model: A pretrained contrastive learning model to encode states to z-representations. 
            minari_dataset: The type of minari dataset to use.
            n_episodes: The number of episodes to sample. 
            episodeData: If you want to import the episodeData from minari into this dataset. 

        """
        assert cl_model != None, "Must input a contrastive learning model to obtain z-representations!"

        self.cl_model = cl_model 

        if episodeData: 
            self.episodeData = episodeData
        else: 
            self.minari_dataset = minari.load_dataset(minari_dataset)
            self.episodeData = self.minari_dataset.sample_episodes(n_episodes=n_episodes) # list of episodes [ep1, ep2, ep3, ...]

        # precompute all z representations and store them 
        self.z_data = [] 
        with torch.no_grad(): 
            for ep in self.episodeData: 
                x = torch.as_tensor(ep.observations["observation"], dtype=torch.float32)
                z = cl_model(x)
                self.z_data.append(z) 

    def __len__(self): 
        """
        Returns the number of episodes in the dataset. 
        """
        return len(self.z_data)

    def __getitem__(self, idx): 
        """
        Returns the z-representation specified by "idx". 
        This sample is the list of states in the form of a tensor. 
        """
        return self.z_data[idx]

        

