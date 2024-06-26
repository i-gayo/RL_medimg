import torch 
import torchvision
import numpy as np 
from utils import CancerDataset, DataSampler
#### Idea : #
# https://gymnasium.farama.org/environments/atari/ms_pacman/

# Patient-specific training : train on indivdiual patients


# Train on multiple patients : 


# Playing around with reward function : reward agent for finding lesion WITH shaped reward and with NO shaped reward 
# Shaped reward vs no shaped reward 

# Lessons : Formulating a reward function is tricky -> may require inverse reinforcmeent learning / engineering 



# Activtiies: 

# Patient-specific

    # Non-shaped reward
    
    
    # Shaped reward 
    
    
# Multiple patients training

class ImgEnv_discrete():
    
    """
    An environment to simulate lesion training 
    """
    
    def __init__(self, 
                 data_sampler,
                 patch_size = 30):
        """
        Initialises which environment to learn from -> single patient or using multiple patients for lesion detection
        """
        
        # Initialise action_space 
        
        self.data_sampler = data_sampler
        self.patch_size = patch_size 
        
        # Sample data 
        self.img, self.label, self.start_x, self.start_y  = self.sample_data()
        self.img_size = self.img.squeeze().size() # Assumes 2D image
        # Initialise starting position 
        self.current_pos = torch.tensor([self.start_x, self.start_y])
        self.visited_states = torch.zeros_like(self.img)
        
    def sample_data(self):
        """
        Sample data given data sampler : assumes widht and depth are the same 
        """
        
        img, label, patient_name = self.data_sampler.sample_data()
        depth, width = img.size()[1:]
        
        # To asssert : patch size and img size are divisible (non-zero remainder)
        assert depth % self.patch_size == 0
        assert width % self.patch_size == 0
            
        # patch_size within image : 
        self.num_patches = round(img.size()[-1] / self.patch_size) # how many patches within image 
        
        if depth == width:
        
            # Idx to sample from 
            sample_idx = np.arange(0, depth , self.patch_size)
            x_idx = np.random.choice(sample_idx)
            y_idx = np.random.choice(sample_idx)
        else:
            sample_idx = np.arange(0, depth , self.patch_size)
            x_idx = np.random.choice(sample_idx)
            
            sample_idx = np.arange(0, width, self.patch_size)
            y_idx = np.random.choice(sample_idx)
        
        start_x, start_y = x_idx, y_idx

        return img.squeeze(), label.squeeze(), start_x, start_y 

        # Initialise middle patches (eg 0)
                         
    def step(self, action):
        """
        Performs step in the environment 
        """
        
        # 1. UPDATE X AND Y POSITION 
        # if action : 0 Left x
        # if action : 1 left y 
        # if action 2 up x
        # if action 3 lower y 
        # if action 5: terminate 
        # 5 actions to choose from! 
        
        INIT_X = self.current_pos[0]
        INIT_Y = self.current_pos[1]
        DONE = False 
            
        if action == 0: 
            X = INIT_X + self.patch_size
            Y = INIT_Y 
        elif action == 1: 
            X = INIT_X - self.patch_size 
            Y = INIT_Y
        elif action == 2: 
            X = INIT_X 
            Y = INIT_Y - self.patch_size 
        elif action == 3:
            X = INIT_X 
            Y = INIT_Y + self.patch_size
        else: 
            DONE = True 
            X = INIT_X
            Y = INIT_Y
            
        # Check that X and Y are within image limits (eg 0 < X < IMG_SIZE ) if not, keep the same patch! 
        if X < 0:
            X = INIT_X 
        elif X > self.img_size[0] - self.patch_size:
            X = INIT_X
        
        if Y <= 0:
            Y = INIT_Y 
        elif Y > self.img_size[1] - self.patch_size:
            Y = INIT_Y 
        
        print(f"Init x : {INIT_X} X : {X} \n init y : {INIT_Y} Y : {Y}")
        # TODO : Penalise for not moving patches (eg stuck at the edge)
                
        # 2. sChange patch obsevration : extract new patch 
        obs, visited_states = self.get_patch(self.img, X, Y, self.patch_size, self.visited_states)
        
        # Update visited states grid 
        reward = self.compute_reward(self.img, self.label, X,Y)
        self.current_pos = torch.tensor([X,Y])
        
        # compute reward : -0.1 if no lesion, 
        # -50 if terminate with no lesion 
        # + 100 if terminate with lesion 
        
        # Take step within environment -> right / left / centre 
        info = {'current_pos' : self.current_pos}
        
        return obs, reward, DONE, info 
    
    def reset(self):
        """
        Resets environment : if single patient -> initialise starting position to find lesion once more 
        """
        # Sample data 
        self.img, self.label, self.start_x, self.start_y  = self.sample_data()
        
        # Initialise starting position 
        self.current_pos = torch.tensor([self.start_x, self.start_y])
        self.visited_states = torch.zeros_like(self.img)
        
    def visualise_strategy(self):
        """
        Given sequence of actions, visualise where agent moves 
        """
    
    def compute_reward(self, img, label, X,Y):
        """
        Checks if current position has lesion or not 
        """

        label_score = label[X:X+self.patch_size, Y:Y+self.patch_size]
        scores = torch.unique(label_score)

        if torch.any(scores >= 2):
            reward = 10 # hits lesion! 
        else:
            reward = -0.1 # outside prostate 
        
        return reward
    
    def get_patch(self, img, x_idx, y_idx, patch_size = 30, visited_states = None):
        """
        A function to obtain current patch position 
        
        """
        # 1. Obtain patch from original image 

        assert x_idx <= self.img_size[0] - patch_size
        assert y_idx <= self.img_size[1] - patch_size 
        
        patch = img[x_idx:x_idx+patch_size, y_idx:y_idx+patch_size]
        
        # 2. Keep history of previous patch positions 
        if visited_states is None:
            # Initiliase empty array if no visited states yet 
            visited_states = torch.zeros_like(img)
        
        visited_states[x_idx:x_idx+patch_size, y_idx:y_idx+patch_size] = 1.0 
        resize = torchvision.transforms.Resize((patch_size, patch_size))
        resize_visited_states = resize(visited_states.unsqueeze(0)).squeeze()
    
        return patch, resize_visited_states 
    

    
if __name__ == '__main__':
    
    # Initialise: 
        
    DATASET_PATH = './Data/processed_data'
    
    cancer_ds = CancerDataset(DATASET_PATH,
                              mode = 'train',
                              single_patient = True,
                              patient_idx = 1,
                              give_2d = True)
    cancer_sampler = DataSampler(cancer_ds)
    
    env = ImgEnv_discrete(cancer_sampler,
                          patch_size = 10)
    
    obs, reward, done, info = env.step([1])