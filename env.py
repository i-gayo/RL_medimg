import torch 
import torchvision
import numpy as np 
from utils import CancerDataset, DataSampler
import gym
from gym import spaces
from models import CNNFeatureExtractor, ResNetFeatureExtractor
import copy 

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

class ImgEnv_discrete(gym.Env):
    
    """
    An environment to simulate lesion training 
    """
    
    def __init__(self, 
                 data_sampler,
                 patch_size = 30,
                 max_steps = 30,
                 start_center = True,
                 use_shaped_reward = False):
        """
        Initialises which environment to learn from -> single patient or using multiple patients for lesion detection
        """
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2, patch_size, patch_size), dtype=np.float64)
        self.action_space = spaces.Discrete(5) # x (-5,0,+5) y (-5,0,5) z (fire or no fire), end (terminate or keep going)

        # Initialise action_space 
        
        self.data_sampler = data_sampler
        self.patch_size = patch_size 
        self.start_center = start_center
        self.use_shaped_reward = use_shaped_reward
        
        # Sample data 
        self.img, self.label, self.start_x, self.start_y  = self.sample_data()
        self.img_size = self.img.squeeze().size() # Assumes 2D image
        # Initialise starting position 
        self.current_pos = torch.tensor([self.start_x, self.start_y])
        self.visited_states = torch.zeros_like(self.img)
        self.max_steps = 30 
        self.step_count = 0 
        
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
        
        boundary_idx = 2
        if depth == width:
        
            # Idx to sample from 
            sample_idx = np.arange(0, depth , self.patch_size)
            if self.start_center: 
                # START AT CENTRE OF SAMPLE IDX 
                mid_idx = round((len(sample_idx) + 1 )/2) -1 # subtract 1 bc of 0 idx 
                x_idx = sample_idx[mid_idx]
                y_idx = sample_idx[mid_idx]
            else:
                x_idx = np.random.choice(sample_idx[boundary_idx:-boundary_idx])
                y_idx = np.random.choice(sample_idx[boundary_idx:-boundary_idx])
        else:
            sample_idx = np.arange(0, depth , self.patch_size)
            if self.start_center:
                mid_idx = round((len(sample_idx) + 1 )/2) -1 # subtract 1 bc of 0 idx 
                x_idx = sample_idx[mid_idx]
            else:
                x_idx = np.random.choice(sample_idx[boundary_idx:-boundary_idx])
            
            sample_idx = np.arange(0, width, self.patch_size)
            if self.start_center:
                mid_idx = round((len(sample_idx) + 1 )/2) -1 # subtract 1 bc of 0 idx 
                y_idx = sample_idx[mid_idx]
            else:
                y_idx = np.random.choice(sample_idx[boundary_idx:-boundary_idx])
        
        start_x, start_y = x_idx, y_idx

        return img.squeeze(), label.squeeze(), start_x, start_y 

        # Initialise middle patches (eg 0)
                         
    def step(self, action):
        """
        Performs step in the environment 
        """
        # 
        action = action
        self.step_count += 1
        # 1. UPDATE X AND Y POSITION 
        # if action : 0 Left x
        # if action : 1 left y 
        # if action 2 up x
        # if action 3 lower y 
        # if action 5: terminate 
        # 5 actions to choose from! 
        # UPDATE OBSERVATION AND STATUS
        INIT_X = copy.deepcopy(self.current_pos[0])
        INIT_Y = copy.deepcopy(self.current_pos[1])
        
        if self.step_count >= self.max_steps:
            DONE = True 
            X = INIT_X
            Y = INIT_Y 
        else:

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
                print(f"Chosen done  aciton : {action}")
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
            
            #f"Init x : {INIT_X} X : {X} \n init y : {INIT_Y} Y : {Y}")
            # TODO : Penalise for not moving patches (eg stuck at the edge)
        self.current_pos = torch.tensor([X,Y])        
        # 2. sChange patch obsevration : extract new patch 
        
        # Change "visited states" as final value -> 3 for debugging only!!! 
        
        if DONE: 
            state_val = 1.5
        else:
            state_val = 1.0
        patch, resize_visited_states, obs, visited_states = self.get_patch(self.img, X, Y, self.patch_size, self.visited_states, state_val)
        
        self.visited_states = visited_states # Update visited states 
        
        # visited states is not updated atm; current posiiton only saved NOT previous positions -> TODO : add option to add this later!
        
        # from matplotlib import pyplot as plt 
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(obs)
        # axs[1].imshow(visited_states)
        # Update visited states grid 
        if self.use_shaped_reward:
            prev_pos = torch.tensor([INIT_X, INIT_Y])

            reward = self.compute_shaped_reward(self.img, 
                                                    self.label, 
                                                    prev_pos, 
                                                    self.current_pos)
        else:
            
            reward = self.compute_reward(self.img, self.label, X,Y)

        if DONE:
            # Positive rweard if found lesion ; negative if not found lesion 
            if reward < 0: 
                reward = -10 #-5 no lesion found  # before was -5 
            else: 
                reward = reward*10 # +10 if found lesion # after was + 10
    
        # compute reward : -0.1 if no lesion, 
        # -50 if terminate with no lesion 
        # + 100 if terminate with lesion 
        
        # Take step within environment -> right / left / centre 
        info = {'current_pos' : self.current_pos,
                'visited_states' : self.visited_states,
                'img' : self.img,
                'label' : self.label}
        
        print(f"Step count : {self.step_count} Reward : {reward}")
        
        return obs, reward, DONE, info 
    
    def reset(self):
        """
        Resets environment : if single patient -> initialise starting position to find lesion once more 
        """
        print(f"\n Episode restarting : ")
        # Sample data 
        self.img, self.label, self.start_x, self.start_y  = self.sample_data()
        self.step_count = 0
        
        # Initialise starting position 
        self.current_pos = torch.tensor([self.start_x, self.start_y])
        self.visited_states = torch.zeros_like(self.img)
        
        # 2. sChange patch obsevration : extract new patch 
        patch, resize_visited_states, obs, visited_states  = self.get_patch(self.img, self.start_x, self.start_y, self.patch_size, self.visited_states, state_val = 0.5)
        self.visited_states = visited_states # update initial position 
        
        return obs 
  
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
            reward = 1 # hits lesion! 
            
            # Include shaped reward : +0.1, -0.5
            
        else:
            reward = -0.1 # outside prostate 
        
        return reward
    
    def compute_shaped_reward(self, img, label, prev_pos, new_pos):
        """
        Includes shaped reward function to current implementation for improved learning
        
        """
        print(f"COMPUTING REWARD")
        X = new_pos[0]
        Y = new_pos[1]
        
        label_score = label[X:X+self.patch_size, Y:Y+self.patch_size]
        scores = torch.unique(label_score)

        if torch.any(scores >= 2):
            reward = 1 # hits lesion! 
            
        else:
            reward = -0.2 # outside prostate 
            
        # Include shaped reward (+0.1 for getting lcoser, -0.1 otherwise)
        print(f"DEBUG BEFORE SHAPED REWARD")
        shaped_reward = self.shaped_reward(img, label, prev_pos, new_pos)
        print(f"DEBUG AFTER SHAPED REWARD")
        reward = reward + shaped_reward 
        print("DEBUG AFTER ADDING REWARDS")
        print(f"Reward : {reward}")
        return reward
    
    def shaped_reward(self, img, label, prev_pos, new_pos, magnitude = 0.1):
        """
        Check if new step is closer to or further than lesion
        """
        
        print(f"{torch.where(label >= 2)}")
        x_lesion, y_lesion = torch.stack(torch.where(label >= 2)).float().mean(dim=1)
        #x_lesion, y_lesion = 0.0,0.0
        
        lesion_com = torch.tensor([int(x_lesion), int(y_lesion)])
        # check dist from prev pos 
        
        prev_dist = torch.mean(torch.abs(prev_pos - lesion_com).float())
        # check distance form new pos 
        new_dist = torch.mean(torch.abs(new_pos - lesion_com).float())
        
        # if closer, reward = 1, if further reward = -1, if same reward = 0
        signed_reward = magnitude*torch.sign(prev_dist - new_dist) # If +ve : closer, if -ve : further, if 0 : same
        
        return signed_reward
        
    def get_patch(self, img, x_idx, y_idx, patch_size = 30, visited_states = None, state_val = 1):
        """
        A function to obtain current patch position 
        state_val : 0 not visited, 0.5 start, 1.0 visited, 2.0 : terminal_state
        """
        # 1. Obtain patch from original image 

        assert x_idx <= self.img_size[0] - patch_size
        assert y_idx <= self.img_size[1] - patch_size 
        
        patch = img[x_idx:x_idx+patch_size, y_idx:y_idx+patch_size]
        
        # 2. Keep history of previous patch positions 
        if visited_states is None:
            # Initiliase empty array if no visited states yet 
            visited_states = torch.zeros_like(img)
        

        visited_states[x_idx:x_idx+patch_size, y_idx:y_idx+patch_size] = state_val
        
        resize = torchvision.transforms.Resize((patch_size, patch_size))
        resize_visited_states = 1.0*(resize(visited_states.unsqueeze(0)).squeeze() > 0.0)

        # optional, combine obs: 
        obs = torch.stack([patch, resize_visited_states])
        
        
        return patch, resize_visited_states, obs, visited_states 



if __name__ == '__main__':
    
    # Initialise: 
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.ppo.policies import CnnPolicy
    
    DATASET_PATH = './Data/processed_data'
    PATCH_SIZE = 15 
    
    cancer_ds = CancerDataset(DATASET_PATH,
                              mode = 'train',
                              single_patient = True,
                              patient_idx = 1,
                              give_2d = True)
    cancer_sampler = DataSampler(cancer_ds)
    
    env = ImgEnv_discrete(cancer_sampler,
                          patch_size = PATCH_SIZE)
    
    obs, reward, done, info = env.step([1])
    LOG_DIR = './logs'
    policy_kwargs = dict(features_extractor_class = CNNFeatureExtractor, \
        features_extractor_kwargs=dict(num_channels = 1)) #, activation_fn = torch.nn.Tanh)
    
    agent = PPO(CnnPolicy, 
            env, 
            n_epochs = 2, 
            tensorboard_log = LOG_DIR, 
            policy_kwargs = policy_kwargs)
    
    #agent.learn(1000, 
    
    print('fuecoco')