# Loading relevant datasets, dataloader and environments 
from utils import DataSampler, CancerDataset, SaveOnBestTrainingReward
from env import ImgEnv_discrete
from models import CNNFeatureExtractor, ResNetFeatureExtractor

from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os 
import torch 
from torch.utils.tensorboard import SummaryWriter

# Things to modify : 

# Observatino : use history or no history 

# Reward : use shaped reward or no shaped reward! 

#     

import argparse 

parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='debug',
                    help='Log dir to save results to')

parser.add_argument('--single_patient',
                    metavar='single_patient',
                    type=str,
                    action='store',
                    default='False',
                    help='To train patient-specific or multi-patient')

parser.add_argument('--patient_idx',
                    metavar='patient_idx',
                    type=str,
                    action='store',
                    default='10',
                    help='Which patient idx for training')

args = parser.parse_args()

def evaluate_agent(env, agent, num_episodes = 20):
    
    all_ep_rewards = [] 
    all_ep_len = [] 
    
    for i in range(num_episodes):
        obs = env.reset()
        ep_reward = 0 
        ep_len = 0 
        done = False 
        while not done:
            action, _states = agent.predict(obs, deterministic = True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward 
            ep_len += 1 
        all_ep_rewards.append(ep_reward)
        all_ep_len.append(ep_len)

    # Compute mean reward, mean len 
    
    all_ep_rewards = torch.tensor(all_ep_rewards)
    all_ep_len = torch.tensor(all_ep_len)
    
    mean_reward = torch.mean((all_ep_rewards))
    mean_ep_len = torch.mean((all_ep_len))
    
    print(f"mean_reward : {mean_reward} ± {torch.std(all_ep_rewards)}, mean_ep_len : {mean_ep_len}± {torch.std(all_ep_len)} \n")
        
    # Save episode runs (Reward, Episode length)
    return mean_reward, mean_ep_len
    
    return all_ep_rewards, all_ep_len

if __name__ == '__main__':
    
    DATASET_PATH = './Data/processed_data'
    PATCH_SIZE = 30
    BASE_FOLDER = 'trained_models'
    LOG_DIR = os.path.join(BASE_FOLDER, args.log_dir)   #'./train_results_multipatient'
    os.makedirs(LOG_DIR, exist_ok = True)
    
    TRAIN_SINGLEPATIENT = (args.single_patient == 'True')
    PATIENT_IDX = int(args.patient_idx)
    
    train_ds = CancerDataset(DATASET_PATH,
                              mode = 'train',
                              single_patient = TRAIN_SINGLEPATIENT,
                              patient_idx = PATIENT_IDX,
                              give_2d = True)
    
    train_sampler = DataSampler(train_ds)
    
    # Initialise validation dataset 
    val_ds = CancerDataset(DATASET_PATH,
                              mode = 'val',
                              single_patient = TRAIN_SINGLEPATIENT,
                              patient_idx = PATIENT_IDX,
                              give_2d = True)
    
    val_sampler = DataSampler(val_ds)
    
    if TRAIN_SINGLEPATIENT: 
        # Initialise at dif point sof grid each time
        start_centre = False
    else: 
        # If multi-patient : start from centre of grid always, for robustness (to learn what strategies for finding lesion
        # are most effective)
        start_centre = True 

    env = ImgEnv_discrete(train_sampler,
                          patch_size = PATCH_SIZE, 
                          start_center = start_centre)
    
    env = Monitor(env, filename=LOG_DIR)
    
    val_env = ImgEnv_discrete(val_sampler, 
                              patch_size = PATCH_SIZE, 
                              start_center = start_centre)
    writer = SummaryWriter(os.path.join(LOG_DIR, 'val_runs'))
    
    # Initialise policy kwargs!!!
    policy_kwargs = dict(features_extractor_class = CNNFeatureExtractor, \
        features_extractor_kwargs=dict(num_channels = 2)) #, activation_fn = torch.nn.Tanh)
    
    # Initialise agent
    agent = PPO(CnnPolicy, 
            env, 
            n_epochs = 2, 
            tensorboard_log = LOG_DIR, 
            policy_kwargs = policy_kwargs)
    
    # Initialise learning process with validation 
    callback_train = SaveOnBestTrainingReward(check_freq=10, log_dir=LOG_DIR)

    NUM_TRIALS = 100000
    NUM_VAL_EP = 20 
    best_reward = - float('inf')
    
    for i in range(NUM_TRIALS):    

        print(f"Commencing training")
        agent.learn(10000, 
                    callback = callback_train,
                    tb_log_name = 'train_runs',
                    reset_num_timesteps = False)
        
        # Evaluate agent on validation set
        print(f"Validation results")
        val_reward, val_len = evaluate_agent(val_env, 
                                            agent, 
                                            num_episodes = NUM_VAL_EP)
        
        if val_reward > best_reward:
            best_reward = val_reward
            agent.save(os.path.join(LOG_DIR, 'best_val_model.zip'))
            print(f"Saving new validation reward : {val_reward}")
            
        writer.add_scalar('val_reward', val_reward, i)
        writer.add_scalar('val_len', val_len, i)

    init_obs = env.reset()
    
    #action = agent.predict(init_obs)
    
    print(f"Finished training")
    
    