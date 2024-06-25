import torch 
import numpy as np 
import os 
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt 
import pandas as pd 
#from augment import transform 

class CancerDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 img_dir, 
                 seed = 42, 
                 mode = 'train',
                 single_patient = True,
                 patient_idx = 1,            
                 augment = False,
                 give_2d = True):
        
        self.img_dir = img_dir
        self.all_patients = os.listdir(img_dir)
        self.all_patients = [patient for patient in self.all_patients if not(patient.endswith('.DS_Store'))]
        
        #self.exclude_patients = ['P439017', 'P439025', 'P457052', 'P457058', 'P457105', 'P582018', 'P582034']# exclude for training 
        self.mode = mode 
        self.augment = augment 
        self.give_2d = give_2d # whether to give 2d or 3d data for processing 
        self.single_patient = single_patient 
        self.patient_idx = patient_idx 
        
        # Split into train, test, val using random seed 
        num_patients = len(self.all_patients) 
        train_len = int(0.6*num_patients)
        val_len = int(0.1*num_patients)
        test_len = num_patients - (train_len + val_len)
        np.random.seed(seed)
        
        all_idxs = np.random.permutation(len(self.all_patients))
        self.train_idxs = all_idxs[:train_len]
        self.val_idxs = all_idxs[train_len:train_len+val_len]
        self.test_idxs = all_idxs[train_len+val_len:]
        self.all_idxs = {'train' : self.train_idxs, 'val' : self.val_idxs, 'test' : self.test_idxs}
        
        print(f'chicken')
        #print('Train idxs:{}'.format(self.train_idxs))
        # Generate random idxs for train, test, val 
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idxs)
        elif self.mode == 'val':
            return  len(self.val_idxs)
        elif self.mode == 'test':
            return len(self.test_idxs)
        else:
            raise ValueError('Mode not recognised: should be train, test or val')

    
    def __getitem__(self, idx):
        
        if self.single_patient:
            # Use custom idx set for single patient data  
            idx = self.patient_idx

        patient_idx = self.all_idxs[self.mode][idx]
        # print(f'Patient indx : {patient_idx}')
        
        ### Load T2 data:
        img_path = os.path.join(self.img_dir, self.all_patients[patient_idx], 'T2.nii.gz')
        img = self.normalise_data(torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(img_path))))
        img = img.permute(1,2,0)
        
        label_path = os.path.join(self.img_dir, self.all_patients[patient_idx], 'GladTumorGT.nii.gz')
        label = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
        label = label.permute(1,2,0)
        
        patient_name = self.all_patients[patient_idx]
        
        if self.give_2d:
            
            print(f"2D SLICE given")
            # Find centroid of cancer and use this as reference slice idx 
            cancer_label = 1.0*(label >= 2)
            coords = torch.where(cancer_label)
            com_coords = [coord.float().mean().int().item() for coord in coords]
            
            # Obtain z index 
            slice_idx = com_coords[-1]
            img = img[:,:,slice_idx]
            label = label[:,:,slice_idx]
            print('fuecoco')
            
        return img, label, patient_name 
    
    def normalise_data(self, data, mode = 'imgs'):
        
        ### Normalise data between 0 and 1 for t2w, adc, dwi images
        if mode == 'imgs':
            normalised_img = self._normalise(data)
        else:
            # Convert barzel zones to def 2 = 1, def 2 = 1
            barzel_img = self.process_barzel(data) 
                        
            # Normalise between 0 and 1 with 0 being background 
            max_img = 2 # def 2
            min_img = -2 # def 1
            
            normalised_img = ((barzel_img - min_img)/(max_img - min_img)) 

        return normalised_img 
    
        
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        # Test if image is 0s to begin with 
        if torch.max(img) == 0:
            # do not normalise if empty mask -> just return original image to prevent nans
            return img
        
        else:
            max_img = torch.max(img)
            min_img = torch.min(img)

            #Normalise values between 0 to 1
            normalised_img =  ((img - min_img)/(max_img - min_img)) 

            return normalised_img
        
if __name__ == '__main__':
    
    DATASET_PATH = './Data/processed_data'
    
    cancer_ds = CancerDataset(DATASET_PATH,
                              mode = 'train',
                              single_patient = True,
                              patient_idx = 1,
                              give_2d = True)
    
    img, label, patient = cancer_ds[0]
    
    
    print('fuecoco')