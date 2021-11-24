import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image



class Image_Caption_Dataset(Dataset):
    def __init__(self,root_dir,df, feature_extractor,tokenizer,max_target_length=512):
        self.root_dir = root_dir
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length=max_target_length
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        #return image
        image_path = self.df['images'][idx]
        text = self.df['text'][idx]
        #prepare image
        image = Image.open(self.root_dir+'/'+image_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        #add captions by encoding the input
        captions = self.tokenizer(text,
                                 padding='max_length',
                                 max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding


def load_dataset(root_dir,csv_path,feature_extractor,tokenizer,max_target_length=512):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = Image_Caption_Dataset(root_dir,train_df,feature_extractor,tokenizer,max_target_length)
    val_dataset = Image_Caption_Dataset(root_dir,val_df,feature_extractor,tokenizer,max_target_length)
    return train_dataset,val_dataset

