import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np


# create patch embedding 
class Patches(nn.Module):
    def __init__(self, embedding_dim, img_size, patch_size ,n_channels ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size 
        # self.n_patch = (img_size//patch_size) **2
        self.n_patch = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])


        # creating  patches using  conv_layer 
        self.proj = nn.Conv2d(n_channels,embedding_dim,kernel_size=patch_size,stride=patch_size)
        
    

    def forward(self,x):
        B,C,H,W = x.shape
        x = self.proj(x) # (B, embed_dim , n_pathces**0.5, n_pathces *0.5)
        x = x.flatten(2)
        x = x.transpose(1,2) # (B , n_patches ,embedding)
        return x


# create positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, n_patches):
        super().__init__()

        # Learnable positional embedding including class token
        self.positional_embeddings = nn.Parameter(torch.randn(1, n_patches + 1, embedding_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # Fixed here
    
    def forward(self, x):
        batch_size = x.shape[0]
        cls_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # Concatenate class token
        return x + self.positional_embeddings  # Correct the positional embedding size

       

   
#  creating  attention layer 
class AttentionHead(nn.Module):
    def __init__(self, d_model ,head_size): # (embedding , output of attention)
        super().__init__()
        self.head_side = head_size

        # creating Query ,Key ,Value
        self.query = nn.Linear(d_model,head_size)
        self.key = nn.Linear(d_model ,  head_size)
        self.value = nn.Linear(d_model, head_size)

    
    def forward(self , x):
        # Get  Q K V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2,-1)
        attention  = attention/(self.head_side ** 0.5)
        attention = torch.softmax(attention ,dim =1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.full_connected = nn.Linear(d_model ,d_model)
        self.heads = nn.ModuleList([AttentionHead(d_model , self.head_size) for _ in range(n_heads)])


    def forward(self,x):
        # combine attention  heads 
        out = torch.cat([head(x) for head in  self.heads],dim = -1)
        out = self.full_connected(out)
        return out 

# create Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model ,n_heads ,scale_mlp = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # sub-layer 1 normalization 
        self.L1 = nn.LayerNorm(d_model)
        # Multi head Attention
        self.mha = MultiHeadAttention(d_model ,n_heads)
        # Sub-Layer 2 Normalization
        self.L2 = nn.LayerNorm(d_model)

        # Multi layer perceptron 
        self.mlp = nn.Sequential(nn.Linear(d_model ,d_model*scale_mlp),
                                nn.GELU(),
                                nn.Linear(d_model*scale_mlp ,d_model))
        

    def forward(self ,x):
        out = x + self.mha(self.L1(x)) # 1st Residual connection 
        out = out + self.mlp(self.L2(out)) # second Residual connection 

        return out
    

# creating Dataset  
def preprocess_data(data_set_path, batch_size=32, img_size=(32, 32), train_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_set_path, transform=transform)

    # Get the class names
    class_names = dataset.classes  
    num_classes = len(class_names) 

    # Split the dataset
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)) 


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # Include num_workers for parallel loading 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    print("Classes:", class_names)
    print("Number of classes:", num_classes)


    return train_loader, test_loader, class_names, num_classes   # return classes and count


# preprocess_data("/home/bhavit/Desktop/VIT-transformer/Rice_Image_Dataset")
