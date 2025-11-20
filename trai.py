from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(root_dir,label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, index):
        img_name=self.img_path[index]
        img_item_path=os.path.join(self.path,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir="hymenoptera_data\\hymenoptera_data\\train"
ants_dir="ants"
bees_dir="bees"
ant_dataset=MyData(root_dir,ants_dir)
bees_dataset=MyData(root_dir,bees_dir)
train_dataset=ant_dataset+bees_dataset
print(len(train_dataset))

writer=SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()