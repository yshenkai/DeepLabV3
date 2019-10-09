import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
np.random.seed(100)
class VOC_Data_Generator():
    def __init__(self,root,image_folder,label_folder,image_size,train=True,batch_size=1):
        self.image_size=image_size
        self.root=root
        self.image_name=[]
        self.bacth=batch_size
        if train:
            with open(os.path.join(root,"ImageSets/Segmentation","train.txt")) as f:
                for name in f:
                    self.image_name.append(name.strip('\n'))
        else:
            with open(os.path.join(root,"ImageSets/Segmentation","val.txt")) as f:
                for name in f:
                    self.image_name.append(name.strip('\n'))
        if train:
            np.random.shuffle(self.image_name)
        classes=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,255.]
        self.label_encode=LabelEncoder()
        self.label_encode.fit(classes)
        self.image_folder=image_folder
        self.label_folder=label_folder
    def __len__(self):
        return len(self.image_name)

    def gen_data(self):
        while True:
            image_data=[]
            label_data=[]
            count=0
            for name in self.image_name:
                image_path = os.path.join(self.root, self.image_folder, name + ".jpg")
                label_path = os.path.join(self.root, self.label_folder, name + ".png")
                image = Image.open(image_path)
                label = Image.open(label_path)
                image = image.resize(size=self.image_size,resample=Image.BILINEAR)
                label = label.resize(size=self.image_size,resample=Image.NEAREST)
                #label = self.label_encode.transform(label)
                image=img_to_array(image)
                label=img_to_array(label).reshape((self.image_size[0]*self.image_size[1],))
                image_data.append(image)
                label_data.append(label)
                count+=1
                if count % self.bacth==0:
                    image_array=np.array(image_data)
                    label_array=np.array(label_data).flatten()
                    label_array=self.label_encode.transform(label_array)
                    label_array =to_categorical(label_array,num_classes=22)
                    label_array=label_array.reshape((self.bacth,self.image_size[0]*self.image_size[1],22))
                    yield image_array,label_array
                    image_data=[]
                    label_data=[]
                    count=0






