from load_data import VOC_Data_Generator
from model import DeepLabV3
from utils import *
from keras.callbacks import Callback,ModelCheckpoint
model_checkpoint=ModelCheckpoint(filepath="weight.{epoch:02d}.h5",monitor="val_loss",verbose=1,save_best_only=True)
model=DeepLabV3()

train_gen=VOC_Data_Generator(root="/home/sk/VOCdevkit/VOC2012",image_folder="JPEGImages",label_folder="SegmentationClass",image_size=(256,256),train=True,batch_size=4).gen_data()
val_gen=VOC_Data_Generator(root="/home/sk/VOCdevkit/VOC2012",image_folder="JPEGImages",label_folder="SegmentationClass",image_size=(256,256),train=False,batch_size=3).gen_data()

model.fit_generator(train_gen,steps_per_epoch=366,epochs=150,callbacks=[model_checkpoint],validation_data=val_gen,validation_steps=483)