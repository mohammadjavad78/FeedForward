from Layer import *
import sys
sys.path.insert(0, './utlis/')
from Read_yaml import *

# learning_rate,batch_size,num_epochs,dataset_path,
# train_split,model_name,pretrained,optimizer_name,
# weight_decay,scheduler_name,step_size,gamma 


import pickle

def unpack(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data1=unpack('./datasets/data_batch_1')[b'data']
label1=unpack('./datasets/data_batch_1')[b'labels']

data2=unpack('./datasets/data_batch_2')[b'data']
label2=unpack('./datasets/data_batch_2')[b'labels']

data3=unpack('./datasets/data_batch_3')[b'data']
label3=unpack('./datasets/data_batch_3')[b'labels']

data4=unpack('./datasets/data_batch_4')[b'data']
label4=unpack('./datasets/data_batch_4')[b'labels']

data5=unpack('./datasets/data_batch_5')[b'data']
label5=unpack('./datasets/data_batch_5')[b'labels']

data1  = np.append(data1,data2,axis=0)
data1  = np.append(data1,data3,axis=0)
data1  = np.append(data1,data4,axis=0)
data1  = np.append(data1,data5,axis=0)
label1 = np.append(label1,label2,axis=0)
label1 = np.append(label1,label3,axis=0)
label1 = np.append(label1,label4,axis=0)
label1 = np.append(label1,label5,axis=0)


# yaml=Getyaml()
# xinp=data1[0].flatten()
# l=Layer(xinp=xinp,numoutp=yaml['model_num_classes'])
# print(l.forward())
# l.backward(xinp,0.001,np.ones_like(xinp))
# print(l.forward())




########################multilayer###########################
# yaml=Getyaml()
# xinp=data1[0].flatten()
# l1=Layer(xinp=xinp,numoutp=16)
# l2=Layer(xinp=l1,numoutp=yaml['model_num_classes'])
# print(l1)
# # print(l2)
# l2.backward(0.001,1)
# # print(l2.forward())
# # l2
# print(l1)
# # print(l2)
# # print(l.forward())

# yaml=Getyaml()
xinp=np.array([-0.1 , 0.2])
l1=Layer(xinp=xinp,numoutp=3,activation_func='tanh')
# print(l1)
l1[0][0]=1
l1[0][1]=2
l1[1][0]=-1
l1[1][1]=1
l1[2][0]=2
l1[2][1]=0
l1[0].b=1
l1[1].b=0
l1[2].b=-1
l1.forward()
print(l1)
# l1.w=
l2=Layer(xinp=l1,numoutp=1,activation_func='tanh')
l2[0][0]=-1
l2[0][1]=1
l2[0][2]=1
l2[0].b=0.5
l2.forward()
print(l2)
# print(l1)
# # print(l2)
l2.backward(0.01,[1],1)
# l2.forward()
# l2
print(l1)
print(l2)
# print(l.forward())
