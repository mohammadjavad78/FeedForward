from Neoron import *
import random


class Layer:

    def __init__(self,xinp=[1,1,1],numoutp=5,mu=0,sigma=1,bias=0,activation_func='leakyrelu'):
        self.dim=numoutp
        self.Neurons=[]
        self.middle=False
        self.last=xinp
        if(type(xinp)==type(np.array([]))):
            self.xinp=xinp
        else:
            self.middle=True
            self.xinp=xinp.outp
        self.outp=[]
        for i in range(self.dim):
            self.Neurons.append(Neoron(w=np.random.normal(mu, sigma, len(self.xinp))+bias,activation_func=activation_func,inp=self.xinp,truemomentom=False))
            self.outp.append(self.Neurons[-1].forward())
        self.outp=np.array(self.outp)

    def forward(self):
        if(self.middle):
            self.last.forward()
        self.outp=[]
        for i in self.Neurons:
            i.inp=self.xinp
            self.outp.append(i.forward())
        self.outp=np.array(self.outp)
        return self.outp

            
    def totalbackward(self,learning_rate,next_grad1,next_grad2):
        for i in range(len(self.Neurons)):
            print(next_grad1,next_grad2,'nnnnnnnnnnnnnnn')
            grad1,grad2=self.Neurons[i].backward(learning_rate,next_grad1[i],next_grad2)
            if(type(self.last)!=type(np.array([]))):
                self.last.totalbackward(learning_rate,grad1,grad2)
        

    def backward(self,learning_rate,next_grad1,next_grad2):
        self.totalbackward(learning_rate,next_grad1,next_grad2)
        self.forward()


    def __str__(self):
        string=""
        for i in range(self.dim):
            string+=f"Neoron{i} => inp:{str(self.Neurons[i].inp)} w:{str(self.Neurons[i].w)} b:{str(self.Neurons[i].b)} out:{self.Neurons[i].out}\n"
        return string


    def __getitem__(self,item):
        return self.Neurons[item]    


    def __setitem__(self,item,value):
        self.Neurons[item]=value  
            

# if(__name__=="__main__"):
#     L=Layer()
#     print(L)
#     L.xinp=np.array([2,2,2])
#     L.forward()
#     print(L)
#     L.backward([1,1,1,1,1],0.1,[1,1,1,1,1])
#     L.forward()
#     print(L)