import numpy as np

class Neoron:
    def __init__(self,w,activation_func='sigmoid',inp=None,truemomentom=False):
        self.w=np.array(w)
        self.b=0
        self.inp=np.array(inp)
        if(activation_func=='sigmoid'):
            self.activation_func=self.sigmoid
            self.grad_activation_func=self.grad_sigmoid
        elif(activation_func=='tanh'):
            self.activation_func=self.tanh
            self.grad_activation_func=self.grad_tanh
        elif(activation_func=='ReLU'):
            self.activation_func=self.relu
            self.grad_activation_func=self.grad_relu
        elif(activation_func=='leakyrelu'):
            self.activation_func=self.leakyrelu
            self.grad_activation_func=self.grad_leakyrelu
        elif(activation_func=='ELU'):
            self.activation_func=self.ELU
            self.grad_activation_func=self.grad_ELU
        else:
            self.activation_func=self.ELU
            self.grad_activation_func=self.grad_ELU

        try:
            self.out=self.activation_func(np.dot(self.w,self.inp.T)+self.b)
        except:
            self.out=None
        self.truemomentom=truemomentom
        self.deltaw=0
        self.deltab=0

    def forward(self):
        # print(np.dot(self.w,self.inp.T),self.b)
        self.out=self.activation_func(np.dot(self.w,self.inp.T)+self.b)
        return self.out

    def backward(self,learning_rate=0.01,next_grad1=[1],next_grad2=1,momentom=0.1):
        if(not self.truemomentom):
            momentom=0
        self.w=self.w-momentom*self.deltaw-np.array(self.inp)*learning_rate*self.out*next_grad1
        self.b=self.b-momentom*self.deltab-learning_rate*self.out*next_grad2
        self.deltaw=momentom*self.deltaw+np.array(self.inp)*learning_rate*self.out*next_grad1
        self.deltab=momentom*self.deltab+learning_rate*self.out*next_grad2
        return [np.array(self.inp)*self.out*next_grad1,self.out*next_grad2]

    def sigmoid(self,x):
        return 1/(1+np.exp(np.array(x)))
    
    def grad_sigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def tanh(self,x):
        return np.tanh(np.array(x))

    def grad_tanh(self,x):
        return 1/((np.cosh(x))**2)

    def relu(self,x):
        return max(0,np.array(x))
    
    def grad_relu(self,x):
        return (np.array(x)>0)*1+0
    
    def leakyrelu(self,x):
        return max(0.1*np.array(x),np.array(x))
    
    def grad_leakyrelu(self,x):
        return (np.array(x)>=0)*1+(np.array(x)<0)*0.1
    
    def ELU(self,x,alpha=0.1):
        return (x>=0)*(x)+(x<0)*(alpha*(np.exp(x)-1))
        
    def grad_ELU(self,x,alpha=0.1):
        return (np.array(x)>=0)*1+(np.array(x)<0)*alpha*np.exp(x)

    def __str__(self):
        return str(self.w)

    def __getitem__(self,i):
        return self.w[i]
        
    def __setitem__(self, i, value):
        self.w[i]=value



if(__name__=="__main__"):
    N=Neoron(np.array([1,2]),'ReLU',np.array([1,2]))
    print(N.backward([1,3],0.1,2))