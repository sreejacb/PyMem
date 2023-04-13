import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


class memristor_models():
    def __init__(self,Roff,Ron,Rint,Amplitude,freq,time_duration,sample_rate,p,j,model):
        self.initial_Roff=Roff
        self.initial_Ron=Ron
        self.Rint=Rint
        self.Amplitude=Amplitude
        self.freq=freq
        self.time_duration=time_duration
        self.time_duration=time_duration
        self.sample_rate=sample_rate
        self.p=p
        self.j=j
        self.model= model
    
    # window functions
    def jog(self,x1,p):
        f_x = 1-(((2*x1)-1)**(2*p))
        return f_x

    def Prodro(self,x1,p,j):
        f_x = j*(1-(((x1-0.5)**2)+0.75**p))
        return f_x
        
    def biolek(self,x1,p,i):
        if i<0:
            i=0
        else:
            i=1
        f_x=1-((x1-i)**p)
        return f_x
    
    def zha(self,x1,p,j,i):
        if i<0:
            i=0
        else:
            i=1
        f_x=j*((1-(0.25*(x1-i)**2)+0.75)**p)
        return f_x
    
    def ideal_model(self):
        start_time = 0
        time = np.arange(start_time, self.time_duration, 1/self.sample_rate)
        sinewave = self.Amplitude * np.sin(2 * np.pi * self.freq * time + 0)
        v_mem=sinewave
        D=10*pow(10,-9)
        uv=10*pow(10,-15)
        delta_R=self.initial_Roff-self.initial_Ron
        x=(self.initial_Roff-self.Rint)/delta_R
        x_t=[]
        i_mem=[]
        x_t.append(x)
        R_mem=[]
        G=[]
        f1=[]
        r_val=(self.initial_Ron*x_t[0])+(self.initial_Roff*(1-x_t[0]))
        R_mem.append(r_val)
        k=(uv*self.initial_Ron)/(D**2)
        i_mem.append(0)
        for i in range(1,len(v_mem)):
            i_val=v_mem[i]/R_mem[i-1]
            i_mem.append(i_val)

            if self.model=='Joglekar':
                f=self.jog(x_t[i-1],self.p)
                f1.append(f)
            if self.model== 'Prodromakis':
                f=self.Prodro(x_t[i-1],self.p,self.j)
                f1.append(f)
            if self.model== 'Biolek':
                f=self.biolek(x_t[i-1],self.p,i_val)
                f1.append(f)
            if self.model== 'Zha':
                f=self.zha(x_t[i-1],self.p,self.j,i_val)
                f1.append(f)
            dx_dt=k*i_mem[i-1]*f
            dx=dx_dt*(time[i-1]-time[i])
            x=dx+x_t[i-1]
            x_t.append(x)
            r_temp=(self.initial_Ron*x)+(self.initial_Roff*(1-x))
            if r_temp<self.initial_Ron:
                r_temp=self.initial_Ron
            if r_temp>self.initial_Roff:
                r_temp=self.initial_Roff
            R_mem.append(r_temp)
            G.append(1/r_temp)
            
        self.Roff=max(R_mem)
        self.Ron=min(R_mem)
    
        return v_mem,i_mem,G,x_t,time,f1

    def plot(self):
        v,curr,G,x,t,f=self.ideal_model()
        plt.plot(v,curr)
        plt.ylabel('i')
        plt.xlabel('v')
        plt.show()
        
    def neural_weight(self,neural_weight,X_max,X_min):
        self.neural_weight=np.array(neural_weight)

        new_min = (1/self.Roff) 
        new_max = (1/self.Ron)

        # new_weights = []
        self.mapped_values = []
        idx = 0
        for item in self.neural_weight:
            self.mapped_values.append([])
            for x in item:
                scaled_x = ((np.abs(x) - X_min) / (X_max - X_min)) * (new_max - new_min) + new_min
                if x<0:
                    scaled_x = scaled_x*-1
                self.mapped_values[idx].append(scaled_x)
            idx += 1
            
        # Iterate over the old weights and biases and compute the new values
        # for weight in neural_weight:
        #     new_weight = ((abs(weight) - X_min) / (X_max - X_min)) * (new_max - new_min) + new_min
        #     new_weight = [(new_weight[i] * -1) if weight[i]<0 else new_weight[i] for i in range(len(new_weight)) ]
        #     new_weights.append(new_weight)

        # self.mapped_values = new_weights  

    def variability(self,partition,variability_percentage_Ron,variability_percentage_Roff):
        v,curr,G,x,t,f = self.ideal_model()
        self.partition = partition
        self.variability_percentage_Ron = variability_percentage_Ron  
        self.variability_percentage_Roff = variability_percentage_Roff

        # partitioning
        self.l2 = []
        self.l2.append(1/self.Roff)
        step = ((1/self.Ron)-(1/self.Roff))/(self.partition-1)
        for i in range(1,self.partition):
            (self.l2).append(self.l2[0]+(step*i))

        # adding variability to the list
        new_Goff = (1/self.Roff)+(((1/self.Roff)*self.variability_percentage_Roff)/100)
        (self.l2).append(new_Goff)
        
        new_Gon = (1/self.Ron)+(((1/self.Ron)*self.variability_percentage_Ron)/100)
        (self.l2).append(new_Gon)
        
        temp = [val for val in self.l2 if (val<=new_Gon and val>=new_Goff)]
        self.l2 = temp
                
        self.l2.sort()
      
    def new_weights(self):
        self.new_values = []
        idx = 0

        def closest(lst, K):
            return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

        for values in self.mapped_values:
            self.new_values.append([])
            for value in values:
                close_val = closest(self.l2,abs(value))
                if value <0:
                    close_val = close_val*-1000
                else:
                    close_val = close_val*1000
                self.new_values[idx].append(close_val)
            idx += 1

        return self.new_values

    def Relative_Error (self):
      Weights_with_var= self.new_weights()
      self.variability(self.partition,0,0)
      Weights_without_var= self.new_weights()
      
      error=[]
      for i, j in zip(np.array(Weights_without_var), np.array(Weights_with_var)):
          l = (np.abs(i-j)/i)
          error.append(l)

      error  = np.array(error) 
      return np.abs(np.sum(error))*100/error.size
    