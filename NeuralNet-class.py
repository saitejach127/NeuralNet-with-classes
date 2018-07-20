import numpy as np

class NeuralNet:
    def __init__(self):
        np.random.seed(1)
        ic = int(input("Input_classes"))
        oc = int(input("Output classes"))
        self.syn0 = 2*np.random.random((ic,25)) - 1
        self.syn1 = 2*np.random.random((25,36)) - 1
        self.syn2 = 2*np.random.random((36,oc)) - 1

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def deriv(self, x):
        return x*(1-x)

    def train(self, inp_data, label, computing):

        for i in range(computing):

            l0 = inp_data
            l1 = self.sigm(inp_data.dot(self.syn0))
            l2 = self.sigm(l1.dot(self.syn1))
            l3 = self.sigm(l2.dot(self.syn2))

            l3_error = label - l3
            l3_delta = l3_error*self.deriv(l3)
            if( i %10000 == 0):
                print(np.mean(l3_error))

            l2_error = l3_delta.dot(self.syn2.T)
            l2_delta = l2_error*self.deriv(l2)

            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error*self.deriv(l1)
            
            self.syn0 += l0.T.dot(l1_delta)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn2 += l2.T.dot(l3_delta)
            

    def predict(self, input_):
        l1 = self.sigm(input_.dot(self.syn0))
        l2 = self.sigm(l1.dot(self.syn1))
        l3 = self.sigm(l2.dot(self.syn2))

        return l3

nn = NeuralNet()


        
        

        
