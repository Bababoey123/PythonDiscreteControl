import numpy as np
import matplotlib.pyplot as plt 

class Plotting:
    def __init__(self):
        return
    def plotAll(self,Log,title):
        ## plotting
        plt.figure()
        plt.plot(Log.t_hist, Log.y_hist)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("position")
        plt.title(title)


        plt.figure()
        plt.plot(Log.t_hist, Log.u_hist)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("control input")
        plt.title(title)

        plt.show()
        
        return
    
    