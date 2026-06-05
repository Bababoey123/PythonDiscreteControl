from Metrics_Plotting import SimLog

import scipy.linalg as la
import control as ct 
import numpy as np
import csv
def as_csv(csv_title:str,logs):
    with open(csv_title+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'output','input']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0,len(logs.t_hist)) :
            writer.writerow({'time': logs.t_hist[i] ,'output': logs.y_hist[i],'input':logs.u_hist[i]})
    return

def Compute_Desired_RST(P_list:list,plant_discrete_tf:ct.TransferFunction):
    """Calculates the apropirate R,S polynomilas from the denominator P, a continuous transfer funtion 

    Args:
        P (ct.TransferFunction): the transfer function containing the desired denominator form
        plant_discrete_tf (ct.TransferFunction): the discret tranfer function of the plant
    """
    #### extracting the polynomials
    A=np.array(plant_discrete_tf.den_list[0][0],dtype=float)
    B=np.array(plant_discrete_tf.num_list[0][0],dtype=float)
    P=np.array(P_list,dtype=float)
    #### checking degrees 
    n=len(A)-1 ## degA
    p=len(P)
    if p < 2*n:
        fill_with_zeros=np.zeros(2*n-p)
        P=np.concatenate([P,fill_with_zeros])
    elif p > 2*n:
        raise ValueError("Degree of P too high ")
    #### building the matrix 
    Sylvester=np.zeros((2*n,2*n))
    for i in range(n):
        Sylvester[i:i+len(A),i]=A ## writes a as collums shifted downward at each i
    B_aligned = np.zeros(len(A))
    B_aligned[-len(B):] = B ; #rewriting B so it has the same lengh as a but with zeros for the higher powers of z
    for i in range(n):
        Sylvester[i:i+len(A), n + i] = B_aligned
    try:
        x = la.solve(Sylvester, P)
    except la.LinAlgError:
        raise ValueError("Sylvester matrix is singular. Check for unstable common pole-zero cancellations!")
    #### extractiong the solutions into R and S
    R_coeffs = x[:n]
    S_coeffs = x[n:]
    R=ct.tf(R_coeffs,[1],plant_discrete_tf.dt)
    S=ct.tf(S_coeffs,[1],plant_discrete_tf.dt)
    #### computing T
    # Evaluate polynomials at z = 1 (sum of coefficients)
    B_1 = np.sum(B)
    Pcl_1 = np.sum(P)
    
    # T dynamic gain scalar
    T_scalar= Pcl_1 / B_1
    T=ct.tf(T_scalar,[1],plant_discrete_tf.dt)
    
    return R,S,T