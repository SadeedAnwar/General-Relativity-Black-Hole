
#%%
import sympy as sp
import math as mt
import numpy as np


#%%
# We need to define an index raising function as well
# The index rasing funcion will be defined by virtue of the metric

def index_u(r , metric , x):

    dimension = len(x)
    
    r_raised = [[0 for i in range(dimension)]for i in range(dimension)]
    metric_mat = sp.Matrix(metric)
    metric_inv = metric_mat.inv()
    

    term = 0
    for kappa in range(dimension):
        for mu in range(dimension):
            term = 0
            for sigma in range(dimension):
                term += metric_inv[mu, sigma]*r[sigma][kappa]

            r_raised[mu][kappa] = term.simplify()

    return r_raised




#%%

# We are gonna have to define a contraction function as well that takes a tensor and then contracts it

def contr_1(r , x):
    dimension = len(x)
    # Here we require a two fold contraction
    r_contr = [[0 for i in range(dimension)]for i in range(dimension)]
    for sigma in range(dimension):
        
        for kappa in range(dimension):
            term = 0
            for rho in range(dimension):
                term += r[rho][sigma][rho][kappa]
        r_contr[sigma][kappa] = term.simplify() 

    return r_contr

#%%

# The second contraction will be tricky as we would need to raise an index before contraction
# Now we define a second contraction

def contr_2(r , x):
    dimension = len(x)
    # Here we require a two fold contraction
    r_contr2 = 0
    term = 0
    
    for sigma in range(dimension):
        term += r[sigma][sigma]

    
    r_contr2 = term.simplify()
    return r_contr2


#%%

# Define a function for calculating the determinant of the metirc
# Here the metric is diagonal so it will suffice to calculate the product of diagonal elements

def det(metric , x):

    determinant = 1
    dimension = len(x)
    for mu in range(dimension):
        for nu in range(dimension):
            if metric[mu][nu] != 0:
                determinant = determinant*(metric[mu][nu])
            else:
                continue
    return determinant


#%%

# A function for calculating Christoffel Symbols

def Christoffel(metric , x):

    metric_mat = sp.Matrix(metric)
    metric_inv = metric_mat.inv()
   
    dimension = len(x)             
   
   
    # Firstly one needs a construction for an empty 4x4x4 matrix or tensor

    Gamma = [[[0 for i in range(dimension)] for i in range (dimension)]for i in range(dimension)]


    # A general christoffel would be given as 
    # Gamma[alpha][mu][nu] = 1/2*(metric_inv[alpha][delta])((sp.diff(metric_mat[mu][delta] , x[nu])) + (sp.diff(metric[nu][delta] , x[mu])) - (sp.diff(metric[mu][nu] , x[delta])))
    # Now we need to devise a clever way of storing these Christoffels
    # for every mu there will be a 2x2 matrix of numbers meaning that every mu would specify a 2x2 matrix of numbers 
    # The next index nu will hence represent a list within the above mentioned matrix
    # Finally the last index alpha will specify a particular entry within that list
    # One should also realize that he is working within the torsion less manifold of space time namely Gamma[alpha][mu][nu] = Gamma[alpha][nu][mu]
    
    for mu in range (dimension):
         for nu in range(dimension):
            for alpha in range(dimension):
                term = 0
                for delta in range (dimension):   
                    term += 1/2*(metric_inv[alpha, delta])*((sp.diff(metric[mu][delta] , x[nu])) + (sp.diff(metric[nu][delta] , x[mu])) - (sp.diff(metric[mu][nu] , x[delta])))
   
                Gamma[alpha][mu][nu] = term.simplify()
                #print(Gamma[alpha][mu][nu])
        
        

    return Gamma 


#%%

# Construct a Riemann tensor with the help of the christoffel function


def Riemann(Cs , x):

    dimension = len(x)

    # We construct a 4x4x4x4 matrix to be filled

    R = [[[[0 for j in range (dimension)]for j in range(dimension)]for j in range(dimension)]for j in range(dimension)]

    # Now a general expression for a Riemann curavtiure tensor will be given as 
    # R[rho][sigma][xi][kappa] = sp.diff(Christoffel(g , x)[rho][sigma][kappa] , xi) - sp.diff(Christoffel(g , x)[rho][sigma][xi] , kappa)
    #  + Christoffel(g , x)[rho][xi][delta]*Christoffel(g,x)[delta][kappa][sigma] - Christoffel(g , x)[rho][kappa][delta]*Christoffel(g,x)[delta][xi][sigma]   
    nz_terms = 0
    for rho in range(dimension):
        for sigma in range(dimension):
            for xi in range(dimension):
                for kappa in range(dimension):
                    term = 0
                    term1 = sp.diff( Cs[rho][sigma][kappa] , x[xi]).simplify()
                    term2 = sp.diff(Cs[rho][sigma][xi] , x[kappa]).simplify()
                    term3 = 0
                    term4 = 0

                    # The 3 and the 4 terms require a summation upon the same indices in both term
                    # In accordance with the Einstein summation notation.
                                        
                    for delta in range(dimension):
                       
                        term3 += (Cs[rho][xi][delta]*Cs[delta][kappa][sigma]).simplify()
                        term4 += (Cs[rho][kappa][delta]*Cs[delta][xi][sigma]).simplify()
                    term += term1 - term2 + term3 - term4  
                    R[rho][sigma][xi][kappa] = term.simplify()
                     #if R[rho][sigma][xi][kappa] != 0:
                        #nz_terms += 1
                    #print(R[rho][sigma][xi][kappa])
                    

                     
    #print(nz_terms)

    return R



#%%

# Now we need a function to calculate the optical curvature
# The optical curvature tensor rather, The optical curavture might be obtained by it's contraction.

def Optical_curvature(r, metric, x):
    dimension = len(x)
    K = [[[[0 for j in range (dimension)]for j in range(dimension)]for j in range(dimension)]for j in range(dimension)]

    for rho in range(dimension):
        for sigma in range(dimension):
            for xi in range(dimension):
                for kappa in range(dimension):
                    term = ((r[rho][sigma][xi][kappa])/det(metric , x)) 
                    K[rho][sigma][xi][kappa] = term.simplify()
                    #if K[rho][sigma][xi][kappa] != 0:
                        #print(K[rho][sigma][xi][kappa])
                    #else:
                        #continue
    
    return K


#%%

# Defining variables with the proper assumptions

r = sp.symbols('r' , positive = True , real = True)
theta = sp.symbols('theta' , positive = True , real = True)
M = sp.symbols('M' , positive = True , real = True)
l = sp.symbols('l' , positive = True, real = True )
t = sp.symbols('t' , positive = True , real = True)
phi = sp.symbols('phi' , positive = True , real = True)




# Constructing a coordinate system, rather specifying it
x = [t,  r , theta , phi]

#x = [r , theta]

kappa = (1 - (2*M)/r)

# The Schwarzchild metric

g = [[-(1 - (2*M)/r + l**2)  ,   0 ,  0 ,  0],
      
      [0 ,  1/(1 - (2*M)/r) , 0 ,  0],
      
      [0 , 0 , r**2 , 0],
      
      [0 , 0 , 0 , (r**2)*((sp.sin(theta))**2) ]]

#%%

# calculating the determinant
det_g = det(g , x)

#%%
#Calculating the christoffel symbols

Gamma = Christoffel(g , x)
print(Gamma)

#%%
# Riemann curvature calculation
R = Riemann(Gamma , x)
print(R)

#%%
#Optical Curvature

K_1 = Optical_curvature(R , g , x)
print(K_1)


# %%
# Contracting the Riemann

R_contr = contr_1(R , x)
print(R_contr)

# %%
# Raising the index

R_contr_u = index_u(R_contr , g , x)
print(R_contr_u)

# %%
OC = contr_2(K_contr_u , x)
print(OC)


# %%
