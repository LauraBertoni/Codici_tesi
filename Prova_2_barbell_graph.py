#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

#%%
def GoogleMatrix(G, alpha):
    A = nx.to_numpy_array(G, dtype=float) 
    n = A.shape[0] 
    A = A.T 
    out_degree = np.zeros((n, 1))
    for i in range(n):
        out_degree[i] = A[:, i].sum()
        if out_degree[i] == 0:  
           out_degree[i] = n 
           A[:, i] = 1
    E =A.copy()
    
    for i in range(n):
        E[:,i] = A[:, i] / out_degree[i]
    
    Google_Matrix = alpha * E + (1 - alpha) / n * np.ones((n, n))
   
    return Google_Matrix

#%%
def crea_pi(A):
    n = A.shape[0]

    somma_colonne = np.sum(A, axis=0)
    è_stocastica = np.allclose(somma_colonne, np.ones(n))
    if not è_stocastica:
        print("La matrice non è stocastica") 

    psi = np.zeros((n**2, n))
    nodi = np.eye(n)
    for j in range(n):
        dx = np.zeros((n,1))
        for k in range(n):
            dx +=  np.sqrt(A[k, j]) * nodi[:,[k]]
        vettore_nodo= nodi[:,[j]]
        psi[:,[j]] = np.kron(vettore_nodo, dx)
    
    Pi = np.dot(psi, psi.T)
    
    return Pi, psi

#%%
def crea_s(n):
    nodi = np.eye(n)
    Swap = np.zeros((n**2, n**2))
    for j in range(n):
        for i in range(n):
            Swap[:, j*n + i]= np.kron(nodi[:,i], nodi[:,j])
    return Swap

#%%
def crea_U(S, Pi, n):
    U_op = S @ (2 * Pi - np.eye(n**2))
    return U_op

#%%
def I_q_media(M, phi_0, U_op, n):
    I_q_i_M = np.zeros(n)
    # L'operatore a 2 step è U^2
    W_op = U_op @ U_op
    phi_t = phi_0.copy()
    
    for _ in range(M):
        matrice_stato = phi_t.reshape((n, n))
        prob_nodi = np.sum(np.abs(matrice_stato)**2, axis=0) 
        I_q_i_M += prob_nodi
        phi_t = W_op @ phi_t
        
    I_q_i_M /= M
    return I_q_i_M

#%% 
G = nx.barbell_graph(7,1, create_using=None)

#%%
# alpha = 0.85
alpha = 0.85
G_matrix = GoogleMatrix(G, alpha)
n = G_matrix.shape[0]

Pi, psi = crea_pi(G_matrix)
Swap = crea_s(n)
U_op = crea_U(Swap, Pi, n)

#%%
# Stato iniziale uniformemente distribuito
psi_0 = np.sum(psi, axis=1, keepdims=True) / np.sqrt(n)

#Calcoliamo l'importanza quantistica media per M passi
M = 500
I_q_M = I_q_media(M, psi_0, U_op, n)

#Calcoliamo il PageRank classico con NetworkX
I_classico = nx.pagerank(G, 0.85)
PageRank_classico = np.array([I_classico[i] for i in sorted(I_classico.keys())])
print("alpha=", alpha)
df = pd.DataFrame({
    "PageRank quantistico": I_q_M,
    "PageRank classico": PageRank_classico
})
print(df)

#%% 
#Stato inziale su nodo 1 (in questa configurazione coincide con il nodo 6)
psi_0 = psi[:,6]
M = 500
I_q_M_psi_1 = I_q_media(M, psi_0, U_op, n)

#Stato iniziale somma pesata nodo 1 e 9
psi_0 = psi[:,6]+ psi[:, 7]
psi_0 = psi_0 / np.sqrt(2)
I_q_M_psi_1_9 = I_q_media(M, psi_0, U_op, n)

df = pd.DataFrame({
    "PageRank quantistico stato iniziale uniformemente distribuito": I_q_M,
    "PageRank quantistico stato iniziale psi_1": I_q_M_psi_1,
    "PageRank quantistico stato iniziale psi_1+psi_9 normalizzato": I_q_M_psi_1_9,
})
print(df)

# %%
