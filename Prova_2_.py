#%%
import numpy as np
import networkx as nx
import pandas as pd
from scipy.linalg import null_space
np.set_printoptions(precision=6, suppress=True)

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
def crea_discriminante(G):
    n = G.shape[0]
    D = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            D[j,k ]= np.sqrt(G[j, k] * G[k, j])
    return D

#%%
def crea_A(psi, n):
    A_operator = np.zeros((n**2, n))
    
    for j in range(n):
        A_operator[:, j] = psi[:, j]
        
    return A_operator

#%%
def calcola_limite(coeff, autovettori, i, autovalori, n):

    importanza = 0
    autovalori_sq = np.squeeze(autovalori) ** 2
    
    num_coeff = len(coeff)
    for j in range(num_coeff):
        for k in range(num_coeff):
            if np.abs(autovalori_sq[j] - autovalori_sq[k]) < 1e-9:
                M_j = autovettori[:, j].reshape((n, n))
                M_k = autovettori[:, k].reshape((n, n))
                
                braket = np.vdot(M_k[:, i], M_j[:, i])
                
                importanza += coeff[j] * np.conj(coeff[k]) * braket
    return np.real(importanza)

#%% 
G = nx.DiGraph()
G.add_nodes_from(range(6)) 
G.add_edges_from([(0,1),(0,2),(0,3),(0,4),
                  (1,5), (2,5), (3,5), (4,5)])
                  

#%%
# alpha = 0.85
alpha = 0.85
G_matrix = GoogleMatrix(G, alpha)
n = G_matrix.shape[0]

Pi, psi = crea_pi(G_matrix)
Swap = crea_s(n)
U_op = crea_U(Swap, Pi, n)
Discriminante = crea_discriminante(G_matrix)


#%%
n = 6
mu = np.zeros((n**2,1), dtype=complex)
mu_autovettori_U = np.zeros((n**2, n**2), dtype=complex)
lambda_autovalore, lambda_autovettre = np.linalg.eigh(Discriminante)

A_operatore = crea_A(psi, n)

# Studiamo prima quelli diversi da 1 e -1
idx = 0
for i, autovalore in enumerate(lambda_autovalore):
    if np.isclose(np.abs(autovalore), 1.0):
        # Saltiamo in questa fase, verranno pescati dal null_space di U +- I
        continue
        
    mu[idx] = np.exp(1j * np.arccos(autovalore))
    mu[idx + 1] = np.exp(-1j * np.arccos(autovalore))
    
    autovettore = lambda_autovettre[:, [i]]
    Stato_A = A_operatore @ autovettore 
    Stato_Swap = Swap @ Stato_A
    
    v1 = Stato_A - mu[idx] * Stato_Swap
    v1 = v1 / np.linalg.norm(v1)
    
    v2 = Stato_A - mu[idx + 1] * Stato_Swap
    v2 = v2 / np.linalg.norm(v2)
    
    mu_autovettori_U[:, [idx]] = v1
    mu_autovettori_U[:, [idx + 1]] = v2
    
    idx += 2

# 1. Creiamo la matrice Identità della stessa dimensione di U
I = np.eye(n**2)
autovettori_di_uno = null_space(U_op - I)
num_1 = autovettori_di_uno.shape[1]
for i in range(num_1):
    mu[idx] = 1.0
    mu_autovettori_U[:, idx] = autovettori_di_uno[:, i]
    idx += 1

autovettori_di_meno_uno = null_space(U_op + I)
num_meno_1 = autovettori_di_meno_uno.shape[1]
for i in range(num_meno_1):
    mu[idx] = -1.0
    mu_autovettori_U[:, idx] = autovettori_di_meno_uno[:, i]
    idx += 1
#%%
# Stato iniziale uniformemente distribuito
psi_0 = np.sum(psi, axis=1, keepdims=True) / np.sqrt(n)
I_q = np.zeros(n)
coeff = np.zeros(n**2, dtype=complex)
for k in range(2*n):
    u_k = mu_autovettori_U[:, k] 
    coeff[k] = np.vdot(u_k, psi_0) 

# Calcolo della probabilità limite per ciascun nodo 
for i in range(n):
    Importanza = calcola_limite(coeff, mu_autovettori_U, i, mu, n)
    I_q[i] = Importanza

# Stato iniziale sul nodo 2
psi_0 = psi[:, 1]
I_q_2 = np.zeros(n)
coeff = np.zeros(n**2, dtype=complex)
for k in range(2*n):
    u_k = mu_autovettori_U[:, k] 
    coeff[k] = np.vdot(u_k, psi_0) 

# Calcolo della probabilità limite per ciascun nodo 
for i in range(n):
    Importanza = calcola_limite(coeff, mu_autovettori_U, i, mu, n)
    I_q_2[i] = Importanza

# Stato iniziale uniformemente distirbuito sui nodi 2, 3, 4 e 5
psi_0 = psi[:, 1] + psi[:,2] + psi[:,3] + psi[:,4]
psi_0 = psi_0 / 2
I_q_2_3_4_5 = np.zeros(n)
coeff = np.zeros(n**2, dtype=complex)
for k in range(2*n):
    u_k = mu_autovettori_U[:, k] 
    coeff[k] = np.vdot(u_k, psi_0) 

# Calcolo della probabilità limite per ciascun nodo 
for i in range(n):
    Importanza = calcola_limite(coeff, mu_autovettori_U, i, mu, n)
    I_q_2_3_4_5[i] = Importanza

df = pd.DataFrame({
    "PageRank quantistico stato iniziale uniformemente distribuito": I_q,
    "PageRank quantistico stato iniziale psi_2": I_q_2,
    "PageRank quantistico stato iniziale uniformemente distribuito sui nodi 2, 3, 4 e 5": I_q_2_3_4_5,
})
print(df)


# %%
