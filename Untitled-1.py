#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
# %%
G = nx.scale_free_graph(150)

#%%
alpha = 0.85
G_matrix = GoogleMatrix(G, alpha)
n=G_matrix.shape[0]
Pi, psi = crea_pi(G_matrix)
Swap = crea_s(n=G_matrix.shape[0])
U_op = crea_U(Swap, Pi, n=G_matrix.shape[0])
Discriminante = crea_discriminante(G_matrix)

phi_0 = np.sum(psi, axis=1, keepdims=True) / np.sqrt(G_matrix.shape[0])


#%%
# Trovo gli autovalori e i rispettivi autovettori di U 
n = G_matrix.shape[0]
mu_dyn = np.zeros((2*n,), dtype=complex)
mu_autovettori_dyn = np.zeros((n**2, 2*n), dtype=complex)
lambda_autovalore, lambda_autovettre = np.linalg.eig(Discriminante)

A_operatore = crea_A(psi, n)

# Studiamo gli autovalori e autovettori dinamici
i = 0
for autovalore in lambda_autovalore:
    mu_dyn[i] = np.exp(1j * np.arccos(autovalore))
    mu_dyn[i + n] = np.exp(-1j * np.arccos(autovalore))
    autovettore = lambda_autovettre[:, [i]]
    Stato_A = A_operatore @ autovettore 
    Stato_Swap = Swap @ Stato_A
    mu_autovettori_dyn[:, [i]] = Stato_A - mu_dyn[i] * Stato_Swap
    mu_autovettori_dyn[:, [i]] = mu_autovettori_dyn[:, [i]] / np.linalg.norm(mu_autovettori_dyn[:, [i]])
    mu_autovettori_dyn[:, [i+n]] = Stato_A - mu_dyn[i+n] * Stato_Swap
    mu_autovettori_dyn[:, [i+n]] = mu_autovettori_dyn[:, [i+n]] / np.linalg.norm(mu_autovettori_dyn[:, [i+n]])
    i += 1
    print(i)

# %%
# Calcolo della probabilità limite per ciascun nodo
I_q_limite = np.zeros(n)

# Calcolo dei coefficienti c_k
c = np.zeros(2*n, dtype=complex)
for k in range(2*n):
    u_k = mu_autovettori_dyn[:, k] 
    c[k] = np.vdot(u_k, phi_0) 

prob = 0
for i in range(n):
    Importanza = calcola_limite(c, mu_autovettori_dyn, i, mu_dyn, n)
    I_q_limite[i] = Importanza
    prob += Importanza
    print(f"Nodo {i}: {Importanza:.6f}")

# Classifica dei nodi (PageRank Quantistico)
classifica_nodi = np.argsort(I_q_limite)[::-1]
print("\nPageRank quantistico")
nodi_quantistico =np.zeros((n,1))
i =0
for rank, nodo in enumerate(classifica_nodi):
    nodi_quantistico[i]=nodo
    print(f" {rank + 1:2d}: Nodo {nodo:2d} (Q-PageRank = {I_q_limite[nodo]:.6f})")
    i +=1


# Calcolo PageRank classico
I_classico = nx.pagerank(G, alpha)
vector_classico = np.array([I_classico[i] for i in range(n)])

classifica_classica = np.argsort(vector_classico)[::-1]
print("\nPageRank classico")
nodi_classico = np.zeros((n,1))
i =0
for rank, nodo in enumerate(classifica_classica):
    nodi_classico[i]= nodo
    print(f" {rank + 1:2d}: Nodo {nodo:2d} (PR Classico = {vector_classico[nodo]:.6f})")
    i +=1

#%%
# Calcolo della distanza Kendall Tau
from scipy.stats import kendalltau

def distanza_kendalltau(x, y):
    """
    Calcola la distanza di Kendall Tau (frazione di coppie discordanti)
    e il coefficiente tau-a a partire da due liste ordinate di ID nodi.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    
    # Mappa ciascun nodo alla sua posizione (rank) in y
    y_ranks = {val: idx for idx, val in enumerate(y)}
    x_in_y_positions = [y_ranks[val] for val in x]
    
    discordant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x_in_y_positions[i] > x_in_y_positions[j]:
                discordant_pairs += 1
                
    total_pairs = n * (n - 1) / 2
    dist_normalizzata = discordant_pairs / total_pairs
    tau = 1 - 2 * dist_normalizzata
    return tau, dist_normalizzata

# 1. Metodo corretto usando scipy direttamente sui punteggi (consigliato in presenza di pari merito / ties)
tau_scores, _ = kendalltau(I_q_limite, vector_classico)
dist_normalizzata_scores = (1 - tau_scores) / 2

# 2. Metodo corretto usando la funzione custom sulle liste ordinate dei nodi
tau_custom, dist_normalizzata_custom = distanza_kendalltau(nodi_quantistico, nodi_classico)
