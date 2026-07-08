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

# %%
# Calcolo della probabilità limite per ciascun nodo
I_q_limite = np.zeros(n)

# Calcolo dei coefficienti c_k
c = np.zeros(2*n, dtype=complex)
for k in range(2*n):
    u_k = mu_autovettori_dyn[:, k] 
    c[k] = np.vdot(u_k, phi_0) 

# Calcolo della probabilità limite per ciascun nodo e riordinamento dei nodi in base all'importanza quantistica
for i in range(n):
    Importanza = calcola_limite(c, mu_autovettori_dyn, i, mu_dyn, n)
    I_q_limite[i] = Importanza
   
classifica_nodi = np.argsort(I_q_limite)[::-1]
ranking_quantistico =np.zeros((n,1))
nodi_quantistico = np.zeros((n,1))
i =0
for rank, nodo in enumerate(classifica_nodi):
    ranking_quantistico[i]=nodo 
    nodi_quantistico[nodo] = i+1
    i +=1

# Calcolo PageRank classico e riordino dei nodi in base all'importanza classica
I_classico = nx.pagerank(G, alpha)
vector_classico = np.array([I_classico[i] for i in range(n)])

classifica_classica = np.argsort(vector_classico)[::-1]
ranking_classico = np.zeros((n,1))
nodi_classico = np.zeros((n,1))
i =0
for rank, nodo in enumerate(classifica_classica):
    ranking_classico[i]= nodo
    nodi_classico[nodo] = i+1
    i +=1

#%%
# Definizione della distanza Tau di Kendall
def distanza_kendalltau(x, y):
   
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    
    coppie_discordi = 0
    for i in range(n):
        for j in range(i):
            coppia_x = x[i] - x[j]
            coppia_y = y[i] - y[j]
            if coppia_x * coppia_y < 0:
                coppie_discordi += 1
    coppie_totali = n * (n - 1) / 2
    dist_normalizzata = coppie_discordi / coppie_totali
    return coppie_discordi, dist_normalizzata

#%%
#Definzione intersection distance
def intersection_distance(x, y, k):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    dist = 0

    for i in range(k):
        X = set(x[:i+1])
        Y = set(y[:i+1])
        diff = X ^ Y
        dist += len(diff) / (2 * (i + 1))
    
    dist_normalizzata = dist / k
    return dist_normalizzata

#%%
coppie_dis, dist_kendall = distanza_kendalltau(nodi_quantistico, nodi_classico)
dist_intersection_10 = intersection_distance(ranking_quantistico, ranking_classico, 10)
dist_intersection_75 = intersection_distance(ranking_quantistico, ranking_classico, 75)
dist_intersection_150 = intersection_distance(ranking_quantistico, ranking_classico, 150)

print(f"Distanza di Kendall: {dist_kendall}")
print(f"Distanza di intersezione (top 10): {dist_intersection_10}")
print(f"Distanza di intersezione (top 75): {dist_intersection_75}")
print(f"Distanza di intersezione (top 150): {dist_intersection_150}")



