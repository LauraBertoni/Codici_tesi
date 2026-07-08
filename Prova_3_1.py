#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
import os
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
   
    return Google_Matrix, out_degree

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
G = nx.Graph()
G.add_nodes_from(range(21)) 
G.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                  (1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
                  (6,7),
                  (12,7),(12,8),(12,9),(12,10),(12,11),(12,13),
                  (7,8),(8,9),(9,10),(10,11),(11,13),(13,7),
                  (13,14),
                  (20,14),(20,15),(20,16),(20,17),(20,18),(20,19),
                  (14,15),(15,16),(16,17),(17,18),(18,19),(19,14),
                  (19,5)])

#%%
alpha = 1.0
G_matrix, out_degree = GoogleMatrix(G, alpha)
n=G_matrix.shape[0]
Pi, psi = crea_pi(G_matrix)
Swap = crea_s(n=G_matrix.shape[0])
U_op = crea_U(Swap, Pi, n=G_matrix.shape[0])
Discriminante = crea_discriminante(G_matrix)
phi_0 = np.sum(psi, axis=1, keepdims=True) / np.sqrt(n)


#%%
n = 21
mu = np.zeros((n**2,1), dtype=complex)
mu_autovettori_U = np.zeros((n**2, n**2), dtype=complex)
lambda_autovalore, lambda_autovettre = np.linalg.eigh(Discriminante)

A_operatore = crea_A(psi, n)

# Studiamo prima gli autovalori diversi da 1 e -1
idx = 0
for i, autovalore in enumerate(lambda_autovalore):
    if np.isclose(np.abs(autovalore), 1.0):
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

# Calcoliamo gli autostati relativi a 1 e -1
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
# Calcoliamo il PageRank quantistico al variare dello stato iniziale 
I_q_limite = np.zeros((n,n))

for i in range(21):
    psi_0 = psi[:, [i]]
    c_num = np.zeros(n**2, dtype=complex)
    for k in range(n**2):
        u_k = mu_autovettori_U[:, k] 
        c_num[k] = np.vdot(u_k, psi_0) 
    
    # Normalizziamo il PageRank quantistico calcolato
    for j in range(21):
        I_q_limite[j, i] = calcola_limite(c_num, mu_autovettori_U, j, mu, 21)/out_degree[j]

# %%
#Rappresentiamo graficamente il PageRank normalizzato
N = I_q_limite.shape[0]

# Soglia massima 
soglia_massima = 0.035

cmap = cm.get_cmap('viridis').copy()  
cmap.set_over('white')                

fig, ax = plt.subplots(figsize=(7, 6))

matrice_orientata = I_q_limite.T

im = ax.imshow(
    matrice_orientata, 
    cmap=cmap, 
    origin='lower',  
    vmax=soglia_massima,
    extent=[0.5, N + 0.5, 0.5, N + 0.5]
)

# Configurazione degli assi
ax.set_xlabel('Il nodo di arrivo $j$', fontsize=12)
ax.set_ylabel('Il nodo iniziale $i$', fontsize=12)

# Gestione dei numeri sui bordi 
passo_ticks = [1, 5, 10, 15, 20] if N >= 21 else list(range(1, N + 1))
ax.set_xticks(passo_ticks)
ax.set_yticks(passo_ticks)

cbar = fig.colorbar(im, ax=ax, extend='neither', shrink=0.70)

plt.tight_layout()
#Salvataggio immagine
home_dir = os.path.expanduser("~")
percorso_salvataggio = os.path.join(home_dir, "Desktop", "PR_Q_normalizzato.png")
plt.savefig(percorso_salvataggio, dpi=300)


# %%
# Calcoliamo il PageRank a partire da psi_1
I_q_1 = I_q_limite[:, [0]]  
for i in range(n):
    I_q_1[i]= I_q_1[i]
I_q_1 = np.asarray(I_q_1, dtype=float).flatten()
N = len(I_q_1)
nodi_j = np.arange(1, N + 1, dtype=float)

# Creazione del grafico cartesiano
fig, ax = plt.subplots(figsize=(6, 4))

x1 = nodi_j[:-1]
x2 = nodi_j[1:]
y1 = I_q_1[:-1]
y2 = I_q_1[1:]

segments = np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1)

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(0, I_q_1.max())

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, zorder=2)
lc.set_array(y1) 
ax.add_collection(lc)

ax.scatter(nodi_j, I_q_1, c=I_q_1, cmap=cmap, norm=norm, s=25, edgecolor='none', zorder=3)

ax.set_xlabel('Il nodo di arrivo $j$', fontsize=11)
ax.set_ylabel(r'Quantum PageRank normalizzato ', fontsize=11)

ax.set_xlim(1, N)
ax.set_ylim(0, I_q_1.max() * 1.2)

passo_ticks = [1, 5, 10, 15, 20] if N >= 21 else list(range(1, int(N) + 1))
ax.set_xticks(passo_ticks)

ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=0.8)
ax.tick_params(axis='x', which='minor', length=3)
ax.tick_params(axis='x', which='major', length=5)

ax.set_axisbelow(True)
ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

ax.set_frame_on(True)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# Salvataggio sul Desktop
plt.tight_layout()
home_dir = os.path.expanduser("~")
percorso_salvataggio = os.path.join(home_dir, "Desktop", "grafico_vettore_page_rank_gradiente_1.png")
plt.savefig(percorso_salvataggio, dpi=300)


# %%
# Calcoliamo il PageRank a partire da psi_13
I_q_13 = I_q_limite[:, [12]]  
for i in range(n):
    I_q_13[i]= I_q_13[i]
I_q_13 = np.asarray(I_q_13, dtype=float).flatten()
N = len(I_q_13)
nodi_j = np.arange(1, N + 1, dtype=float)

# Creazione del grafico cartesiano
fig, ax = plt.subplots(figsize=(6, 4))

x1 = nodi_j[:-1]
x2 = nodi_j[1:]
y1 = I_q_13[:-1]
y2 = I_q_13[1:]

segments = np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1)

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(0, I_q_13.max())

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, zorder=2)
lc.set_array(y1) 
ax.add_collection(lc)

ax.scatter(nodi_j, I_q_13, c=I_q_13, cmap=cmap, norm=norm, s=25, edgecolor='none', zorder=3)

ax.set_xlabel('Il nodo di arrivo $j$', fontsize=11)
ax.set_ylabel(r'Quantum PageRank normalizzato ', fontsize=11)

ax.set_xlim(1, N)
ax.set_ylim(0, I_q_13.max() * 1.2)

passo_ticks = [1, 5, 10, 15, 20] if N >= 21 else list(range(1, int(N) + 1))
ax.set_xticks(passo_ticks)

ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=0.8)
ax.tick_params(axis='x', which='minor', length=3)
ax.tick_params(axis='x', which='major', length=5)

ax.set_axisbelow(True)
ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

ax.set_frame_on(True)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# Salvataggio sul Desktop
plt.tight_layout()
home_dir = os.path.expanduser("~")
percorso_salvataggio = os.path.join(home_dir, "Desktop", "grafico_vettore_page_rank_gradiente_13.png")
plt.savefig(percorso_salvataggio, dpi=300)

# %%
# Calcoliamo il PageRank a partire da psi_21
I_q_21 = I_q_limite[:, [20]]  
for i in range(n):
    I_q_21[i]= I_q_21[i]
I_q_21 = np.asarray(I_q_21, dtype=float).flatten()
N = len(I_q_21)
nodi_j = np.arange(1, N + 1, dtype=float)

# Creazione del grafico cartesiano
fig, ax = plt.subplots(figsize=(6, 4))

x1 = nodi_j[:-1]
x2 = nodi_j[1:]
y1 = I_q_21[:-1]
y2 = I_q_21[1:]

segments = np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1)

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(0, I_q_21.max())

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, zorder=2)
lc.set_array(y1) 
ax.add_collection(lc)

ax.scatter(nodi_j, I_q_21, c=I_q_21, cmap=cmap, norm=norm, s=25, edgecolor='none', zorder=3)

ax.set_xlabel('Il nodo di arrivo $j$', fontsize=11)
ax.set_ylabel(r'Quantum PageRank normalizzato ', fontsize=11)

ax.set_xlim(1, N)
ax.set_ylim(0, I_q_21.max() * 1.2)

passo_ticks = [1, 5, 10, 15, 20] if N >= 21 else list(range(1, int(N) + 1))
ax.set_xticks(passo_ticks)

ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=0.8)
ax.tick_params(axis='x', which='minor', length=3)
ax.tick_params(axis='x', which='major', length=5)

ax.set_axisbelow(True)
ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

ax.set_frame_on(True)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# Salvataggio sul Desktop
plt.tight_layout()
home_dir = os.path.expanduser("~")
percorso_salvataggio = os.path.join(home_dir, "Desktop", "grafico_vettore_page_rank_gradiente_21.png")
plt.savefig(percorso_salvataggio, dpi=300)


# %%
# Calcolo l'importanza media con M = 30
M = 30
I_media = np.zeros((n,n))
for i in range(n):
    phi_0 = psi[:, [i]]
    I_media[:, i] = I_q_media(M, phi_0, U_op, n)
for j in range(n):
    I_media[j, :] = I_media[j, :] / out_degree[j]
# %%

N = I_media.shape[0]

matrice_orientata = I_media.T
soglia_massima = 0.035

cmap = cm.get_cmap('viridis').copy()  
cmap.set_over('white')                

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    matrice_orientata, 
    cmap=cmap, 
    origin='lower',  # Mette il nodo 1 in basso a sinistra
    vmax=soglia_massima,
    extent=[0.5, N + 0.5, 0.5, N + 0.5]
)

# Configurazione degli assi
ax.set_xlabel('Il nodo di arrivo $j$', fontsize=12)
ax.set_ylabel('Il nodo iniziale $i$', fontsize=12)

# Gestione dei numeri sui bordi 
passo_ticks = [1, 5, 10, 15, N] if N >= 21 else list(range(1, N + 1))
ax.set_xticks(passo_ticks)
ax.set_yticks(passo_ticks)

cbar = fig.colorbar(im, ax=ax, extend='neither', shrink=0.70)

plt.tight_layout()
plt.tight_layout()
plt.savefig('I_q_media_30.png', dpi=300)

# %%
# Calcolo l'importanza media con M = 50
M = 50
I_media_50 = np.zeros((n,n))
for i in range(n):
    phi_0 = psi[:, [i]]
    I_media_50[:, i] = I_q_media(50, phi_0, U_op, n)
for j in range(n):
    I_media_50[j, :] = I_media_50[j, :] / out_degree[j]
# %%

N = I_media_50.shape[0]

matrice_orientata = I_media_50.T

soglia_massima = 0.035

cmap = cm.get_cmap('viridis').copy()  
cmap.set_over('white')                

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    matrice_orientata, 
    cmap=cmap, 
    origin='lower',  # Mette il nodo 1 in basso a sinistra
    vmax=soglia_massima,
    extent=[0.5, N + 0.5, 0.5, N + 0.5]
)

# Configurazione degli assi
ax.set_xlabel('Il nodo di arrivo $j$', fontsize=12)
ax.set_ylabel('Il nodo iniziale $i$', fontsize=12)

# Gestione dei numeri sui bordi 
passo_ticks = [1, 5, 10, 15, N] if N >= 21 else list(range(1, N + 1))
ax.set_xticks(passo_ticks)
ax.set_yticks(passo_ticks)

cbar = fig.colorbar(im, ax=ax, extend='neither', shrink=0.70)

plt.tight_layout()
plt.tight_layout()
plt.savefig('I_q_media_50.png', dpi=300)

# %%
# Calcolo l'importanza media con M = 100
M = 100
I_media_100 = np.zeros((n,n))
for i in range(n):
    phi_0 = psi[:, [i]]
    I_media_100[:, i] = I_q_media(100, phi_0, U_op, n)
for j in range(n):
    I_media_100[j, :] = I_media_100[j, :] / out_degree[j]
# %%
N = I_media_100.shape[0]

matrice_orientata = I_media_100.T

soglia_massima = 0.035

cmap = cm.get_cmap('viridis').copy()  
cmap.set_over('white')                

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    matrice_orientata, 
    cmap=cmap, 
    origin='lower',  # Mette il nodo 1 in basso a sinistra
    vmax=soglia_massima,
    extent=[0.5, N + 0.5, 0.5, N + 0.5]
)

# Configurazione degli assi
ax.set_xlabel('Il nodo di arrivo $j$', fontsize=12)
ax.set_ylabel('Il nodo iniziale $i$', fontsize=12)

# Gestione dei numeri sui bordi 
passo_ticks = [1, 5, 10, 15, N] if N >= 21 else list(range(1, N + 1))
ax.set_xticks(passo_ticks)
ax.set_yticks(passo_ticks)

cbar = fig.colorbar(im, ax=ax, extend='neither', shrink=0.70)

plt.tight_layout()
plt.tight_layout()
plt.savefig('I_q_media_30.png', dpi=300)
# %%
G = nx.karate_club_graph()
print("Node Degree")
for v in G:
    print(f"{v:4} {G.degree(v):6}")

nx.draw_circular(G, with_labels=True)
plt.show()