import plotly.graph_objects as go
import numpy as np
import complete_ross as fct
import ross as rs
from scipy import linalg as la
import matplotlib.pyplot as plt


material_exo = rs.Material(name="mat_exo", rho=7850, E=2.05 * 10**11, Poisson=0.3)


Low_pressure_elements = []
low_pressure_seal_elements = []
low_pressure_elements = []

for i in range(8) : 
    Low_pressure_elements.append(rs.ShaftElement(L = 0.1, material=material_exo, n=i, idl=0, odl=0.05))

low_pressure_elements.append(
    rs.DiskElement.from_geometry(n=1, material=material_exo, width=0.02, i_d=0.05, o_d=0.257)
)

low_pressure_elements.append(
    rs.DiskElement.from_geometry(n=7, material=material_exo, width=0.02, i_d=0.05, o_d=0.3)
)

low_pressure_seal_elements.append(rs.BearingElement(n=0, kxx=3e7, kyy=3e7, cxx=0, cyy=0))
low_pressure_seal_elements.append(rs.BearingElement(n=8, kxx=3e7, kyy=3e7, cxx=0, cyy=0))
# low_pressure_seal_elements.append(rs.BearingElement(n=6, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0))
rotor_lp = rs.Rotor(
    shaft_elements=Low_pressure_elements,
    bearing_elements=low_pressure_seal_elements,
    disk_elements=low_pressure_elements,
)
hight_pressure_elements = []
hight_pressure_seal_elements = []
hight_pressure_disk_elements = []

for i in range(4) : 
    hight_pressure_elements.append(rs.ShaftElement(L = 0.1, material=material_exo, n=i, idl=0.07, odl=0.08))

hight_pressure_disk_elements.append(
    rs.DiskElement.from_geometry(n=2, material=material_exo, width=0.015, i_d=0.08, o_d=0.46)
)

hight_pressure_seal_elements.append(rs.BearingElement(n=0, kxx=2.5e7, kyy=2.5e7, cxx=0, cyy=0))
# hight_pressure_seal_elements.append(rs.BearingElement(n=4, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0))
rotor_hp = rs.Rotor(
    shaft_elements=hight_pressure_elements,
    bearing_elements=hight_pressure_seal_elements,
    disk_elements=hight_pressure_disk_elements,
)


axial_shaft = [rs.ShaftElement(0.1, 0, 0.05, material=material_exo) for _ in range(8)]


coaxial_shaft = [rs.ShaftElement(0.1, 0.07, 0.08, material=material_exo) for _ in range(4)]


disks = [
    rs.DiskElement.from_geometry(n=1, material=material_exo, width=0.02, i_d=0.05, o_d=0.257),
    rs.DiskElement.from_geometry(n=7, material=material_exo, width=0.02, i_d=0.05, o_d=0.3),
    rs.DiskElement.from_geometry(n=11, material=material_exo, width=0.015, i_d=0.08, o_d=0.46)
]

bearings = [
    rs.BearingElement(0, kxx=3*10**7, kyy=3*10**7, cxx=0),
    rs.BearingElement(8, kxx=3*10**7, kyy=3*10**7, cxx=0),
    rs.BearingElement(9, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0),
    rs.BearingElement(6, n_link=13, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0)
]

shaft = [axial_shaft, coaxial_shaft]
rotor_coaxial = rs.CoAxialRotor(shaft, disks, bearings)


# mode = fct.get_mode(rotor_coaxial,0,num_modes=28)
# eigenvector = mode["evectors"]
# eigenvector = eigenvector.imag
# eigenvalue = (mode["wd"])/(2*np.pi)
# print(eigenvalue)
A = fct.get_A(rotor_coaxial, speed=0)
eigenvalue, eigenvector = la.eig(A)
f = np.imag(eigenvalue) / (2 * np.pi)
f = np.abs(f)  # Prendre la valeur absolue des fréquences pour le tri
pos = np.argsort(f)
f = f[pos]
eigenvector = eigenvector[:, pos]
eigenvector = eigenvector.imag
# M = rotor_coaxial.M()
# K = rotor_coaxial.K(0)
# G_lp = rotor_lp.G()
# # print("Gyrioscopique Low pressure \t",G_lp.shape)
# G_hp = rotor_hp.G() * 1.5
# # print("Gyrioscopique High pressure \t",G_hp.shape)
# G1 = np.hstack((G_lp, np.zeros((G_lp.shape[0], rotor_coaxial.ndof - G_lp.shape[1]))))
# G2 = np.hstack((np.zeros((G_hp.shape[0],rotor_coaxial.ndof - G_hp.shape[1])), G_hp))

# G = np.vstack((G1, G2))

# B = np.block([
#     [np.zeros_like(M), M],
#     [M, np.zeros_like(M)]
# ])

# A = np.block([
#     [K, np.zeros_like(M)],
#     [np.zeros_like(M), -M]
# ])
# from scipy.linalg import eig
# # Résolution du problème aux valeurs propres généralisé
# eigenvalues, X = eig(A, B)

# # Fréquences en Hz
# f = np.imag(eigenvalues) / (2 * np.pi)
# f = np.abs(f)  # Prendre la valeur absolue des fréquences pour le tri
# # Tri des fréquences et des vecteurs propres associés
# pos = np.argsort(f)
# f = f[pos]
# print(f)
# eigenvector = X[:, pos]
# # mode = fct.get_mode(rotor_coaxial,0)
# # eigenvector = mode["evectors"]
# eigenvector = eigenvector.imag


# Dimensions of the shafts
DextOm1 = 50e-3  # [m], diameter of the low pressure shaft
DintOm2 = 70e-3  # [m], inner diameter of the high pressure shaft
DextOm2 = 80e-3  # [m], outer diameter of the high pressure shaft

# Geometry
node_coordinates = np.array([
    [0, 0, 0],                 # node 1
    [0, 0, 10],
    [0, 0, 20],
    [0, 0, 30],
    [0, 0, 40],
    [0, 0, 50],
    [0, 0, 60],
    [0, 0, 70],
    [0, 0, 80],                # node 9
    [-(DextOm2 + DintOm2) / 2, 0, 20],  # node 10
    [-(DextOm2 + DintOm2) / 2, 0, 30],
    [-(DextOm2 + DintOm2) / 2, 0, 40],
    [-(DextOm2 + DintOm2) / 2, 0, 50],
    [-(DextOm2 + DintOm2) / 2, 0, 60]   # node 14
])
node_coordinates[:, 2] *= 1e-2  # Convert to meters

# Element connections between nodes
element_nodes = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
    [10, 11], [11, 12], [12, 13], [13, 14]
])

# Beams (low pressure and high pressure shafts)
number_beam_elements = element_nodes.shape[0]

# Initialize beam element locations with two nodes having four degrees of freedom each
loc_beam_elements = np.zeros((number_beam_elements, 2 * 4))

# Index and amplification factors
# index_f = np.linspace(1, 28, 28, dtype=int)  # Mode index
# # Initial array
# n = [ 100,  300,  300,  600, 1000, 2000, 2000, 2000, 2000, 2000, 2000,
#        2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
#        2000, 2000, 2000, 2000, 2000, 2000]
print("pokklpkl")
index_f = np.arange(1, 28, 4, dtype=int)  # Mode index
n = [ 100,  300,  300,  600, 1000, 2000, 2000, 2000]

# Définir le nombre de lignes et de colonnes pour les sous-plots
nrows = 4
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

# Aplatir la grille des axes pour un accès facile
axes = axes.flatten()

# Boucle sur les index_f pour tracer les sous-plots
for j in range(len(index_f)):
    ax = axes[j]  # Accéder au subplot spécifique

    # Boucle sur les éléments de poutre
    for i in range(number_beam_elements):
        index = element_nodes[i, :]  # Ajustement pour l'indexation Python (0-based)
        loc_beam_elements[i, :] = [4 * index[0] - 4, 4 * index[0] - 3, 4 * index[1] - 4, 4 * index[1] - 3,
                                   4 * index[0] - 2, 4 * index[0] - 1, 4 * index[1] - 2, 4 * index[1] - 1]

        x1, y1, z1 = node_coordinates[index[0] - 1]
        x2, y2, z2 = node_coordinates[index[1] - 1]
        x1_def = x1 + n[j] * eigenvector[int(loc_beam_elements[i, 0]), index_f[j] - 1]
        y1_def = y1 + n[j] * eigenvector[int(loc_beam_elements[i, 4]), index_f[j] - 1]
        z1_def = z1
        x2_def = x2 + n[j] * eigenvector[int(loc_beam_elements[i, 2]), index_f[j] - 1]
        y2_def = y2 + n[j] * eigenvector[int(loc_beam_elements[i, 6]), index_f[j] - 1]
        z2_def = z2

        # Tracé des poutres originales et déformées en fonction des conditions
        if i < 8:
            ax.plot([z1, z2], [x1, x2], '-o', color=[0.7, 0.7, 0.7], linewidth=2)  # original
            ax.plot([z1_def, z2_def], [x1_def, x2_def], '-o', color='red', linewidth=2)  # déformé
        else:
            ax.plot([z1, z2], [x1, x2], '-o', color=[0.7, 0.7, 0.7], linewidth=2)  # original
            ax.plot([z1, z2], [x1 + (DextOm2 + DintOm2), x2 + (DextOm2 + DintOm2)], 
                    '-o', color=[0.7, 0.7, 0.7], linewidth=2)  # original décalé
            ax.plot([z1_def, z2_def], [x1_def, x2_def], '-o', color='blue', linewidth=2)  # déformé
            ax.plot([z1_def, z2_def], [x1_def + (DextOm2 + DintOm2), x2_def + (DextOm2 + DintOm2)], 
                    '-o', color='blue', linewidth=2)  # déformé décalé

    ax.set_xlabel("Z-coordinate [m]", fontsize=12)
    ax.set_ylabel("X-coordinate [m]", fontsize=12)
    ax.set_title(f"Beam Deformation for index_f[{j}]", fontsize=14)
    ax.grid(True)

# Ajustement de l'espacement des sous-plots
plt.tight_layout()

# Enregistrer la figure complète
plt.savefig("mode_test/beam_deformation_all.png", format='png')
plt.show()

##############################################################################
from plotly.subplots import make_subplots

# Matrices et valeurs d'exemple
A = fct.get_A(rotor_coaxial, speed=0)
eigenvalue, eigenvector = la.eig(A)
f = np.imag(eigenvalue) / (2 * np.pi)
f = np.abs(f)
pos = np.argsort(f)
f = f[pos]
eigenvector = eigenvector[:, pos]
eigenvector = eigenvector.imag

# Dimensions et géométrie des poutres
DextOm1 = 50e-3  
DintOm2 = 70e-3  
DextOm2 = 80e-3  

node_coordinates = np.array([
    [0, 0, 0],
    [0, 0, 10],
    [0, 0, 20],
    [0, 0, 30],
    [0, 0, 40],
    [0, 0, 50],
    [0, 0, 60],
    [0, 0, 70],
    [0, 0, 80],
    [-(DextOm2 + DintOm2) / 2, 0, 20],
    [-(DextOm2 + DintOm2) / 2, 0, 30],
    [-(DextOm2 + DintOm2) / 2, 0, 40],
    [-(DextOm2 + DintOm2) / 2, 0, 50],
    [-(DextOm2 + DintOm2) / 2, 0, 60]
])
node_coordinates[:, 2] *= 1e-2  

element_nodes = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
    [10, 11], [11, 12], [12, 13], [13, 14]
])

# Définition du nombre d'éléments de poutre
number_beam_elements = element_nodes.shape[0]
loc_beam_elements = np.zeros((number_beam_elements, 2 * 4))

# Index et amplification pour les déformations
index_f = np.arange(1, 28, 4, dtype=int)
n = [100, 300, 300, 600, 1000, 2000, 2000, 2000]

# Créer les sous-plots
fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=[f"Mode shape {j}" for j in range(len(index_f))],
    vertical_spacing=0.15
)

# Boucle sur les index_f pour créer les tracés dans chaque sous-plot
for j in range(len(index_f)):
    row = (j // 2) + 1  # Calculer la ligne du sous-plot
    col = (j % 2) + 1   # Calculer la colonne du sous-plot
    
    # Boucle sur les éléments de poutre
    for i in range(number_beam_elements):
        index = element_nodes[i, :]
        loc_beam_elements[i, :] = [
            4 * index[0] - 4, 4 * index[0] - 3, 4 * index[1] - 4, 4 * index[1] - 3,
            4 * index[0] - 2, 4 * index[0] - 1, 4 * index[1] - 2, 4 * index[1] - 1
        ]
        
        x1, _, z1 = node_coordinates[index[0] - 1]
        x2, _, z2 = node_coordinates[index[1] - 1]
        
        x1_def = x1 + n[j] * eigenvector[int(loc_beam_elements[i, 0]), index_f[j] - 1]
        x2_def = x2 + n[j] * eigenvector[int(loc_beam_elements[i, 2]), index_f[j] - 1]
        
        # Tracé des poutres originales et déformées
        color = 'red' if i < 8 else 'blue'
        
        fig.add_trace(go.Scatter(
            x=[z1, z2], y=[x1, x2],
            mode='lines+markers',
            line=dict(color='grey', width=2),
            marker=dict(size=6),
            name='Original' if i == 0 else "",
            showlegend=(j == 0),
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[z1, z2], y=[x1_def, x2_def],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=6),
            name='Déformé' if i == 0 else "",
            showlegend=(j == 0),
        ), row=row, col=col)

# Mise en page pour appliquer la police, taille, et grille
fig.update_layout(
    template='plotly_white',
    font=dict(family="Computer Modern", size=22, color='black'),
    height=1000,
    width=1000,
    xaxis_title='Z-coord. [m]',
    yaxis_title='X-coord. [m]',
    showlegend=False,
)

# Mettre à jour la taille de la police des sous-titres
fig.for_each_annotation(lambda a: a.update(font=dict(size=22)))
fig.update_xaxes(title_font=dict(size=22))
fig.update_yaxes(title_font=dict(size=22))

# Sauvegarde en fichier PDF
fig.write_image("beam_deformation_all.pdf", format="pdf", engine="kaleido")

# Affichage
fig.show()





