import ross as rs
import numpy as np
import plotly.graph_objects as go
import complete_ross as fct


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
low_pressure_seal_elements.append(rs.BearingElement(n=6, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0))
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
    rs.DiskElement.from_geometry(n=2, material=material_exo, width=0.015, i_d=0.08, o_d=0.038)
)

hight_pressure_seal_elements.append(rs.BearingElement(n=0, kxx=2.5e7, kyy=2.5e7, cxx=0, cyy=0))
hight_pressure_seal_elements.append(rs.BearingElement(n=4, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0))
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
    rs.DiskElement.from_geometry(n=11, material=material_exo, width=0.015, i_d=0.08, o_d=0.038)
]

bearings = [
    rs.BearingElement(0, kxx=3*10**7, kyy=3*10**7, cxx=0),
    rs.BearingElement(8, kxx=3*10**7, kyy=3*10**7, cxx=0),
    rs.BearingElement(9, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0),
    rs.BearingElement(6, n_link=13, kxx=2.5*10**7, kyy=2.5*10**7, cxx=0)
]
""
# Création du rotor coaxial
shaft = [axial_shaft, coaxial_shaft]
rotor_coaxial = rs.CoAxialRotor(shaft, disks, bearings)
rotor_coaxial.plot_rotor()

G_lp = rotor_lp.G()
# print("Gyrioscopique Low pressure \t",G_lp.shape)
G_hp = rotor_hp.G() * 1.5
# print("Gyrioscopique High pressure \t",G_hp.shape)

G1 = np.hstack((G_lp, np.zeros((G_lp.shape[0], rotor_coaxial.ndof - G_lp.shape[1]))))
G2 = np.hstack((np.zeros((G_hp.shape[0],rotor_coaxial.ndof - G_hp.shape[1])), G_hp))
G = np.vstack((G1, G2))


max_spin = 20000/60 * 2 * np.pi # rad/s
samples = 40
speed_range = np.linspace(0, max_spin, samples)
# fct.run_campbell(rotor_coaxial, speed_range, frequencies = 7, Gyro=G) 

# run_critical_speed = fct.run_critical_speed(rotor_coaxial, num_modes=7, Gyro=G)

# print("wd = ",run_critical_speed["wd"])
# print("wn = ",run_critical_speed["wn"])

# fct.run_cambell_2_rotor(rotor_lp, rotor_hp, speed_range, frequencies=3, frequency_type="wn")
fct.run_campbell(rotor_lp, speed_range, frequencies = 10)
# print("G_hp", G_hp.shape)
# fct.run_campbell(rotor_hp, speed_range, frequencies = 7, Gyro=rotor_hp.G() * 1.5)
