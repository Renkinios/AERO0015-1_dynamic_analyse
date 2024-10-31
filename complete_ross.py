import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as las
# from matplotlib import pyplot as plt
from scipy import optimize
import warnings
import plotly.io as pio
import plotly.graph_objects as go

'''
This code is inspired by the library of Ross, it is just be change to work for us exemple.

'''

def _eigen(
        rotor,
        speed,
        num_modes=12,
        frequency=None,
        sorted_=True,
        sparse=None,
        synchronous=False,
        Gyro = [],
    ):
        A = get_A(rotor, speed=speed, frequency=frequency, synchronous=synchronous, Gyro = Gyro)
        filter_eigenpairs = lambda values, vectors, indices: (
            values[indices],
            vectors[:, indices],
        )

        if synchronous:
            evalues, evectors = la.eig(A)

            idx = np.where(np.imag(evalues) != 0)[0]
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)
            idx = np.where(np.abs(np.real(evalues) / np.imag(evalues)) < 1000)[0]
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)
        else:
            if sparse:
                try:
                    evalues, evectors = las.eigs(
                        A,
                        k=min(2 * num_modes, max(num_modes, A.shape[0] - 2)),
                        sigma=1,
                        which="LM",
                        v0=np.ones(A.shape[0]),
                    )
                except las.ArpackError:
                    evalues, evectors = la.eig(A)
            else:
                evalues, evectors = la.eig(A)

            if sparse is not None:
                idx = np.where((np.imag(evalues) != 0) & (np.abs(evalues) > 0.1))[0]
                evalues, evectors = filter_eigenpairs(evalues, evectors, idx)

        if sorted_:
            idx = rotor._index(evalues)
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)

        return evalues, evectors

def get_A(rotor, Gyro = [] ,speed=0, frequency=None, synchronous=False):
    if frequency is None:
        frequency = speed

    Z = np.zeros((rotor.ndof, rotor.ndof))
    I = np.eye(rotor.ndof)

    if len(Gyro) != 0:
    
        A22 = la.solve(-rotor.M(frequency,synchronous=synchronous), (np.array(Gyro) * speed))
        A = np.vstack(
            [np.hstack([Z, I]),
            np.hstack([la.solve(-rotor.M(frequency, synchronous=synchronous), rotor.K(frequency)), A22])])
        # fmt: on
    else:
        # fmt: off
        A = np.vstack(
            [np.hstack([Z, I]),
            np.hstack([la.solve(-rotor.M(frequency, synchronous=synchronous), rotor.K(frequency)), la.solve(-rotor.M(frequency,synchronous=synchronous), (rotor.G() * speed))])])
        # fmt

    return A

def get_mode(rotor, speed, num_modes=12, sparse=True, synchronous=False, Gyro = []):
    evalues, evectors = _eigen(
        rotor = rotor, speed=speed, num_modes=num_modes, sparse=sparse, synchronous=synchronous, Gyro = Gyro
    )
    
    wn_len = num_modes // 2
    wn = (np.absolute(evalues))[:wn_len]
    wd = (np.imag(evalues))[:wn_len]
    damping_ratio = (-np.real(evalues) / np.absolute(evalues))[:wn_len]
    
    return {"wn": wn, "wd": wd, "damping_ratio": damping_ratio, "evectors": evectors, "real_part": (np.real(evalues))[:wn_len]}


def run_campbell(rotor, speed_range, frequencies=6, Gyro=[], frequency_type="wd", slope_critic_speed = 1, units="rad/s", two_shaft = False):
    matrix = []
    for i, w in enumerate(speed_range):
        modal = get_mode(rotor, w, num_modes=2 * frequencies, sparse=True, synchronous=False, Gyro=Gyro)
        if frequency_type == "wn":
            wn = modal["wn"][:frequencies]
        else:
            wn = modal["wd"][:frequencies]
        matrix.append(wn)
    
    matrix = np.array(matrix).T

    # Initialiser la figure
    fig = go.Figure()
    # Ajouter les traces pour chaque mode
    for i, wn in enumerate(matrix):
        for j in range(1,len(wn)):
            slides = (wn[j] - wn[j-1]) 
            if slides > 0:
                marker_symbol = 'triangle-up'
            else:
                 marker_symbol = 'triangle-down'
            # Ajouter des points pour chaque mode avec des couleurs et symboles appropriés
            fig.add_trace(go.Scatter(
            x=np.array(speed_range[j]),
            y=np.array(wn[j]),
            mode='markers',
            marker=dict(size=10, symbol=marker_symbol, color='#8B0000'),
            showlegend=False,
            hoverinfo='skip'  # Désactiver les info-bulles
    ))
    critical_speed = run_critical_speed(rotor, num_modes= frequencies * 2, Gyro=Gyro, slope=slope_critic_speed)
    if frequency_type == "wn":
        crictal_speed = critical_speed["wn"][(critical_speed["wn"]) < speed_range[-1]]
    else:
        crictal_speed = critical_speed["wd"][critical_speed["wd"] < speed_range[-1]]
    
    # Ajouter le point de la vitesse critique avec une info-bulle personnalisée
    fig.add_trace(go.Scatter(
        x=crictal_speed,
        y=crictal_speed * slope_critic_speed,
        mode='markers',
        marker=dict(symbol="x", color="black", size=10),
        name="Crit. Speed",
        showlegend=True,
        hovertemplate=f"Frequency ({units}): %{{y:.2f}}<br>Critical Speed ({units}): %{{x:.2f}}"
    ))
    if two_shaft :

        critical_speed = run_critical_speed(rotor, num_modes= frequencies * 2, Gyro=Gyro, slope=1.5)
        if frequency_type == "wn":
            crictal_speed_2 = critical_speed["wn"][(critical_speed["wn"]) < speed_range[-1]]
        else:
            crictal_speed_2 = critical_speed["wd"][critical_speed["wd"] < speed_range[-1]]

        fig.add_trace(go.Scatter(
            x=crictal_speed_2,
            y=crictal_speed_2 * 1.5,
            mode='markers',
            marker=dict(symbol="x", color="black", size=10),
            name="Crit. Speed",
            # showlegend=True,
            hovertemplate=f"Frequency ({units}): %{{y:.2f}}<br>Critical Speed ({units}): %{{x:.2f}}"
        ))
        fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range * 1.5,
        mode='lines',
        line=dict(color="#556B2F", dash='dashdot'),
        showlegend=True,
        name="1.5x speed",
        hoverinfo='skip'  # Désactiver les info-bulles
    ))
        

    # Créer le nom dynamique pour la ligne de pente critique, sans la montrer dans la légende
    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range * slope_critic_speed,
        mode='lines',
        line=dict(color='blue', dash='dashdot'),
        showlegend=True,
        name=f"{slope_critic_speed}x speed",
        hoverinfo='skip'  # Désactiver les info-bulles
    ))


    # Ajouter une ligne verticale pour la vitesse critique sans la montrer dans la légende
    fig.add_shape(type='line',
        x0=5000/60 * 2 * np.pi,
        x1=5000/60 * 2 * np.pi,
        y0=0,
        y1=max(matrix.flatten()),
        line=dict(color='black', dash='dash'),
    )
    fig.add_trace(go.Scatter(
    x=[None], y=[None],  # Aucun point affiché
    mode='markers',
    marker=dict(size=10, symbol='triangle-up', color='#8B0000'),  # Couleur transparente
    name='Forward'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],  # Aucun point affiché
        mode='markers',
        marker=dict(size=10, symbol='triangle-down', color='#8B0000'),  # Couleur transparente
        name='Backward'
    ))

    # Mise en forme du graphique et positionnement de la légende en bas
    fig.update_layout(
        xaxis_title='Vitesse [rad/s]',
        yaxis_title='Pulsation Naturelle [rad/s]',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",  # Légende horizontale
            yanchor="top",
            y=-0.3,  # Positionnement en bas
            xanchor="center",
            x=0.5
        ),
        width=800,
        height=600,
        xaxis=dict(
            range=[speed_range[0], speed_range[-1]]
        )
    )

    # Afficher le plot
    fig.show()




def run_critical_speed(rotor, slope  = 1 ,num_modes=12, rtol=0.005, Gyro = []):
    modal = get_mode(rotor, 0, num_modes = num_modes, Gyro = Gyro)
    _wn = modal["wn"]
    _wd = modal["wd"]
    wn = np.zeros_like(_wn)
    wd = np.zeros_like(_wd)

    for i in range(len(wn)):
        wn_func = lambda s: (slope * s - get_mode(rotor, s, num_modes = num_modes, Gyro= Gyro)["wn"][i])
        wn[i]   =  optimize.newton(wn_func, _wn[i], tol=rtol)

    for i in range(len(wd)):
        wd_func = lambda s: (s * slope - get_mode(rotor, s, num_modes = num_modes, Gyro= Gyro)["wd"][i])
        wd[i]   =  optimize.newton(wn_func, _wd[i], tol=rtol)

    return {"wn": wn, "wd": wd}

def run_cambell_2_rotor(rotor1, rotor2, speed_range, frequencies=6, frequency_type="wd", Gyro=[], units="rad/s"):
    matrix1 = []
    matrix2 = []
    for i, w in enumerate(speed_range):
        modal1 = get_mode(rotor1, w, num_modes=frequencies *2, sparse=True, synchronous=False)
        modal2 = get_mode(rotor2, w, num_modes=frequencies *2, sparse=True, synchronous=False, Gyro=Gyro)
        if frequency_type == "wn":
            wn1 = modal1["wn"][:frequencies]
            wn2 = modal2["wn"][:frequencies]
        else:
            wn1 = modal1["wd"][:frequencies]
            wn2 = modal2["wd"][:frequencies]
        matrix1.append(wn1)
        matrix2.append(wn2)
    
    matrix1 = np.array(matrix1).T
    matrix2 = np.array(matrix2).T

    # Initialiser la figure
    fig = go.Figure()

    # Ajouter les traces pour chaque mode
    for i, (wn1, wn2) in enumerate(zip(matrix1, matrix2)):
        for j in range(1,len(wn1)):
            slides1 = (wn1[j] - wn1[j-1]) 
            slides2 = (wn2[j] - wn2[j-1]) 
            if slides1 > 0:
                marker_symbol1 = 'triangle-up'
            else:
                marker_symbol1 = 'triangle-down'
            if slides2 > 0:
                marker_symbol2 = 'triangle-up'
            else:
                marker_symbol2 = 'triangle-down'
        
            # Ajouter des points pour chaque mode avec des couleurs et symboles appropriés
            fig.add_trace(go.Scatter(
            x=np.array(speed_range[j]),
            y=np.array(wn1[j]),
            mode='markers',
            marker=dict(size=10, symbol=marker_symbol1, color='#8B0000'),
            showlegend=False,
            hoverinfo='skip'  # Désactiver les info-bulles
        ))
            fig.add_trace(go.Scatter(
            x=np.array(speed_range[j]),
            y=np.array(wn2[j]),
            mode='markers',
            marker=dict(size=10, symbol=marker_symbol2, color='#7B68EE'),
            showlegend=False,
            hoverinfo='skip'  # Désactiver les info-bulles
            ))   
        # Ajouter les lignes de référence
    critical_speed_1 = run_critical_speed(rotor1, num_modes= frequencies *2)
    if frequency_type == "wn":
        crictal_speed_1 = critical_speed_1["wn"][critical_speed_1["wn"] < speed_range[-1]]
    else:
        crictal_speed_1 = critical_speed_1["wd"][critical_speed_1["wd"] < speed_range[-1]]
    fig.add_trace(go.Scatter(
        x = crictal_speed_1,
        y = crictal_speed_1 ,
        mode='markers',
        marker=dict(size=10, symbol='x', color='black'),
        name="Crit. Speed first rotor",
        showlegend=True,
        hovertemplate=f"Frequency ({units}): %{{y:.2f}}<br>Critical Speed ({units}): %{{x:.2f}}"
    ))

    critical_speed_2 = run_critical_speed(rotor2, num_modes= frequencies * 2, Gyro=Gyro, slope=1.5)
    if frequency_type == "wn":
        crictal_speed_2 = critical_speed_2["wn"][critical_speed_2["wn"] < speed_range[-1]]
    else:
        crictal_speed_2 = critical_speed_2["wd"][critical_speed_2["wd"] < speed_range[-1]]
    
    fig.add_trace(go.Scatter(
        x = crictal_speed_2,
        y = crictal_speed_2 * 1.5,
        mode='markers',
        marker=dict(size=10, symbol='x', color='black'),
        name="Crit. Speed second rotor",
        showlegend=True,
        hovertemplate=f"Frequency ({units}): %{{y:.2f}}<br>Critical Speed ({units}): %{{x:.2f}}"
    ))
    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range,
        mode='lines',
        line=dict(color='blue', dash='dashdot'),
        name='y=x'
    ))

    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range * 1.5,
        mode='lines',
        line=dict(color='green', dash='dashdot'),
        name='y=1.5x'
    ))

    # Ajouter une ligne verticale
    fig.add_shape(type='line',
        x0=5000/60 * 2 * np.pi,
        x1=5000/60 * 2 * np.pi,
        line=dict(color='black', dash='dash'),
    )

    # Mise en forme du graphique
    fig.update_layout(
        xaxis_title='Vitesse (rad/s)',
        yaxis_title='Pulsation Naturelle (rad/s)',
        template='plotly_white',
        showlegend=True,
        width=800,
        height=600,
        xaxis=dict(range=[speed_range[0], speed_range[-1]])
    )

    # Afficher le plot
    pio.show(fig)
    

def run_damping_mode(rotor1, rotor2, speed_range, frequencies=6, frequency_type="wd", Gyro=[]):
    matrix1 = []
    matrix2 = []
    damping1 = []
    damping2 = []
    instab = []
    
    for i, w in enumerate(speed_range):
        modal1 = get_mode(rotor1, w, num_modes=frequencies, sparse=True, synchronous=False)
        modal2 = get_mode(rotor2, w, num_modes=frequencies, sparse=True, synchronous=False, Gyro=Gyro)
        if frequency_type == "wn":
            wn1 = modal1["wn"][:frequencies]
            wn2 = modal2["wn"][:frequencies]
        else:
            wn1 = modal1["wd"][:frequencies]
            wn2 = modal2["wd"][:frequencies]

        if modal1["real_part"][:frequencies] > 0:
            instab.append(modal1["real_part"][:frequencies])
            instab.append(modal2["real_part"][:frequencies])
        

        damping1.append(modal1["real_part"][:frequencies])
        damping2.append(modal2["real_part"][:frequencies])
        matrix1.append(wn1)
        matrix2.append(wn2)
    
    matrix1 = np.array(matrix1).T
    matrix2 = np.array(matrix2).T
    damping1 = np.array(damping1).T
    damping2 = np.array(damping2).T

    # Calculer le min et le max de damping1 et damping2
    min_damping = min(damping1.min(), damping2.min())
    max_damping = max(damping1.max(), damping2.max())

    # Ajuster les bornes pour avoir une petite marge
    margin = (max_damping - min_damping) * 0.1
    y_min = min_damping - margin
    y_max = max_damping + margin

    # Initialiser la figure
    fig = go.Figure()

    # Ajouter les traces pour chaque mode
    for i, (wn1, wn2, d1, d2) in enumerate(zip(matrix1, matrix2, damping1, damping2)):
        slides1 = (wn1[-1] - wn1[0]) / (speed_range[1] - speed_range[0])
        slides2 = (wn2[-1] - wn2[0]) / (speed_range[1] - speed_range[0])
        marker_symbol1 = 'triangle-up' if slides1 > 0 else 'triangle-down'
        marker_symbol2 = 'triangle-up' if slides2 > 0 else 'triangle-down'
        
        # Boucle sur chaque point pour rotor1
        for j, d1_value in enumerate(d1):
            color_symbol1 = 'red' if d1_value > 0 else 'blue'
            fig.add_trace(go.Scatter(
                x=[speed_range[j]],
                y=[d1_value],  # Utilisation du damping ratio de rotor1
                mode='markers+lines',
                marker=dict(size=10, symbol=marker_symbol1, color=color_symbol1),
                name=f'Mode {i+1} - Rotor 1',
                showlegend=(j == 0)  # Montrer la légende seulement pour le premier point de la série
            ))
        
        # Boucle sur chaque point pour rotor2
        for j, d2_value in enumerate(d2):
            color_symbol2 = 'red' if d2_value > 0 else 'blue'
            fig.add_trace(go.Scatter(
                x=[speed_range[j]],
                y=[d2_value],  # Utilisation du damping ratio de rotor2
                mode='markers+lines',
                marker=dict(size=10, symbol=marker_symbol2, color=color_symbol2),
                name=f'Mode {i+1} - Rotor 2',
                showlegend=(j == 0)  # Montrer la légende seulement pour le premier point de la série
            ))

    # Ajouter une ligne verticale
    fig.add_shape(type='line',
        x0=5000 / 60 * 2 * np.pi,
        x1=5000 / 60 * 2 * np.pi,
        y0=y_min,
        y1=y_max,  # Ajuster la ligne à la hauteur de l'axe Y
        line=dict(color='black', dash='dash'),
    )

    # Mise en forme du graphique
    fig.update_layout(
        xaxis_title='Vitesse (rad/s)',
        yaxis_title='Damping Ratio',
        template='plotly_white',
        showlegend=True,
        width=800,
        height=600,
        xaxis=dict(range=[speed_range[0], speed_range[-1]]),
        yaxis=dict(range=[y_min, y_max])  # Ajuster l'échelle de l'axe Y
    )

    # Afficher le plot
    pio.show(fig)
