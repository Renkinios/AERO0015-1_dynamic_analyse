import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as las
# from matplotlib import pyplot as plt
from scipy import optimize
import warnings
import plotly.io as pio
import plotly.graph_objects as go

def _eigen(
        rotor,
        speed,
        num_modes=12,
        frequency=None,
        sorted_=True,
        sparse=True,
        synchronous=False,
        Gyro = [],
    ):
        A = get_A(rotor, speed=speed, frequency=frequency, synchronous=synchronous, Gyro = Gyro)

        # evalues, evectors = la.eig(A)
        # idx = np.where(np.imag(evalues) != 0)[0]
        # evalues = evalues[idx]
        # evectors = evectors[:, idx]
        # idx = np.where(np.abs(np.real(evalues) / np.imag(evalues)) < 1000)[0]
        # evalues = evalues[idx]
        # evectors = evectors[:, idx]
        

        # idx = rotor._index(evalues)

        # return evalues[idx], evectors[:, idx]

        # if A is None:
        # A = get_AA(speed=speed, frequency=frequency, synchronous=synchronous)

        if synchronous:
            evalues, evectors = la.eig(A)
            idx = np.where(np.imag(evalues) != 0)[0]
            evalues = evalues[idx]
            evectors = evectors[:, idx]
            idx = np.where(np.abs(np.real(evalues) / np.imag(evalues)) < 1000)[0]
            evalues = evalues[idx]
            evectors = evectors[:, idx]
        else:
            if sparse is True:
                try:
                    evalues, evectors = las.eigs(
                        A,
                        k=2 * num_modes,
                        sigma=1,
                        ncv=4 * num_modes,
                        which="LM",
                        v0=rotor._v0,
                    )
                    # store v0 as a linear combination of the previously
                    # calculated eigenvectors to use in the next call to eigs
                    rotor._v0 = np.real(sum(evectors.T))

                    # Disregard rigid body modes:
                    idx = np.where(np.abs(evalues) > 0.1)[0]
                    evalues = evalues[idx]
                    evectors = evectors[:, idx]
                except las.ArpackError:
                    evalues, evectors = la.eig(A)
            else:
                evalues, evectors = la.eig(A)

        if sorted_ is False:
            return evalues, evectors

        idx = rotor._index(evalues)

        return evalues[idx], evectors[:, idx]

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

def run_modal(rotor, speed, num_modes=12, sparse=True, synchronous=False, Gyro = []):
    evalues, evectors = _eigen(
        rotor = rotor, speed=speed, num_modes=num_modes, sparse=sparse, synchronous=synchronous, Gyro = Gyro
    )
    
    wn_len = num_modes // 2
    wn = (np.absolute(evalues))[:wn_len]
    wd = (np.imag(evalues))[:wn_len]
    damping_ratio = (-np.real(evalues) / np.absolute(evalues))[:wn_len]
    
    return {"wn": wn, "wd": wd, "damping_ratio": damping_ratio, "evectors": evectors}


def run_campbell(rotor, speed_range, frequencies=6, Gyro=[], frequency_type="wd", slope_critic_speed = 1):
    matrix = []
    for i, w in enumerate(speed_range):
        modal = run_modal(rotor, w, num_modes=2 * frequencies, sparse=True, synchronous=False, Gyro=Gyro)
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
        slides = (wn[len(wn)-1] - wn[5]) / (speed_range[1] - speed_range[0])
        marker_symbol = 'triangle-up' if slides > 0 else 'triangle-down'
        
        # Ajouter des points pour chaque mode avec des couleurs et symboles appropriés
        fig.add_trace(go.Scatter(
            x=speed_range,
            y=wn,
            mode='markers',
            marker=dict(size=10, symbol=marker_symbol, color='#8B0000'),
            # name=f'Mode {i+1}'
        ))
    critical_speed = run_critical_speed(rotor, num_modes= frequencies * 2, Gyro=Gyro)
    if frequency_type == "wn":
        crictal_speed = critical_speed["wn"]
    else:
        crictal_speed = critical_speed["wd"]
    
    fig.add_trace(go.Scatter(
        x = crictal_speed,
        y = crictal_speed * slope_critic_speed,
        mode='markers',
        marker=dict(size=10, symbol='circle-x', color='black'),
    ))

    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range * slope_critic_speed,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='y=1.5x'
    ))

    # Ajouter une ligne verticale
    fig.add_shape(type='line',
        x0=5000/60 * 2 * np.pi,
        x1=5000/60 * 2 * np.pi,
        y0=0,
        y1=max(matrix.flatten()),
        line=dict(color='black', dash='dash'),
    )

    # Mise en forme du graphique
    fig.update_layout(
        # title='Diagramme de Campbell',
        xaxis_title='Vitesse (rad/s)',
        yaxis_title='Pulsation Naturelle (rad/s)',
        template='plotly_white',
        showlegend=True,
        width=800,
        height=600,
        xaxis=dict(
        range=[speed_range[0], speed_range[len(speed_range)-1]]) 
    )

    # Afficher le plot
    fig.show()




def run_critical_speed(rotor, slope  = 1 ,num_modes=12, rtol=0.005, Gyro = []):
    modal = run_modal(rotor, 0, num_modes = num_modes, Gyro = Gyro)
    _wn = modal["wn"]
    _wd = modal["wd"]
    wn = np.zeros_like(_wn)
    wd = np.zeros_like(_wd)

    for i in range(len(wn)):
        wn_func = lambda s: (slope * s - run_modal(rotor, s, num_modes = num_modes, Gyro= Gyro)["wn"][i])
        wn[i]   =  optimize.newton(wn_func, _wn[i], tol=rtol)

    for i in range(len(wd)):
        wd_func = lambda s: (s * slope - run_modal(rotor, s, num_modes = num_modes, Gyro= Gyro)["wd"][i])
        wd[i]   =  optimize.newton(wn_func, _wd[i], tol=rtol)

    return {"wn": wn, "wd": wd}

def run_cambell_2_rotor(rotor1, rotor2, speed_range, frequencies=6, frequency_type="wd", Gyro=[]):
    matrix1 = []
    matrix2 = []
    for i, w in enumerate(speed_range):
        modal1 = run_modal(rotor1, w, num_modes=frequencies, sparse=True, synchronous=False)
        modal2 = run_modal(rotor2, w, num_modes=frequencies, sparse=True, synchronous=False, Gyro=Gyro)
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
        slides1 = (wn1[len(wn1)-1] - wn1[0]) / (speed_range[1] - speed_range[0])
        slides2 = (wn2[len(wn2)-1] - wn2[4]) / (speed_range[1] - speed_range[0])
        marker_symbol1 = 'triangle-up' if slides1 > 0 else 'triangle-down'
        marker_symbol2 = 'triangle-up' if slides2 > 0 else 'triangle-down'
        
        # Ajouter des points pour chaque mode avec des couleurs et symboles appropriés
        fig.add_trace(go.Scatter(
            x=speed_range,
            y=wn1,
            mode='markers',  # Utilisation de 'markers' uniquement
            marker=dict(size=10, symbol=marker_symbol1, color='#8B0000'),
            name=f'Mode {i+1} - Rotor 1'
        ))

        fig.add_trace(go.Scatter(
            x=speed_range,
            y=wn2,
            mode='markers',  # Utilisation de 'markers' uniquement
            marker=dict(size=10, symbol=marker_symbol2, color='#00008B'),
            name=f'Mode {i+1} - Rotor 2'
        ))
        # Ajouter les lignes de référence
    critical_speed_1 = run_critical_speed(rotor1, num_modes= frequencies + 1)
    if frequency_type == "wn":
        crictal_speed_1 = critical_speed_1["wn"]
    else:
        crictal_speed_1 = critical_speed_1["wd"]
    fig.add_trace(go.Scatter(
        x = crictal_speed_1,
        y = crictal_speed_1 * 1,
        mode='markers',
        marker=dict(size=10, symbol='circle-x', color='black'),
    ))

    critical_speed_2 = run_critical_speed(rotor2, num_modes= frequencies + 1, Gyro=Gyro)
    if frequency_type == "wn":
        crictal_speed_2 = critical_speed_2["wn"]
    else:
        crictal_speed_2 = critical_speed_2["wd"]
    fig.add_trace(go.Scatter(
        x = crictal_speed_2,
        y = crictal_speed_2 * 1.5,
        mode='markers',
        marker=dict(size=10, symbol='circle-x', color='black'),
    ))
    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='y=x'
    ))

    fig.add_trace(go.Scatter(
        x=speed_range,
        y=speed_range * 1.5,
        mode='lines',
        line=dict(color='black', dash='dash'),
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
    
    for i, w in enumerate(speed_range):
        modal1 = run_modal(rotor1, w, num_modes=frequencies, sparse=True, synchronous=False)
        modal2 = run_modal(rotor2, w, num_modes=frequencies, sparse=True, synchronous=False, Gyro=Gyro)
        if frequency_type == "wn":
            wn1 = modal1["wn"][:frequencies]
            wn2 = modal2["wn"][:frequencies]
        else:
            wn1 = modal1["wd"][:frequencies]
            wn2 = modal2["wd"][:frequencies]
        
        damping1.append(modal1["damping_ratio"][:frequencies])
        damping2.append(modal2["damping_ratio"][:frequencies])
        matrix1.append(wn1)
        matrix2.append(wn2)
    
    matrix1 = np.array(matrix1).T
    matrix2 = np.array(matrix2).T
    damping1 = np.array(damping1).T
    damping2 = np.array(damping2).T

    # Initialiser la figure
    fig = go.Figure()

    # Ajouter les traces pour chaque mode
    for i, (wn1, wn2, d1, d2) in enumerate(zip(matrix1, matrix2, damping1, damping2)):
        slides1 = (wn1[-1] - wn1[0]) / (speed_range[1] - speed_range[0])
        slides2 = (wn2[-1] - wn2[0]) / (speed_range[1] - speed_range[0])
        marker_symbol1 = 'triangle-up' if slides1 > 0 else 'triangle-down'
        marker_symbol2 = 'triangle-up' if slides2 > 0 else 'triangle-down'
        
        # Ajouter des points pour chaque mode avec des couleurs et symboles appropriés
        fig.add_trace(go.Scatter(
            x=speed_range,
            y=d1,  # Utilisation du damping ratio de rotor1
            mode='markers',  # Utilisation de 'markers' uniquement
            marker=dict(size=10, symbol=marker_symbol1, color='#8B0000'),
            name=f'Mode {i+1} - Rotor 1'
        ))

        fig.add_trace(go.Scatter(
            x=speed_range,
            y=d2,  # Utilisation du damping ratio de rotor2
            mode='markers',  # Utilisation de 'markers' uniquement
            marker=dict(size=10, symbol=marker_symbol2, color='#00008B'),
            name=f'Mode {i+1} - Rotor 2'
        ))

    # Ajouter une ligne verticale
    fig.add_shape(type='line',
        x0=5000 / 60 * 2 * np.pi,
        x1=5000 / 60 * 2 * np.pi,
        y0=0,
        y1=1,  # Ligne qui couvre tout l'axe Y
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
        yaxis=dict(range=[0, 1])  # L'échelle de l'amortissement est souvent entre 0 et 1
    )

    # Afficher le plot
    pio.show(fig)