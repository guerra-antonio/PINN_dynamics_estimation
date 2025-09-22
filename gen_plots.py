import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------------- #
# Make the first plot of fidelities
# Read fidelity data
df = pd.read_csv('dataframe/fidelity_test.csv')

# Fidelity keys
fids_label = {
    0: "fid_01",
    1: "fid_025",
    2: "fid_05",
    3: "fid_10"
}

fids_color = {
    0: "red",
    1: "blue",
    2: "green",
    3: "gold"
}

fids_delta = {
    0: 0.1,
    1: 0.25,
    2: 0.5,
    3: 1.0
}

# Plot 1 setup
cols = 2
n_qubits = 5
rows = (n_qubits + 1) // 2  # ceil(n_qubits / 2)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharex=True)

times = np.linspace(0, 1, 100)

for idx in range(n_qubits):
    row, col = divmod(idx, cols)
    ax = axes[row, col] if rows > 1 else axes[col]

    for f_ in range(4):
        fidelity = df.loc[idx, fids_label[f_]]
        fidelity = np.fromstring(fidelity.strip('[]'), sep=' ')
        ax.plot(times, fidelity, color=fids_color[f_], label=r"$\Delta t = $" + str(fids_delta[f_]))

    ax.set_title(f"{idx + 2} Qubits", fontsize=10)
    ax.grid(True)
    if row == rows - 1:
        ax.set_xlabel("Time")
    if col == 0:
        ax.set_ylabel("Fidelity")
    # ax.set_ylim(0.9, 1.01)
    ax.tick_params(labelsize=8)

# Remove empty subplot if odd number of qubits
if n_qubits % 2 != 0:
    fig.delaxes(axes[-1, -1])

# Global legend
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9)

# Layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fidelity_qubits_2to6.png", dpi=300, bbox_inches='tight')
plt.close()

# Second plot of fidelities
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=True)

times = np.linspace(0, 1, 100)

for i, idx in enumerate([5, 6]):  # 7 qubits -> idx=5, 8 qubits -> idx=6
    ax = axes[i]
    for f_ in range(4):
        fidelity = df.loc[idx, fids_label[f_]]
        fidelity = np.fromstring(fidelity.strip('[]'), sep=' ')
        ax.plot(times, fidelity, color=fids_color[f_], label=r"$\Delta t = $" + str(fids_delta[f_]))

    ax.set_title(f"{idx + 2} Qubits", fontsize=10)  # idx+2 = número de qubits
    ax.grid(True)
    ax.set_xlabel("Time")
    if i == 0:
        ax.set_ylabel("Fidelity")
    ax.tick_params(labelsize=8)

# Leyenda global
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9)

# Layout y guardado
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fidelity_qubits_7and8.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------------------- #
# Make the plot of losses function along the different models
def read_loss(N = 2, dt = 0.1, test = True):
    df = pd.read_csv(f"data_loss/dataloss-model{N}-delta_t{dt}.csv")

    train   = df["train_loss"].to_numpy()
    return train

# Define the different parameters of the models
Ns  = [n for n in range(2, 7)]
dts = [0.1, 0.25, 0.5, 1.0]

rows = len(Ns)
cols = len(dts)

# Plot the losses
plt.clf()

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.25, wspace=0.2)

for i in range(rows):
    for j in range(cols):
        train = read_loss(N=Ns[i], dt=dts[j])
        epochs = np.arange(1000)

        ax = axes[i, j]
        ax.plot(train, color="red", linewidth=0.8)
        ax.set_yscale("log")
        ax.grid(True, linestyle='--', alpha=0.3)

        # Etiqueta de columna (arriba)
        if i == 0:
            ax.set_title(f"$\\Delta t$ = {dts[j]}", fontsize=10)

        # Etiqueta de fila (izquierda)
        if j == 0:
            ax.set_ylabel(f"{Ns[i]} qubits", fontsize=10)

        # Solo última fila con etiquetas en X
        if i == rows - 1:
            ax.set_xlabel("Epochs", fontsize=9)
        # else:
            # ax.set_xticklabels([])

        # Solo primera columna con etiquetas en Y
        # if j > 0:
            # ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("figures/loss_variation_2to6.png", dpi=300, bbox_inches='tight')

# Define the different parameters of the models
Ns  = [7, 8]
dts = [0.1, 0.25, 0.5, 1.0]

rows = len(Ns)
cols = len(dts)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.25, wspace=0.2)

for i in range(rows):
    for j in range(cols):
        train = read_loss(N=Ns[i], dt=dts[j])
        epochs = np.arange(1000)

        ax = axes[i, j]
        ax.plot(train, color="red", linewidth=0.8)
        ax.set_yscale("log")
        ax.grid(True, linestyle='--', alpha=0.3)

        # Etiqueta de columna (arriba)
        if i == 0:
            ax.set_title(f"$\\Delta t$ = {dts[j]}", fontsize=10)

        # Etiqueta de fila (izquierda)
        if j == 0:
            ax.set_ylabel(f"{Ns[i]} qubits", fontsize=10)

        # Solo última fila con etiquetas en X
        if i == rows - 1:
            ax.set_xlabel("Epochs", fontsize=9)
        # else:
            # ax.set_xticklabels([])

        # Solo primera columna con etiquetas en Y
        # if j > 0:
            # ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("figures/loss_variation_7and8.png", dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------------------- #
# Make the plot of the number of trainable parameters

qubits = np.arange(2, 9)
restricted_indices = [5, 6]
params = np.array([
    12576,           # 2 qubits
    197760,          # 3 qubits
    3674624,         # 4 qubits
    5248000,         # 5 qubits
    37763072,        # 6 qubits
    760000000,       # 7 qubits (estimado)
    8800000000       # 8 qubits (estimado)
])

# Make the plot
plt.figure(figsize=(10, 6))
plt.scatter(qubits[:5], params[:5], marker='o', label='Trainable Parameters')
plt.yscale('log')

# Marcar los puntos con Hamiltonianos restringidos
plt.scatter(np.array(qubits)[restricted_indices],
            np.array(params)[restricted_indices],
            color='red', label='Estimated trainable parameters')

# Etiquetas y estilos
plt.xlabel('Number of qubits', fontsize=12)
plt.ylabel('Trainable Parameters', fontsize=12)
plt.title('Scaling of Trainable Parameters with System Size', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("figures/scaling_num-trainable-parameters.png", dpi = 300)


# ---------------------------------------------------------------------------------------- #
# Make the plot of the number of observables required for QPT

# Filas: qubits de 2 a 8
# Columnas: QPTs [1, 2, 4, 10]
import matplotlib.pyplot as plt
import numpy as np

# Datos
observables_array = np.array([
    [15,     30,     60,     150],
    [63,     126,    252,    630],
    [255,    510,    1020,   2550],
    [1023,   2046,   4092,   10230],
    [4095,   8190,   16380,  40950],
    [16383,  32766,  65532,  163830],
    [65535,  131070, 262140, 655350]
])

n_qubits = np.arange(2, 9)
time_grid = [1.0, 0.5, 0.25, 0.1]
colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson']

# Crear figura
plt.figure(figsize=(10, 6))

# Graficar cada curva
for i in range(observables_array.shape[1]):
    plt.plot(n_qubits, observables_array[:, i], marker='o', linestyle='-', color=colors[i],
             label=f'Δt = {time_grid[i]}')

# Ajustes de estilo
plt.yscale('log')
plt.xlabel('Number of Qubits', fontsize=12)
plt.ylabel('Total Observables Required', fontsize=12)
plt.title('Scaling of Observables with Qubits for Different Time Grids', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(title='Time Grid', fontsize=10)
plt.tight_layout()

# Guardar figura
plt.savefig("figures/obs-measurements-scaling.png", dpi=300)
