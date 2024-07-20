import numpy as np

# Parametri della funzione esponenziale
a = 2.0  # ampiezza
b = 0.5  # tasso di crescita
c = 1.0  # offset

# Generazione dei dati
n_points = 256
x = np.linspace(0, 5, n_points)
y = a * np.exp(b * x) + c

# Aggiunta di rumore gaussiano
noise = np.random.normal(0, 0.5, n_points)
y_noisy = y + noise

# Creazione del file di dati
with open('exponential.txt', 'w') as f:
    f.write("x = [")
    f.write(", ".join(map(str, x)))
    f.write("]\n")
    f.write("y = [")
    f.write(", ".join(map(str, y_noisy)))
    f.write("]\n")

print("File 'exponential.txt' generato con successo.")
