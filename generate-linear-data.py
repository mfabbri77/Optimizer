import numpy as np

# Parametri della funzione lineare
a = 2.5  # pendenza
b = 1.0  # intercetta

# Generazione dei dati
n_points = 256
x = np.linspace(0, 10, n_points)
y = a * x + b

# Aggiunta di rumore gaussiano
noise = np.random.normal(0, 0.5, n_points)
y_noisy = y + noise

# Creazione del file di dati
with open('linear.dat', 'w') as f:
    f.write("x = [")
    f.write(", ".join(map(str, x)))
    f.write("]\n")
    f.write("y = [")
    f.write(", ".join(map(str, y_noisy)))
    f.write("]\n")

print("File 'linear.dat' generato con successo.")
