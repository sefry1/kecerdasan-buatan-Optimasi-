import numpy as np
import matplotlib.pyplot as plt

# Definisikan fungsi objektif f(x) = x^2
def objective_function(x):
    return x**2

# Parameter PSO
num_particles = 10
max_iterations = 50
w = 0.5  # Bobot inersia
c1 = 1.5  # Koefisien kognitif
c2 = 1.5  # Koefisien sosial
bounds = (-10, 10)  # Batas pencarian: [-10, 10]

# Inisialisasi partikel
np.random.seed(42)  # Untuk reproduktibilitas
positions = np.random.uniform(bounds[0], bounds[1], num_particles)  # Posisi awal acak
velocities = np.random.uniform(-1, 1, num_particles)  # Kecepatan awal acak
pbest_positions = positions.copy()
pbest_values = np.array([objective_function(x) for x in pbest_positions])
gbest_idx = np.argmin(pbest_values)
gbest_position = pbest_positions[gbest_idx]
gbest_value = pbest_values[gbest_idx]

# Simpan nilai terbaik per iterasi untuk grafik
gbest_values_per_iteration = []

# Loop utama PSO
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Perbarui kecepatan
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - positions[i]) +
                         c2 * r2 * (gbest_position - positions[i]))
        
        # Perbarui posisi
        positions[i] += velocities[i]
        
        # Batasi posisi dalam batas
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])
        
        # Evaluasi fitness
        fitness = objective_function(positions[i])
        
        # Perbarui posisi terbaik pribadi
        if fitness < pbest_values[i]:
            pbest_positions[i] = positions[i]
            pbest_values[i] = fitness
        
        # Perbarui posisi terbaik global
        if fitness < gbest_value:
            gbest_position = positions[i]
            gbest_value = fitness
    
    # Simpan nilai terbaik global untuk iterasi ini
    gbest_values_per_iteration.append(gbest_value)

# Cetak hasil
print(f"Nilai minimum: {gbest_value}")
print(f"Posisi x terbaik: {gbest_position}")

# Buat grafik nilai terbaik per iterasi
plt.plot(range(max_iterations), gbest_values_per_iteration, label="Nilai Terbaik per Iterasi")
plt.xlabel("Iterasi")
plt.ylabel("Nilai Terbaik (f(x))")
plt.title("PSO: Nilai Terbaik per Iterasi")
plt.grid(True)
plt.legend()
plt.savefig('pso_best_value_per_iteration.png')