# ==================================================
# GRÁFICOS ADICIONALES PARA EL ANÁLISIS
# ==================================================

import matplotlib.pyplot as plt
import numpy as np

# Datos simulados (reemplazar con tus datos reales)
energies = np.linspace(0.1, 0.5, 100)
counts = 1000 * energies**0.1 + 500 * np.exp(-(energies - 0.23)**2 / 0.001)

plt.figure(figsize=(10, 6))
plt.plot(energies, counts)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energía (GeV)')
plt.ylabel('Cuentas')
plt.title('Espectro de rayos gamma (ejemplo)')
plt.grid(True)
plt.savefig('../results/grafico_ejemplo.png')