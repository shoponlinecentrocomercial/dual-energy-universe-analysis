# ==================================================
# ANÁLISIS DE DATOS DE MAGIC (TXS 0506+056)
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

print("✅ Script iniciado.")

# ==================================================
# 1. CARGAR DATOS DEL ARCHIVO TXT
# ==================================================

# Ruta al archivo (cambia si es necesario)
FILE_PATH = "MAGIC.txt"

print(f"\n📂 Cargando datos desde: {FILE_PATH}")

# Leer el archivo línea por línea
with open(FILE_PATH, 'r') as f:
    lines = f.readlines()

# Buscar la sección de eventos (#5)
start_idx = None
for i, line in enumerate(lines):
    if '#5 Event list' in line:
        start_idx = i + 1  # La siguiente línea es el encabezado
        break

if start_idx is None:
    print("⚠️ No se encontró la sección de eventos.")
    exit()

# Saltar el encabezado (línea con nombres de columnas)
event_lines = lines[start_idx+1:]

# Procesar eventos
events = []
for line in event_lines:
    # Saltar líneas vacías
    if not line.strip():
        continue
    # Dividir por comas
    parts = line.strip().split(', ')
    if len(parts) != 11:
        continue
    try:
        t = float(parts[0])          # MJD-55000
        E = float(parts[1])          # Energía [GeV]
        theta_on = float(parts[2])   # Distancia angular a la fuente
        used_on_sed = int(parts[7])  # 1 = usado como On en SED
        used_off_sed = int(parts[8]) # 1 = usado como Off en SED
        events.append({
            't': t,
            'E': E,
            'theta_on': theta_on,
            'used_on_sed': used_on_sed,
            'used_off_sed': used_off_sed
        })
    except:
        continue

print(f"✅ {len(events)} eventos cargados.")

# ==================================================
# 2. FILTRAR EVENTOS ON (fuente) y OFF (fondo)
# ==================================================

print("\n📊 Filtrando eventos...")

# Eventos On (usados para el espectro)
on_events = [e for e in events if e['used_on_sed'] == 1]
off_events = [e for e in events if e['used_off_sed'] == 1]

print(f"✅ Eventos On (fuente): {len(on_events)}")
print(f"✅ Eventos Off (fondo): {len(off_events)}")

# Energías de los eventos On
energies_on = np.array([e['E'] for e in on_events])
energies_off = np.array([e['E'] for e in off_events])

# ==================================================
# 3. CONSTRUIR ESPECTRO
# ==================================================

print("\n📊 Construyendo espectro...")

# Definir bins de energía (logarítmicos)
energy_min = 50  # GeV
energy_max = 5000  # GeV (5 TeV)
n_bins = 20

# Bins logarítmicos
bins = np.logspace(np.log10(energy_min), np.log10(energy_max), n_bins + 1)
bin_centers = np.sqrt(bins[:-1] * bins[1:])
bin_widths = bins[1:] - bins[:-1]

# Histograma de eventos On
counts_on, _ = np.histogram(energies_on, bins=bins)
counts_off, _ = np.histogram(energies_off, bins=bins)

# Restar el fondo (Off) de los datos On
# Normalizar por el número de eventos (si Off y On tienen diferentes exposiciones)
# En este caso, usamos la resta directa
counts = counts_on - counts_off * (len(on_events) / len(off_events) if len(off_events) > 0 else 1)

# Errores (Poisson)
counts_err = np.sqrt(counts_on + counts_off * (len(on_events) / len(off_events))**2)

# ==================================================
# 4. AJUSTE DEL FONDO (LEY DE POTENCIAS)
# ==================================================

print("\n🔧 Ajustando fondo...")

def fondo_modelo(energia, amp, index):
    return amp * (energia / 100) ** index

# Usar solo bins con cuentas positivas
mask = counts > 0
energies_fit = bin_centers[mask]
counts_fit = counts[mask]

try:
    popt, _ = curve_fit(fondo_modelo, energies_fit, counts_fit, p0=[100, -2.0], maxfev=5000)
    amp_fit, index_fit = popt
    print(f"✅ Fondo ajustado: amp={amp_fit:.2f}, index={index_fit:.2f}")
    
    # Calcular el fondo
    fondo_ajustado = fondo_modelo(bin_centers, amp_fit, index_fit)
    
    # Restar el fondo
    counts_sin_fondo = counts - fondo_ajustado
    
except Exception as e:
    print(f"⚠️ Error en el ajuste del fondo: {e}")
    # Si falla, usar un fondo simple (media)
    fondo_ajustado = np.mean(counts) * np.ones_like(counts)
    counts_sin_fondo = counts - fondo_ajustado

# ==================================================
# 5. BÚSQUEDA DE LÍNEAS
# ==================================================

print("\n🔍 Buscando líneas...")

def lineas_predichas(n, E0):
    return E0 * n**(-5/4)

# Detectar picos
peaks, _ = find_peaks(
    counts_sin_fondo,
    height=counts_sin_fondo.max() * 0.05,
    distance=2,
    prominence=counts_sin_fondo.max() * 0.02
)

energies_peaks = bin_centers[peaks]
counts_peaks = counts_sin_fondo[peaks]

print(f"✅ {len(peaks)} picos detectados.")
if len(peaks) > 0:
    print("   Energías de los picos (GeV):")
    for i, E in enumerate(energies_peaks[:10]):
        print(f"      {i+1}: {E:.2f} GeV (cuentas: {counts_peaks[i]:.0f})")

# ==================================================
# 6. AJUSTE DE LA RELACIÓN PREDICHA
# ==================================================

E0_ajustado = None
if len(peaks) >= 3:
    print("\n📐 Ajustando la relación predicha Eγ(n) = E0 * n^(-5/4)...")
    idx_sorted = np.argsort(energies_peaks)
    energies_sorted = energies_peaks[idx_sorted]
    n_vals = np.arange(1, len(energies_sorted) + 1)
    try:
        popt, _ = curve_fit(lineas_predichas, n_vals, energies_sorted)
        E0_ajustado = popt[0]
        print(f"✅ E0 ajustado: {E0_ajustado:.2f} GeV")
    except Exception as e:
        print(f"⚠️ El ajuste falló: {e}")

# ==================================================
# 7. VISUALIZACIÓN
# ==================================================

plt.figure(figsize=(12, 8))

# Espectro original
plt.errorbar(bin_centers, counts, yerr=counts_err,
             fmt='o', markersize=5, capsize=2, label='Datos (On-Off)', alpha=0.7)

# Fondo ajustado
plt.plot(bin_centers, fondo_ajustado, 'r-', label='Fondo ajustado', linewidth=2)

# Espectro sin fondo
plt.plot(bin_centers, counts_sin_fondo, 'b-', label='Espectro sin fondo', linewidth=1.5)

# Picos detectados
if len(peaks) > 0:
    plt.scatter(energies_peaks, counts_peaks, color='red', s=50, zorder=5,
                label=f'Picos detectados ({len(peaks)})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energía (GeV)')
plt.ylabel('Cuentas')
plt.title('Espectro de MAGIC - TXS 0506+056')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('espectro_magic_txs.png')
print("\n📈 Gráfico guardado como 'espectro_magic_txs.png'")

# ==================================================
# 8. RESUMEN
# ==================================================
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS (MAGIC - TXS 0506+056)")
print("="*50)
print(f"Eventos On: {len(on_events)}")
print(f"Eventos Off: {len(off_events)}")
print(f"Rango de energías: {energy_min} - {energy_max} GeV")
print(f"Picos detectados: {len(peaks)}")
if E0_ajustado is not None:
    print(f"E0 ajustado (líneas gamma): {E0_ajustado:.2f} GeV")
print("="*50)