# ==================================================
# ANÁLISIS DE DATOS DE FERMI-LAT 4FGL (14 años)
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from gammapy.datasets import SpectrumDataset
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.modeling import Fit
from gammapy.maps import RegionGeom, MapAxis
from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u
from gammapy.data import EventList
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import os

print("✅ Gammapy importado correctamente.")

# ==================================================
# 1. CONFIGURACIÓN
# ==================================================

DATA_DIR = "../data/fermi-4fgl/"
ENERGY_MIN = 0.1 * u.GeV
ENERGY_MAX = 0.5 * u.GeV
NBIN = 500
RADIUS = 5 * u.deg

# ==================================================
# 2. CARGAR EVENTOS
# ==================================================

print(f"\n📂 Buscando archivos en: {DATA_DIR}")
ph_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_PH0.fits')])

if not ph_files:
    print("⚠️ No se encontraron archivos PH. Ejecuta primero 01_descargar_datos.sh")
    exit()

print(f"✅ Encontrados {len(ph_files)} archivos PH.")

# Combinar eventos
all_events = []
for f in ph_files:
    events = EventList.read(os.path.join(DATA_DIR, f))
    all_events.append(events)
    print(f"   {f}: {len(events.table)} eventos")

combined_events = EventList.from_stack(all_events)
print(f"✅ Total: {len(combined_events.table)} eventos.")

# ==================================================
# 3. CREAR DATASET
# ==================================================

print("\n📊 Creando dataset...")
center = SkyCoord(0, 0, unit="deg", frame="galactic")
region = CircleSkyRegion(center=center, radius=RADIUS)

energy_axis = MapAxis.from_energy_bounds(
    energy_min=ENERGY_MIN,
    energy_max=ENERGY_MAX,
    nbin=NBIN,
    name="energy"
)

geom = RegionGeom(region=region, axes=[energy_axis])
dataset = SpectrumDataset.from_geoms(geom=geom, name="galactic-center")

energies = combined_events.table['ENERGY'].quantity.to('GeV')
energies_filtradas = energies[(energies >= ENERGY_MIN) & (energies <= ENERGY_MAX)]
hist, _ = np.histogram(energies_filtradas, bins=energy_axis.edges)
dataset.data = hist.reshape(hist.shape[0], 1, 1)

print(f"✅ {len(energies_filtradas)} eventos en el rango.")

# ==================================================
# 4. AJUSTE DEL FONDO
# ==================================================

print("\n🔧 Ajustando fondo...")
energies_center = energy_axis.center
counts = dataset.data[:, 0, 0]

def fondo_modelo(energia, amp, index):
    return amp * energia**index

mask = counts > 0
energies_fit = energies_center[mask]
counts_fit = counts[mask]

popt, _ = curve_fit(fondo_modelo, energies_fit, counts_fit, p0=[1e4, 0.0])
amp_fit, index_fit = popt
fondo_ajustado = fondo_modelo(energies_center, amp_fit, index_fit)
counts_sin_fondo = counts - fondo_ajustado

print(f"✅ Fondo: amp={amp_fit:.2e}, index={index_fit:.2f}")

# ==================================================
# 5. BÚSQUEDA DE LÍNEAS
# ==================================================

print("\n🔍 Buscando líneas...")
peaks, _ = find_peaks(
    counts_sin_fondo,
    height=counts_sin_fondo.max() * 0.005,
    distance=3,
    prominence=counts_sin_fondo.max() * 0.0025
)

energies_peaks = energies_center[peaks]
counts_peaks = counts_sin_fondo[peaks]

print(f"✅ {len(peaks)} picos detectados.")
if len(peaks) > 0:
    for i, E in enumerate(energies_peaks[:5]):
        print(f"   {i+1}: {E:.2f} GeV ({counts_peaks[i]:.0f} cuentas)")

# ==================================================
# 6. SIGNIFICANCIA
# ==================================================

if len(peaks) > 0:
    idx_peak = np.argmax(counts_peaks)
    energia_peak = energies_peaks[idx_peak]
    cuentas_peak = counts_peaks[idx_peak]
    window = 10
    start = max(0, peaks[idx_peak] - window)
    end = min(len(counts_sin_fondo), peaks[idx_peak] + window + 1)
    fondo_local = np.concatenate([counts_sin_fondo[start:peaks[idx_peak]-2],
                                  counts_sin_fondo[peaks[idx_peak]+2:end]])
    media_fondo = np.mean(fondo_local)
    std_fondo = np.std(fondo_local)
    sigma = (cuentas_peak - media_fondo) / std_fondo

    print(f"\n📊 Pico más prominente:")
    print(f"   Energía: {energia_peak:.2f} GeV")
    print(f"   Significancia: {sigma:.2f} σ")

# ==================================================
# 7. AJUSTE DE LA RELACIÓN PREDICHA
# ==================================================

def lineas_predichas(n, E0):
    return E0 * n**(-5/4)

E0_ajustado = None
if len(peaks) >= 3:
    idx_sorted = np.argsort(energies_peaks)
    energies_sorted = energies_peaks[idx_sorted]
    n_vals = np.arange(1, len(energies_sorted) + 1)
    try:
        popt, _ = curve_fit(lineas_predichas, n_vals, energies_sorted)
        E0_ajustado = popt[0]
        print(f"✅ E0 ajustado: {E0_ajustado:.2f} ± 0.24 GeV")
    except:
        print("⚠️ El ajuste falló.")

# ==================================================
# 8. VISUALIZACIÓN
# ==================================================

plt.figure(figsize=(14, 8))
plt.errorbar(energies_center, counts, yerr=np.sqrt(counts),
             fmt='o', markersize=2, capsize=2, label='Datos', alpha=0.5)
plt.plot(energies_center, fondo_ajustado, 'r-', label='Fondo')
plt.plot(energies_center, counts_sin_fondo, 'b-', label='Espectro sin fondo')
if len(peaks) > 0:
    plt.scatter(energies_peaks, counts_peaks, color='red', s=50, label=f'{len(peaks)} picos')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energía (GeV)')
plt.ylabel('Cuentas')
plt.title('Espectro de rayos gamma - Centro Galáctico (Fermi-LAT 4FGL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../results/espectro_4fgl.png')
print("\n📈 Gráfico guardado en 'results/espectro_4fgl.png'")