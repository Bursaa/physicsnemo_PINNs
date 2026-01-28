# Mathematical Pendulum - Physics-Informed Neural Network (PINN)

## Opis

Ten projekt rozwiązuje równanie wahadła matematycznego przy użyciu Physics-Informed Neural Networks (PINN).

**Równanie ruchu:**

```
d²θ/dt² + (g/l)·sin(θ) = 0
```

gdzie:

- θ - kąt wychylenia [rad]
- g = 9.81 m/s² - przyspieszenie ziemskie
- l = 1.0 m - długość wahadła

**Warunki początkowe:**

- θ(0) = 0.2 rad (≈ 11.5°)
- θ'(0) = 0 rad/s

## Struktura projektu

```
pendulum_ode/
├── pendulum_ode.py           # Definicja równania ODE wahadła
├── pendulum_solver.py         # Solver PINN
├── plot_results_pendulum.py   # Skrypt wizualizacji wyników
└── conf/
    └── config.yaml            # Konfiguracja treningu
```

## Użycie

### 1. Trening modelu

```bash
cd pendulum_ode
python pendulum_solver.py
```

Trening zajmuje ~5-10 minut (20000 iteracji).

### 2. Wizualizacja wyników

```bash
python plot_results_pendulum.py
```

Wygeneruje wykres `pendulum_comparison.png` z:

- Porównaniem rozwiązania PINN z analitycznym
- Błędem aproksymacji
- Statystykami dokładności

## Konfiguracja

Plik `conf/config.yaml` zawiera parametry treningu:

```yaml
arch:
  fully_connected:
    layer_size: 128 # Neurony w warstwie ukrytej
    nr_layers: 6 # Liczba warstw ukrytych

training:
  max_steps: 20000 # Liczba iteracji

batch_size:
  IC: 20 # Batch size dla warunków początkowych
  interior: 1000 # Batch size dla punktów wewnętrznych
```

## Wyniki

Typowa dokładność:

- **Mean absolute error:** < 0.001 rad (< 0.06°)
- **RMSE:** < 0.002 rad (< 0.11°)

Wyniki są zapisywane w katalogu `outputs/pendulum_solver/`.

## Porównanie z rozwiązaniem analitycznym

Dla małych kątów (θ₀ << 1), rozwiązanie analityczne:

```
θ(t) ≈ θ₀·cos(ωt), gdzie ω = √(g/l)
```

PINN uczy się również dla większych kątów, gdzie to przybliżenie nie jest dokładne.

## Wymagania

- Python 3.10+
- PhysicsNeMo Sym
- NumPy
- Matplotlib
- SymPy
