# Projektowanie-i-zastosowania-sieci-neuronowych


# 🧠 Autoenkoder – rekonstrukcja obrazów

Projekt przedstawia prostą implementację autoenkodera w Pythonie, służącego do rekonstrukcji obrazów przy użyciu podstawowych bibliotek.

## 📂 Opis

Autoenkoder wczytuje obrazy z lokalnego folderu, przekształca je na tensory, uczy się ich reprezentacji i dokonuje rekonstrukcji. Wizualizuje oryginalne i odtworzone obrazy przy użyciu `matplotlib`.

## 🛠 Wykorzystane biblioteki

- `os` – obsługa plików i folderów
- `numpy` – operacje macierzowe
- `PIL` (`Pillow`) – ładowanie i przetwarzanie obrazów
- `matplotlib` – wizualizacja

## 🚀 Jak uruchomić

1. **Zainstaluj wymagane biblioteki** :

```bash
pip install numpy pillow matplotlib

KONFIGURACJA venv
    -python -m venv venv
    -venv\Scripts\activate
    INSTALACCJA POTRZEBNYCH BILIOTEK
    -pip install numpy pillow matplotlib
    -