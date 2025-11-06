# sitecustomize.py
# Shim agar import `distutils` di Python 3.13 mengarah ke setuptools._distutils
import sys, importlib, types

try:
    import setuptools._distutils as _sd
except Exception as e:
    raise RuntimeError("Butuh setuptools yang menyertakan _distutils. Jalankan: pip install -U setuptools") from e

# Modul distutils utama
m = types.ModuleType("distutils")
m.__dict__.update(_sd.__dict__)
sys.modules["distutils"] = m

# Submodul distutils.version yang dipakai PySpark
sys.modules["distutils.version"] = importlib.import_module("setuptools._distutils.version")
