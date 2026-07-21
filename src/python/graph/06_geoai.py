# graph-06_geoai.py

# BLOQUE 1. Inicialización de la Plataforma GeoAI
# ------------------------------------------------------------------------------
# Objetivo:
# Inicializar el entorno de ejecución de la Plataforma GeoAI, importar las
# dependencias esenciales, configurar el entorno global, establecer la
# reproducibilidad e inicializar los componentes compartidos por todos los
# módulos de la plataforma.
#
# Pregunta científica:
# ¿El entorno de ejecución garantiza una inicialización reproducible,
# consistente y preparada para integrar los productos científicos generados
# por el pipeline GeoAI?
# ------------------------------------------------------------------------------

# 1.1 Importación de librerías estándar
# ------------------------------------------------------------------------------
from __future__ import annotations

import logging
import platform
import warnings

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 1.2 Importación de librerías científicas
# ------------------------------------------------------------------------------
import networkx as nx
import numpy as np
import pandas as pd
import torch

# 1.3 Importación de librerías geoespaciales
# ------------------------------------------------------------------------------
import geopandas as gpd

# 1.4 Importación de módulos internos del proyecto
# ------------------------------------------------------------------------------
from config import config_project as cfg
from config.config_project import TARGET_VARIABLE
from config import paths

from utils import data_preparation as prep
from utils import results

# 1.5 Configuración global
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

np.set_printoptions(
    suppress=True,
    precision=4
)

# 1.6 Información del proyecto
# ------------------------------------------------------------------------------
PROJECT_NAME = cfg.PROJECT_NAME
PROJECT_VERSION = cfg.PROJECT_VERSION
PROJECT_AUTHORS = cfg.PROJECT_AUTHORS
COUNTRY = cfg.COUNTRY
LANGUAGE = cfg.LANGUAGE

START_TIME = datetime.now()

EXECUTION_ID = START_TIME.strftime("%Y%m%d_%H%M%S")

# 1.7 Configuración de reproducibilidad
# ------------------------------------------------------------------------------
RANDOM_STATE = cfg.PROJECT_SEED

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# 1.8 Configuración del dispositivo de procesamiento
# ------------------------------------------------------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# 1.9 Configuración del sistema de logging
# ------------------------------------------------------------------------------
LOG_FILE = Path(paths.LOGS_DIR) / "geoai.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

LOGGER = logging.getLogger("GeoAI")

# 1.10 Validación de la estructura del proyecto
# ------------------------------------------------------------------------------
def validate_project_structure() -> None:
    """
    Verifica que la estructura oficial del proyecto exista antes de
    iniciar la plataforma GeoAI.
    """

    required_directories = [
        paths.DATA_DIR,
        paths.GRAPHS_DIR,
        paths.MODELS_DIR,
        paths.OUTPUTS_DIR,
        paths.LOGS_DIR,

    ]

    for directory in required_directories:
        if not Path(directory).exists():
            raise FileNotFoundError(
                f"Directorio no encontrado: {directory}"
            )

# 1.11 Inicialización del entorno
# ------------------------------------------------------------------------------
try:
    # Validar la estructura oficial del proyecto
    validate_project_structure()

    # Información del procesador
    processor = platform.processor()
    if not processor:
        processor = "No disponible"

    # Información del investigador principal
    principal_investigator = PROJECT_AUTHORS[0]["name"]

    print("\n" + "-" * 80)
    print("PLATAFORMA INTELIGENTE GEOAI")
    print("-" * 80)

    print(f"Proyecto                 : {PROJECT_NAME}")
    print(f"Versión                  : {PROJECT_VERSION}")
    print(f"Investigador Principal   : {principal_investigator}")
    print(f"País                     : {COUNTRY}")
    print(f"Idioma                   : {LANGUAGE}")

    print("\nEntorno de ejecución")
    print("-" * 80)

    print(f"Python                   : {platform.python_version()}")
    print(f"Sistema Operativo        : {platform.system()}")
    print(f"Procesador               : {processor}")

    print("\nLibrerías")
    print("-" * 80)

    print(f"NumPy                    : {np.__version__}")
    print(f"Pandas                   : {pd.__version__}")
    print(f"PyTorch                  : {torch.__version__}")
    print(f"NetworkX                 : {nx.__version__}")
    print(f"GeoPandas                : {gpd.__version__}")

    print("\nProcesamiento")
    print("-" * 80)

    print(f"CUDA Disponible          : {torch.cuda.is_available()}")
    print(f"Dispositivo              : {DEVICE}")

    print("\nEstado del sistema")
    print("-" * 80)

    print("Configuración global     : OK")
    print("Reproducibilidad         : OK")
    print("Estructura del proyecto  : OK")
    print("Entorno de ejecución     : OK")
    print("Plataforma               : Lista")

    print("-" * 80)

    LOGGER.info("Inicialización de la Plataforma GeoAI completada correctamente.")

except Exception as exc:
    LOGGER.exception("Error durante la inicialización de la Plataforma GeoAI.")

    print("\n" + "-" * 80)
    print("ERROR DURANTE LA INICIALIZACIÓN DE LA PLATAFORMA GEOAI")
    print("-" * 80)
    print(f"Detalle                  : {exc}")
    print("-" * 80)

    raise

# BLOQUE 2. Carga del Dataset Científico
# ------------------------------------------------------------------------------
# Objetivo:
# Cargar y validar el Dataset Científico Certificado mediante la interfaz
# oficial de preparación de datos para su utilización en la construcción del
# GraphData.
#
# Pregunta científica:
# ¿El Dataset Científico se encuentra preparado para iniciar la construcción
# del grafo espacio-temporal?
# ------------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BLOQUE 2. CARGA DEL DATASET CIENTÍFICO")
print("-" * 80)

LOGGER.info("Iniciando la preparación del Dataset Científico.")
try:
    dataset = prep.prepare_dataset()
    print("Dataset Científico preparado correctamente.")
    LOGGER.info(
        "Dataset Científico preparado correctamente."
    )

except Exception as exc:
    LOGGER.exception(
        "Error durante la preparación del Dataset Científico."
    )

    print("\nERROR DURANTE LA PREPARACIÓN DEL DATASET CIENTÍFICO")
    print("-" * 80)
    print(f"Detalle: {exc}")
    print("-" * 80)

    raise

# ------------------------------------------------------------------------------
# BLOQUE 3. Construcción del GraphData
# ------------------------------------------------------------------------------
# Objetivo:
# Construir la representación GraphData a partir del Dataset Científico
# preparado para modelar las relaciones espacio-temporales del sistema
# territorial.
#
# Pregunta científica:
# ¿Es posible representar el Dataset Científico mediante un grafo que
# preserve la estructura espacial y temporal requerida por los modelos GNN?
# ------------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BLOQUE 3. CARGA DEL DATASET CIENTÍFICO")
print("-" * 80)

dataset = prep.prepare_dataset()

print("Dataset Científico preparado correctamente.")
print(f"Registros          : {len(dataset):,}")
print(f"Variables          : {dataset.shape[1]:,}")
print(f"Variable objetivo  : {TARGET_VARIABLE}")

print("\nColumnas del Dataset Científico:")

for i, column in enumerate(dataset.columns, start=1):
    print(f"{i:2d}. {column}")

print(f"\nRegistros          : {len(dataset):,}")
print(f"Total de columnas  : {len(dataset.columns)}")

print("-" * 80)

print("\nCarga del Dataset Científico finalizada correctamente.")
print("-" * 80)