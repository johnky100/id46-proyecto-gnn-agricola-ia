# graph-02_benchmark.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las dependencias necesarias para ejecutar el Benchmark
# Científico de modelos predictivos utilizando el GraphData construido
# durante la etapa de construcción del grafo.
### Producto: - Librerías cargadas correctamente.
### Responde: ¿Qué dependencias requiere el protocolo experimental para comparar
# objetivamente diferentes modelos predictivos?

# Librerías científicas
import json # Exportación de archivos JSON
import joblib # Persistencia de objetos Python
import numpy as np # Operaciones numéricas
import pandas as pd # Manipulación de datos
import torch # Framework de Deep Learning

# PyTorch Geometric
from torch_geometric.data import (
    Data
) # Objeto GraphData

# Configuración oficial del proyecto
from src.python.config.config_project import (
    PROJECT_SEED,
    BENCHMARK_CONFIG,
    BENCHMARK_MODELS,
    BENCHMARK_REPRODUCIBILITY
) # Configuración oficial del Benchmark

# Rutas oficiales
from src.python.config.paths import (
    GRAPH_DATA_DIR,
    BENCHMARK_RESULTS_FILE,
    BENCHMARK_SUMMARY_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_FILE,
    BEST_MODEL_CONFIG_FILE,
    validate_project_structure
) # Rutas oficiales del proyecto

from utils.results import (
    build_exportable_benchmark_result,
    benchmark_results_to_dataframe
)

# Modelos estadísticos
from src.python.models.statistical import (
    run_linear_regression
)

# Modelos de Machine Learning
from src.python.models.machine_learning import (
    run_machine_learning
)

# Modelos de Deep Learning
from src.python.models.deep_learning import (
    run_mlp
)

# Modelos Graph Neural Networks
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    run_gnn
)

from sklearn.preprocessing import (
    StandardScaler
) # Escalamiento de variables

from src.python.graph.graph_pipeline import prepare_graph_collection

print("-" * 80)

# BLOQUE 2. Configuración del Script --------------------------------------
# Objetivo: Configurar el entorno de ejecución, validar la estructura oficial
# del proyecto e inicializar los parámetros necesarios para ejecutar el
# Benchmark Científico de forma reproducible.
# Producto:
# - Entorno de ejecución configurado.
# - Estructura oficial del proyecto validada.
# - Reproducibilidad establecida.
# - Dispositivo de procesamiento seleccionado.
# Pregunta científica:
# ¿El entorno de ejecución cumple las condiciones necesarias para ejecutar
# un Benchmark Científico reproducible y consistente?

# 2.1 Configuración del entorno -------------------------------------------
print("Configurando entorno de ejecución del Benchmark...")

# 2.2 Validación de la estructura del proyecto ----------------------------
validate_project_structure(verbose=True)
print("Estructura del proyecto validada correctamente.")

# 2.3 Configuración de la reproducibilidad -------------------------------
torch.manual_seed(PROJECT_SEED)
np.random.seed(PROJECT_SEED)

cuda_available = torch.cuda.is_available()

if (
    BENCHMARK_REPRODUCIBILITY["deterministic"]
    and cuda_available
):
    torch.cuda.manual_seed(PROJECT_SEED)
    torch.cuda.manual_seed_all(PROJECT_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Semilla del proyecto          : {PROJECT_SEED}")

# 2.4 Selección del dispositivo ------------------------------------------
DEVICE = "cuda" if cuda_available else "cpu"

print(f"Dispositivo de procesamiento  : {DEVICE.upper()}")

# 2.5 Confirmación del bloque --------------------------------------------
print("Entorno de ejecución configurado correctamente.")
print("-" * 80)

# BLOQUE 3. Carga de la Colección Oficial de GraphData
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Cargar la colección oficial de objetos GraphData generada durante la etapa
# de construcción del grafo para utilizarla como entrada del Benchmark
# Científico.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección oficial GraphData fue cargada correctamente para iniciar
# el Benchmark Científico?
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 3. CARGA DE LA COLECCIÓN OFICIAL GraphData")
print("-" * 80)

#------------------------------------------------------------------------------
# 3.1 Localizar GraphData oficiales
#------------------------------------------------------------------------------

graph_files = sorted(
    GRAPH_DATA_DIR.glob("graph_data_20*.pt")
)

if len(graph_files) == 0:
    raise FileNotFoundError(
        "No se encontraron GraphData oficiales."
    )

print(f"GraphData localizados: {len(graph_files)}")

#------------------------------------------------------------------------------
# 3.2 Cargar la colección oficial
#------------------------------------------------------------------------------

graphs: list[Data] = []

for graph_file in graph_files:

    graph = torch.load(
        graph_file,
        weights_only=False,
    )

    if not isinstance(graph, Data):
        raise TypeError(
            f"'{graph_file.name}' no corresponde a un objeto GraphData."
        )

    graphs.append(graph)

print(f"GraphData cargados correctamente: {len(graphs)}")

#------------------------------------------------------------------------------
# 3.3 Registrar resumen de la colección
#------------------------------------------------------------------------------

reference_graph = graphs[0]

print("\nResumen de la colección GraphData")

print(f"Grafos           : {len(graphs)}")
print(f"Nodos            : {reference_graph.num_nodes:,}")
print(f"Aristas          : {reference_graph.num_edges:,}")
print(f"Node Features    : {reference_graph.num_node_features}")

if hasattr(reference_graph, "y") and reference_graph.y is not None:
    print(
        f"Variable objetivo: {tuple(reference_graph.y.shape)}"
    )

#------------------------------------------------------------------------------
# 3.4 Confirmación
#------------------------------------------------------------------------------

print("\nColección oficial GraphData cargada correctamente.")
print("-" * 80)

# BLOQUE 4. Preparación de la Colección Oficial GraphData
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Preparar la colección oficial GraphData para el Benchmark Científico
# utilizando el pipeline oficial de validación, particionado, construcción
# de máscaras y escalamiento.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección oficial GraphData fue preparada correctamente para ejecutar
# el Benchmark Científico bajo un protocolo reproducible?
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 4. PREPARACIÓN DE LA COLECCIÓN OFICIAL GraphData")
print("-" * 80)

preparation_results = prepare_graph_collection(
    graphs=graphs,
    expected_nodes=BENCHMARK_CONFIG["expected_nodes"],
    expected_features=BENCHMARK_CONFIG["expected_features"],
    expected_years=BENCHMARK_CONFIG["expected_years"],
    expected_edges=BENCHMARK_CONFIG.get("expected_edges"),
    train_size=BENCHMARK_CONFIG["train_size"],
    validation_size=BENCHMARK_CONFIG["validation_size"],
    random_state=PROJECT_SEED,
)

graphs = preparation_results["graphs"]
partitions = preparation_results["partitions"]
scaler = preparation_results["scaler"]
validation_report = preparation_results["validation_report"]

print("Colección GraphData preparada correctamente.")
print("-" * 80)

# BLOQUE 5. Modelos Candidatos --------------------------------------------
# Objetivo: Validar la configuración oficial de los modelos que
# participarán en el Benchmark Científico.
# Producto:
# - BENCHMARK_MODELS validado.
# Pregunta científica:
# ¿La configuración oficial de los modelos del Benchmark es consistente y
# suficiente para ejecutar el Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 5. VALIDACIÓN DE LOS MODELOS DEL BENCHMARK")
print("-" * 80)

# 5.1 Validar familias de modelos -----------------------------------------
if not BENCHMARK_MODELS:
    raise ValueError(
        "No existen familias de modelos registradas."
    )

print(
    f"Familias registradas: {len(BENCHMARK_MODELS)}"
)

# 5.2 Validar modelos registrados -----------------------------------------
total_models = sum(
    len(models)
    for models in BENCHMARK_MODELS.values()
)

if total_models == 0:
    raise ValueError(
        "No existen modelos registrados para ejecutar el Benchmark."
    )

print(
    f"Modelos registrados: {total_models}"
)

# 5.3 Validar familias vacías ---------------------------------------------
empty_families = [
    family
    for family, models in BENCHMARK_MODELS.items()
    if not models
]

if empty_families:
    raise ValueError(
        f"Familias sin modelos registrados: {empty_families}"
    )

print("Todas las familias contienen al menos un modelo.")

# 5.4 Registrar configuración ---------------------------------------------
print("\nResumen de modelos registrados:")

for family, models in BENCHMARK_MODELS.items():
    print(
        f"  - {family}: {len(models)} modelo(s)"
    )

# 5.5 Confirmación del bloque ---------------------------------------------
print("\nModelos del Benchmark validados correctamente.")
print("-" * 80)

# BLOQUE 6. Preparación del Benchmark
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Construir la estructura oficial de datos que será utilizada por todas las
# familias de modelos durante el Benchmark Científico a partir de la colección
# GraphData preparada por el pipeline oficial.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Los datos preparados por el pipeline oficial fueron organizados
# correctamente para ejecutar el Benchmark Científico?
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 6. PREPARACIÓN DEL BENCHMARK")
print("-" * 80)

#------------------------------------------------------------------------------
# 6.1 Recuperar información del pipeline oficial
#------------------------------------------------------------------------------

graphs = preparation_results["graphs"]

partitions = preparation_results["partitions"]

scaler = preparation_results["scaler"]

train_index = partitions["train_indices"]
validation_index = partitions["validation_indices"]
test_index = partitions["test_indices"]

reference_graph = graphs[0]

train_mask = reference_graph.train_mask.cpu().numpy()
validation_mask = reference_graph.validation_mask.cpu().numpy()
test_mask = reference_graph.test_mask.cpu().numpy()

#------------------------------------------------------------------------------
# 6.2 Construcción de conjuntos clásicos
#------------------------------------------------------------------------------

x = reference_graph.x.cpu().numpy()
y = reference_graph.y.cpu().numpy().ravel()

x_train = x[train_mask]
y_train = y[train_mask]

x_validation = x[validation_mask]
y_validation = y[validation_mask]

x_test = x[test_mask]
y_test = y[test_mask]

print("Conjuntos clásicos preparados correctamente.")

#------------------------------------------------------------------------------
# 6.3 Consolidación
#------------------------------------------------------------------------------

benchmark_data = {

    "graphs": graphs,

    "x_train": x_train,
    "y_train": y_train,

    "x_validation": x_validation,
    "y_validation": y_validation,

    "x_test": x_test,
    "y_test": y_test,

    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index,

    "scaler": scaler,
}

#------------------------------------------------------------------------------
# 6.4 Registro
#------------------------------------------------------------------------------

print(f"GraphData                 : {len(graphs)}")
print(f"Entrenamiento             : {len(train_index):,}")
print(f"Validación                : {len(validation_index):,}")
print(f"Prueba                    : {len(test_index):,}")

print("\nDatos del Benchmark preparados correctamente.")
print("-" * 80)

# BLOQUE 7. Benchmark Estadístico -----------------------------------------
# Objetivo: Ejecutar el Benchmark Científico para los modelos estadísticos
# utilizando el protocolo experimental oficial.
# Entradas:
# - benchmark_data
# Producto:
# - statistical_results
# Pregunta científica:
# ¿Cuál es el desempeño de los modelos estadísticos bajo el protocolo oficial
# del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 7. BENCHMARK ESTADÍSTICO")
print("-" * 80)

# 7.1 Ejecutar Benchmark Estadístico --------------------------------------
statistical_results = []

statistical_results.append(
    run_linear_regression(
        x_train=benchmark_data["x_train"],
        y_train=benchmark_data["y_train"],
        x_test=benchmark_data["x_test"],
        y_test=benchmark_data["y_test"]
    )
)

# 7.2 Validación -----------------------------------------------------------
if not statistical_results:
    raise ValueError(
        "No se obtuvieron resultados del Benchmark Estadístico."
    )

print(
    f"Modelos estadísticos ejecutados: {len(statistical_results)}"
)

# 7.3 Confirmación ---------------------------------------------------------
print("Benchmark estadístico ejecutado correctamente.")
print("-" * 80)

# BLOQUE 8. Benchmark Machine Learning ------------------------------------
# Objetivo: Ejecutar el Benchmark Científico para los modelos de Machine
# Learning utilizando el protocolo experimental oficial.
# Entradas:
# - MODEL_CANDIDATES
# - benchmark_data
# Producto:
# - machine_learning_results
# Pregunta científica:
# ¿Cuál es el desempeño de los modelos de Machine Learning bajo el protocolo
# oficial del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 8. BENCHMARK MACHINE LEARNING")
print("-" * 80)

# 8.1 Ejecutar Benchmark Machine Learning ---------------------------------
machine_learning_results = []
for model_name in BENCHMARK_MODELS["machine_learning"]:
    machine_learning_results.append(
        run_machine_learning(
            model_name=model_name,
            x_train=benchmark_data["x_train"],
            y_train=benchmark_data["y_train"],
            x_test=benchmark_data["x_test"],
            y_test=benchmark_data["y_test"]
        )
    )

print("-" * 80)

# BLOQUE 9. Benchmark Deep Learning ---------------------------------------
# Objetivo: Ejecutar el Benchmark Científico para los modelos de Deep Learning
# utilizando el protocolo experimental oficial.
# Entradas:
# - benchmark_data
# Producto:
# - deep_learning_results
# Pregunta científica:
# ¿Cuál es el desempeño de los modelos de Deep Learning bajo el protocolo
# oficial del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 9. BENCHMARK DEEP LEARNING")
print("-" * 80)

# 9.1 Ejecutar Benchmark Deep Learning ------------------------------------
deep_learning_results = []

deep_learning_results.append(
    run_mlp(
        x_train=benchmark_data["x_train"],
        y_train=benchmark_data["y_train"],
        x_test=benchmark_data["x_test"],
        y_test=benchmark_data["y_test"]
    )
)

# 9.2 Validación -----------------------------------------------------------
if not deep_learning_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de Deep Learning."
    )

# 9.3 Registro -------------------------------------------------------------
print(
    f"Modelos de Deep Learning ejecutados: {len(deep_learning_results)}"
)

# 9.4 Confirmación ---------------------------------------------------------
print("Benchmark de Deep Learning ejecutado correctamente.")
print("-" * 80)

# BLOQUE 10. Benchmark Graph Neural Networks ------------------------------
# Objetivo: Ejecutar el Benchmark Científico para todas las arquitecturas
# Graph Neural Networks utilizando el protocolo experimental oficial.
#
# Entradas:
# - BENCHMARK_MODELS
# - GNN_CONFIG
# - benchmark_data
#
# Producto:
# - gnn_results
#
# Pregunta científica:
# ¿Cuál es el desempeño de las arquitecturas Graph Neural Networks bajo el
# protocolo oficial del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 10. BENCHMARK GRAPH NEURAL NETWORKS")
print("-" * 80)

# 10.1 Ejecutar Benchmark Graph Neural Networks ----------------------------

gnn_results = []

for model_name in BENCHMARK_MODELS["graph_neural_networks"]:

    if model_name not in GNN_CONFIG:
        raise ValueError(
            f"No existe configuración para el modelo GNN: {model_name}"
        )

    model_config = GNN_CONFIG[model_name]

    gnn_results.append(
        run_gnn(
            model_config=model_config,
            graphs=benchmark_data["graphs"]
        )
    )

# 10.2 Validación ----------------------------------------------------------

if not gnn_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de modelos Graph Neural Networks."
    )

# 10.3 Registro ------------------------------------------------------------

print(f"Arquitecturas GNN ejecutadas: {len(gnn_results)}")

# 10.4 Confirmación --------------------------------------------------------

print("Benchmark de Graph Neural Networks ejecutado correctamente.")
print("-" * 80)

# BLOQUE 11. Consolidación de Resultados ----------------------------------
# Objetivo: Consolidar los resultados obtenidos por todas las familias de
# modelos evaluadas durante el Benchmark Científico.
# Entradas:
# - statistical_results
# - machine_learning_results
# - deep_learning_results
# - gnn_results
# Producto:
# - benchmark_results
# Pregunta científica:
# ¿Los resultados de todas las familias de modelos fueron consolidados
# correctamente para su análisis comparativo?

print("\n" + "-" * 80)
print("BLOQUE 11. CONSOLIDACIÓN DE RESULTADOS")
print("-" * 80)

# 11.1 Consolidación ------------------------------------------------------
benchmark_results = (
    statistical_results
    + machine_learning_results
    + deep_learning_results
    + gnn_results
)

# 11.2 Validación ---------------------------------------------------------
if not benchmark_results:
    raise ValueError(
        "No existen resultados para consolidar el Benchmark."
    )

# 11.3 Verificación del número de modelos --------------------------------
expected_models = sum(
    len(models)
    for models in BENCHMARK_MODELS.values()
)

if len(benchmark_results) != expected_models:
    raise ValueError(
        "El número de resultados no coincide con los modelos "
        "registrados en el Benchmark."
    )

# 11.4 Registro -----------------------------------------------------------
print(f"Modelos esperados           : {expected_models}")
print(f"Resultados consolidados     : {len(benchmark_results)}")

print(f"Estadísticos                : {len(statistical_results)}")
print(f"Machine Learning            : {len(machine_learning_results)}")
print(f"Deep Learning               : {len(deep_learning_results)}")
print(f"Graph Neural Networks       : {len(gnn_results)}")

# 11.5 Confirmación -------------------------------------------------------
print("Resultados del Benchmark consolidados correctamente.")
print("-" * 80)

# BLOQUE 12. Ranking Científico -------------------------------------------
# Objetivo: Construir el ranking oficial del Benchmark Científico y
# seleccionar la mejor arquitectura Graph Neural Network de acuerdo con la
# métrica oficial definida para el proyecto.
#
# Entradas:
# - benchmark_results
# - BENCHMARK_CONFIG
#
# Producto:
# - benchmark_ranking
# - best_model
# - best_model_config
#
# Pregunta científica:
# ¿Cuál es la arquitectura Graph Neural Network con mejor desempeño bajo el
# protocolo oficial del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 12. RANKING CIENTÍFICO")
print("-" * 80)

# 12.1 Selección de la métrica oficial ------------------------------------

ranking_metric = BENCHMARK_CONFIG["ranking_metric"]

# 12.2 Validación ----------------------------------------------------------

if not benchmark_results:

    raise ValueError(
        "No existen resultados para construir el ranking."
    )

if ranking_metric not in benchmark_results[0]:

    raise KeyError(
        f"La métrica '{ranking_metric}' no existe en los resultados "
        "del Benchmark."
    )

# 12.3 Selección de la familia metodológica oficial ------------------------

gnn_results = [

    result
    for result in benchmark_results
    if result["family"] == "graph_neural_networks"

]

if not gnn_results:

    raise ValueError(
        "El Benchmark no produjo resultados para la familia "
        "Graph Neural Networks."
    )

print(
    f"Arquitecturas GNN evaluadas : {len(gnn_results)}"
)

# 12.4 Construcción del ranking oficial -----------------------------------

benchmark_ranking = sorted(

    gnn_results,
    key=lambda result: result[ranking_metric]

)

# 12.5 Selección del modelo oficial ---------------------------------------

best_model = benchmark_ranking[0]

best_model_config = (
    best_model["model_config"].copy()
)

# 12.6 Registro ------------------------------------------------------------

print(
    f"Modelo oficial : {best_model['model_name']}"
)

print(
    f"Familia        : {best_model['family']}"
)

print(
    f"{ranking_metric.upper():<15}: "
    f"{best_model[ranking_metric]:.6f}"
)

print("-" * 80)


# BLOQUE 13. Selección del Modelo Ganador Global ---------------------------
# Objetivo: Seleccionar el modelo con el mejor desempeño de acuerdo con el
# ranking oficial del Benchmark Científico.
# Entradas:
# - benchmark_ranking
# Producto:
# - best_model_result
# - best_model_config
# Pregunta científica:
# ¿Cuál es el modelo con el mejor desempeño predictivo bajo el protocolo
# oficial del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 13. SELECCIÓN DEL MODELO GANADOR GLOBAL")
print("-" * 80)

# 13.1 Validación del ranking ---------------------------------------------
if not benchmark_ranking:
    raise ValueError(
        "El Benchmark no contiene modelos para seleccionar."
    )

# 13.2 Selección del modelo ganador ---------------------------------------
best_model_result = benchmark_ranking[0]

# 13.3 Recuperación de la configuración -----------------------------------
best_model_config = best_model_result.get(
    "model_config",
    {
        "model_code": best_model_result["model_code"],
        "model_name": best_model_result["model_name"],
        "family": best_model_result["family"]
    }
)

# 13.4 Validación ----------------------------------------------------------
required_keys = [
    "model_code",
    "model_name",
    "family"
]

missing_keys = [
    key
    for key in required_keys
    if key not in best_model_config
]

if missing_keys:

    raise ValueError(
        f"Configuración incompleta del modelo ganador: {missing_keys}"
    )

# 13.5 Registro ------------------------------------------------------------
ranking_metric = BENCHMARK_CONFIG["ranking_metric"]

print("\nModelo ganador del Benchmark Científico\n")

print(f"Código                 : {best_model_config['model_code']}")
print(f"Modelo                 : {best_model_config['model_name']}")
print(f"Familia                : {best_model_config['family']}")
print(f"{ranking_metric.upper():<23}: {best_model_result[ranking_metric]}")

# 13.6 Confirmación --------------------------------------------------------
print("\nModelo ganador global seleccionado correctamente.")
print("-" * 80)

# BLOQUE 14. Selección de los Mejores Modelos por Familia ------------------
# Objetivo: Seleccionar el mejor modelo de cada familia a partir del ranking
# oficial del Benchmark Científico.
# Entradas:
# - benchmark_ranking
# Producto:
# - best_models_by_family
# Pregunta científica:
# ¿Cuál es el mejor modelo dentro de cada familia evaluada por el Benchmark
# Científico?
print("\n" + "-" * 80)
print("BLOQUE 14. SELECCIÓN DE LOS MEJORES MODELOS POR FAMILIA")
print("-" * 80)

# 14.1. Validación ---------------------------------------------------------
## Objetivo: Verificar que existe un ranking oficial.
if not benchmark_ranking:
    raise ValueError(
        "El Benchmark no contiene modelos para seleccionar."
    )

# 14.2. Inicialización -----------------------------------------------------
## Objetivo: Crear el contenedor oficial de los mejores modelos.
best_models_by_family = {}

# 14.3. Selección ----------------------------------------------------------
## Objetivo: Seleccionar el mejor modelo de cada familia.
for result in benchmark_ranking:
    family = result["family"]
    if family not in best_models_by_family:
        best_models_by_family[family] = result

# 14.4. Validación ---------------------------------------------------------
## Objetivo: Verificar que se encontró al menos un modelo por familia.
if not best_models_by_family:
    raise ValueError(
        "No fue posible seleccionar los mejores modelos por familia."
    )

# 14.5. Registro -----------------------------------------------------------
## Objetivo: Mostrar el mejor modelo de cada familia.
ranking_metric = BENCHMARK_CONFIG["ranking_metric"]
print("\nMejores modelos por familia\n")

for family, result in best_models_by_family.items():
    print("-" * 80)
    print(f"Familia : {family}")
    print(f"Código   : {result['model_code']}")
    print(f"Modelo   : {result['model_name']}")
    print(
        f"{ranking_metric.upper():<9}: "
        f"{result[ranking_metric]:.6f}"
    )

print("-" * 80)

# 14.6. Confirmación -------------------------------------------------------
print(
    "\nMejores modelos por familia seleccionados correctamente."
)

print("-" * 80)

# BLOQUE 15. Exportación de Resultados ------------------------------------
# Objetivo: Exportar los productos oficiales generados durante el Benchmark
# Científico para garantizar la reproducibilidad del experimento.
# Entradas:
# - benchmark_results
# - benchmark_ranking
# - best_model_config
# Producto:
# - benchmark_results.joblib
# - benchmark_metrics.parquet
# - benchmark_summary.csv
# - benchmark_ranking.csv
# - best_model_config.json
# Pregunta científica:
# ¿Los productos oficiales del Benchmark fueron exportados correctamente?

print("\n" + "-" * 80)
print("BLOQUE 15. EXPORTACIÓN DE RESULTADOS")
print("-" * 80)

# 15.1 Exportación del Benchmark completo ---------------------------------

joblib.dump(
    benchmark_results,
    BENCHMARK_RESULTS_FILE
)

# 15.2 Construcción de tablas ---------------------------------------------

benchmark_metrics = benchmark_results_to_dataframe(
    benchmark_results
)

benchmark_ranking_df = benchmark_results_to_dataframe(
    benchmark_ranking
)

benchmark_summary = benchmark_metrics[
    [
        "model_code",
        "model_name",
        "family",
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time"
    ]
]

# 15.3 Exportación de tablas ----------------------------------------------

benchmark_metrics_parquet = (
    benchmark_metrics.copy()
)

benchmark_metrics_parquet[
    "model_config"
] = benchmark_metrics_parquet[
    "model_config"
].apply(
    lambda config: json.dumps(
        config,
        ensure_ascii=False
    )
)

benchmark_metrics_parquet.to_parquet(
    BENCHMARK_METRICS_FILE,
    index=False
)

benchmark_ranking_df.to_csv(
    BENCHMARK_RANKING_FILE,
    index=False
)

benchmark_summary.to_csv(
    BENCHMARK_SUMMARY_FILE,
    index=False
)

# 15.4 Exportación del modelo ganador -------------------------------------

best_model_config_export = (
    build_exportable_benchmark_result(
        best_model
    )["model_config"]
)

with open(
    BEST_MODEL_CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_model_config_export,
        file,
        indent=4,
        ensure_ascii=False
    )

# 15.5 Validación ----------------------------------------------------------

exported_files = [
    BENCHMARK_RESULTS_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_FILE,
    BENCHMARK_SUMMARY_FILE,
    BEST_MODEL_CONFIG_FILE
]

missing_files = [
    file
    for file in exported_files
    if not file.exists()
]

if missing_files:

    raise FileNotFoundError(
        "No fue posible exportar todos los productos del "
        f"Benchmark: {missing_files}"
    )

# 15.6 Registro ------------------------------------------------------------

print(
    f"Resultados completos      : {BENCHMARK_RESULTS_FILE.name}"
)

print(
    f"Métricas                  : {BENCHMARK_METRICS_FILE.name}"
)

print(
    f"Ranking                   : {BENCHMARK_RANKING_FILE.name}"
)

print(
    f"Resumen                   : {BENCHMARK_SUMMARY_FILE.name}"
)

print(
    f"Modelo ganador            : {BEST_MODEL_CONFIG_FILE.name}"
)

# 15.7 Confirmación --------------------------------------------------------

print(
    "\nProductos oficiales del Benchmark exportados correctamente."
)

print("-" * 80)

# BLOQUE 16. Auditoría de los Resultados del Benchmark ---------------------
# Objetivo: Auditar los resultados obtenidos por todas las familias de
# modelos sin volver a ejecutar el entrenamiento.
# Entradas:
# - benchmark_results
# Producto:
# - Auditoría detallada de los modelos evaluados.
# Pregunta científica:
# ¿Los resultados obtenidos durante el Benchmark Científico son consistentes
# y comparables entre todas las familias de modelos?

print("\n" + "=" * 80)
print("BLOQUE 16. AUDITORÍA DE LOS RESULTADOS DEL BENCHMARK")
print("=" * 80)

# 16.1 Validación de la estructura ----------------------------------------
required_keys = [
    "model_code",
    "model_name",
    "family",
    "rmse",
    "mae",
    "mape",
    "r2",
    "training_time",
    "inference_time"
]

for result in benchmark_results:

    missing_keys = [
        key
        for key in required_keys
        if key not in result
    ]

    if missing_keys:
        raise ValueError(
            f"El modelo '{result.get('model_name', 'DESCONOCIDO')}' "
            f"no contiene las claves obligatorias: {missing_keys}"
        )

print("Estructura de resultados validada correctamente.")

# 16.2 Auditoría de métricas ----------------------------------------------
print("\n" + "=" * 80)
print("MÉTRICAS OFICIALES")
print("=" * 80)

for result in benchmark_results:

    print("\n" + "-" * 80)

    print(f"Modelo             : {result['model_name']}")
    print(f"Familia            : {result['family']}")
    print(f"RMSE               : {result['rmse']:.6f}")
    print(f"MAE                : {result['mae']:.6f}")
    print(f"MAPE               : {result['mape']:.6f}")
    print(f"R²                 : {result['r2']:.6f}")
    print(f"Training time      : {result['training_time']:.6f}")
    print(f"Inference time     : {result['inference_time']:.6f}")

    loss = result.get("loss")

    if loss is not None:
        print(f"Loss               : {loss:.6f}")

# 16.3 Auditoría de predicciones ------------------------------------------
print("\n" + "=" * 80)
print("PREDICCIONES")
print("=" * 80)

for result in benchmark_results:

    y_pred = result.get("y_pred")

    if y_pred is None:
        continue

    print("\n" + "-" * 80)

    print(f"Modelo             : {result['model_name']}")
    print(f"Predicción mínima  : {y_pred.min():.6f}")
    print(f"Predicción máxima  : {y_pred.max():.6f}")
    print(f"Predicción media   : {y_pred.mean():.6f}")
    print(f"Predicción std     : {y_pred.std():.6f}")

    y_true = result.get("y_true")

    if y_true is not None:

        print(f"Objetivo mínimo    : {y_true.min():.6f}")
        print(f"Objetivo máximo    : {y_true.max():.6f}")
        print(f"Objetivo media     : {y_true.mean():.6f}")
        print(f"Objetivo std       : {y_true.std():.6f}")

# 16.4 Resumen ------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN DE LA AUDITORÍA")
print("=" * 80)

print(f"Modelos auditados  : {len(benchmark_results)}")
print("Estado             : Auditoría completada correctamente.")

print("\n" + "=" * 80)
print("Auditoría finalizada correctamente.")
print("=" * 80)

# -----------------------------------------------------------------------------
# JUSTIFICACIÓN METODOLÓGICA DEL MODELO OFICIAL
#
# El Benchmark Científico ejecuta y evalúa modelos pertenecientes a diferentes
# familias metodológicas con el propósito de establecer una línea base
# cuantitativa que permita comparar el desempeño predictivo de enfoques
# estadísticos, Machine Learning, Deep Learning y Graph Neural Networks bajo
# un protocolo experimental único, reproducible y científicamente consistente.
#
# Los modelos estadísticos, de Machine Learning y de Deep Learning convencional
# son considerados modelos de referencia (baselines), proporcionando un punto
# de comparación objetivo frente a las arquitecturas basadas en grafos y
# permitiendo cuantificar las mejoras obtenidas mediante el modelado espacial.
#
# No obstante, el objetivo científico del presente proyecto no consiste en
# identificar el mejor algoritmo tabular de propósito general, sino en
# desarrollar y optimizar una solución GeoAI orientada al modelado, análisis
# prospectivo y forecasting espacio-temporal del sistema agrícola colombiano
# bajo escenarios de cambio climático.
#
# Desde esta perspectiva, las Graph Neural Networks constituyen la familia
# metodológica más adecuada debido a que representan explícitamente la
# estructura espacial del territorio mediante grafos, modelan relaciones de
# vecindad entre municipios, incorporan dependencias espaciales, permiten la
# propagación de información entre territorios conectados y capturan patrones
# espacio-temporales que los modelos tabulares tradicionales no representan de
# forma explícita.
#
# Adicionalmente, la literatura científica reciente identifica a las Graph
# Neural Networks como una de las principales aproximaciones para el análisis
# de datos geoespaciales, redes complejas, sistemas territoriales, forecasting
# espacio-temporal y problemas donde la estructura de conectividad constituye
# un componente fundamental del fenómeno estudiado.
#
# En consecuencia, la selección del modelo oficial del proyecto se realiza
# exclusivamente dentro de la familia Graph Neural Networks, mientras que las
# demás familias metodológicas permanecen como líneas base (baselines) para la
# comparación científica del desempeño predictivo y la cuantificación de las
# mejoras obtenidas mediante el modelado basado en grafos.
#
# Entre las arquitecturas GNN evaluadas, GraphSAGE es seleccionado como modelo
# oficial debido a que implementa un mecanismo de aprendizaje inductivo capaz
# de generalizar hacia nodos no observados durante el entrenamiento, agregar
# información proveniente de los vecinos de cada nodo, escalar eficientemente a
# grafos de gran tamaño y capturar patrones espaciales complejos manteniendo un
# costo computacional adecuado para aplicaciones territoriales.
#
# Estas características convierten a GraphSAGE en una arquitectura idónea para
# el forecasting espacio-temporal, el análisis de escenarios futuros, la
# integración de variables climáticas, agrícolas y territoriales, así como para
# el desarrollo de plataformas GeoAI orientadas al análisis prospectivo del
# cambio climático, la productividad agrícola, la soberanía alimentaria y la
# inteligencia territorial para el apoyo a la toma de decisiones.
#
# Por tanto, el modelo oficial del proyecto no se selecciona con el propósito
# de determinar el mejor algoritmo de regresión entre todas las familias
# existentes, sino de identificar la arquitectura Graph Neural Network con
# mayor capacidad para representar las dependencias espaciales y temporales del
# fenómeno estudiado, garantizando coherencia entre la metodología propuesta,
# el problema científico abordado y los objetivos de investigación definidos
# para el desarrollo de la plataforma GeoAI.
# -----------------------------------------------------------------------------