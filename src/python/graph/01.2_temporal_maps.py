# 03_temporal_maps.py

from pathlib import Path
import argparse
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkb, wkt

YEARS = list(range(2006, 2019)) # Definir los años científicos 2006-2018
EXPECTED_MUNICIPALITIES = 1121 # Número esperado de municipios por año
TARGET_COLUMN = "log_rendimiento" # Variable objetivo que se representará en los mapas

REQUIRED_COLUMNS = [
    "cod_mpio",
    "municipio",
    "departamento",
    "anio",
    "geometry",
    TARGET_COLUMN,
] # Columnas mínimas necesarias para construir el atlas

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generar atlas cartográfico de log_rendimiento 2006-2018."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al archivo dataset_gnn_certificado.parquet."
    )

    parser.add_argument(
        "--output",
        default="outputs/atlas_log_rendimiento",
        help="Carpeta donde se guardarán los mapas."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las imágenes PNG."
    )

    return parser.parse_args()

def convert_geometry(value):
    if value is None:
        return None

    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value)) # Convertir geometría WKB a objeto Shapely

    if isinstance(value, str):
        try:
            return wkt.loads(value) # Convertir geometría WKT a objeto Shapely
        except Exception:
            return None

    return value

def load_dataset(input_path):
    dataset = pd.read_parquet(
        input_path,
        columns=REQUIRED_COLUMNS
    ) # Cargar únicamente las columnas necesarias

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing}"
        )

    dataset["anio"] = pd.to_numeric(
        dataset["anio"],
        errors="coerce"
    ).astype("Int64") # Normalizar el año como entero

    dataset[TARGET_COLUMN] = pd.to_numeric(
        dataset[TARGET_COLUMN],
        errors="coerce"
    ) # Convertir la variable objetivo a formato numérico

    dataset["geometry"] = dataset["geometry"].apply(
        convert_geometry
    ) # Convertir las geometrías al formato espacial

    if dataset["geometry"].isna().any():
        raise ValueError(
            "Existen geometrías faltantes o no reconocibles."
        )

    return gpd.GeoDataFrame(
        dataset,
        geometry="geometry",
        crs="EPSG:4326"
    ) # Crear GeoDataFrame usando coordenadas geográficas

def validate_panel(dataset):
    years_found = sorted(
        dataset["anio"].dropna().astype(int).unique().tolist()
    )

    if years_found != YEARS:
        raise ValueError(
            f"Periodo inesperado. Encontrado: {years_found}. "
            f"Esperado: {YEARS}."
        ) # Verificar que el periodo sea exactamente 2006-2018

    counts = (
        dataset
        .groupby("anio")["cod_mpio"]
        .nunique()
        .reindex(YEARS)
    ) # Contar municipios únicos por año

    if counts.isna().any():
        raise ValueError(
            "Existen años sin registros municipales."
        )

    if not (counts == EXPECTED_MUNICIPALITIES).all():
        raise ValueError(
            "El panel no contiene exactamente "
            f"{EXPECTED_MUNICIPALITIES} municipios en cada año.\n"
            f"Conteos encontrados:\n{counts}"
        ) # Validar el panel balanceado

    duplicated = dataset.duplicated(
        subset=["cod_mpio", "anio"]
    )

    if duplicated.any():
        raise ValueError(
            "Existen registros municipio-año duplicados."
        ) # Evitar duplicidad de observaciones

    return counts

def build_municipality_geometry(dataset):
    municipality_geometry = (
        dataset[
            [
                "cod_mpio",
                "municipio",
                "departamento",
                "geometry",
            ]
        ]
        .drop_duplicates("cod_mpio")
        .copy()
    ) # Crear una geometría única por municipio

    if len(municipality_geometry) != EXPECTED_MUNICIPALITIES:
        raise ValueError(
            "El catálogo espacial no contiene exactamente "
            f"{EXPECTED_MUNICIPALITIES} municipios."
        )

    return gpd.GeoDataFrame(
        municipality_geometry,
        geometry="geometry",
        crs=dataset.crs
    )

def calculate_common_scale(dataset):
    values = dataset[TARGET_COLUMN].dropna()
    if values.empty:
        raise ValueError(
            f"No existen valores válidos para {TARGET_COLUMN}."
        )

    return float(values.min()), float(values.max()) # Obtener escala común para todos los años

def create_year_map(
    dataset,
    municipality_geometry,
    year,
    vmin,
    vmax,
    output_path,
    dpi
):
    values = (
        dataset[
            dataset["anio"].astype(int) == year
        ][
            [
                "cod_mpio",
                TARGET_COLUMN,
            ]
        ]
        .copy()
    ) # Extraer la variable objetivo del año seleccionado

    map_data = municipality_geometry.merge(
        values,
        on="cod_mpio",
        how="left",
        validate="one_to_one"
    ) # Asociar los valores temporales a la geometría municipal

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    map_data.plot(
        column=TARGET_COLUMN,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        legend=True,
        linewidth=0.08,
        edgecolor="black",
        missing_kwds={
            "color": "lightgray",
            "label": "Sin dato"
        }
    ) # Construir mapa coroplético municipal

    ax.set_title(
        f"log_rendimiento municipal. Año {year}",
        fontsize=16
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Exportar mapa anual

    plt.close(fig)

def create_change_map(
    dataset,
    municipality_geometry,
    output_path,
    dpi
):
    first_year = (
        dataset[
            dataset["anio"].astype(int) == 2006
        ][
            ["cod_mpio", TARGET_COLUMN]
        ]
        .rename(
            columns={
                TARGET_COLUMN: "log_rendimiento_2006"
            }
        )
    ) # Extraer valores de 2006

    last_year = (
        dataset[
            dataset["anio"].astype(int) == 2018
        ][
            ["cod_mpio", TARGET_COLUMN]
        ]
        .rename(
            columns={
                TARGET_COLUMN: "log_rendimiento_2018"
            }
        )
    ) # Extraer valores de 2018

    change = first_year.merge(
        last_year,
        on="cod_mpio",
        how="inner",
        validate="one_to_one"
    )

    change["cambio_2006_2018"] = (
        change["log_rendimiento_2018"]
        - change["log_rendimiento_2006"]
    ) # Calcular cambio municipal entre 2006 y 2018

    map_data = municipality_geometry.merge(
        change[
            [
                "cod_mpio",
                "cambio_2006_2018",
            ]
        ],
        on="cod_mpio",
        how="left",
        validate="one_to_one"
    )

    limit = float(
        np.nanmax(
            np.abs(
                map_data["cambio_2006_2018"]
            )
        )
    ) # Definir escala simétrica para representar aumentos y disminuciones

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    map_data.plot(
        column="cambio_2006_2018",
        ax=ax,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        legend=True,
        linewidth=0.08,
        edgecolor="black",
        missing_kwds={
            "color": "lightgray",
            "label": "Sin dato"
        }
    ) # Construir mapa de cambio temporal

    ax.set_title(
        "Cambio municipal de log_rendimiento. 2018 respecto a 2006",
        fontsize=16
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Exportar mapa de cambio

    plt.close(fig)

    return change

def calculate_municipal_trends(dataset):
    rows = []

    for municipality_id, group in dataset.groupby("cod_mpio"):
        group = group.sort_values("anio").copy()

        x = group["anio"].astype(float).to_numpy()
        y = group[TARGET_COLUMN].astype(float).to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3:
            continue

        slope, intercept = np.polyfit(
            x[valid],
            y[valid],
            1
        ) # Estimar pendiente temporal mediante regresión lineal

        fitted = slope * x[valid] + intercept
        residuals = y[valid] - fitted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum(
            (y[valid] - np.mean(y[valid])) ** 2
        )

        r2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        ) # Calcular R cuadrado de la tendencia municipal

        rows.append(
            {
                "cod_mpio": municipality_id,
                "municipio": group["municipio"].iloc[0],
                "departamento": group["departamento"].iloc[0],
                "pendiente_log_rendimiento": slope,
                "intercepto": intercept,
                "r2_tendencia": r2,
                "n_anios": int(valid.sum()),
            }
        )

    return pd.DataFrame(rows)

def classify_trends(trends):
    conditions = [
        trends["pendiente_log_rendimiento"] > 0,
        trends["pendiente_log_rendimiento"] < 0,
    ]

    choices = [
        "Aumento",
        "Disminución",
    ]

    trends["clasificacion_evolucion"] = np.select(
        conditions,
        choices,
        default="Estable"
    ) # Clasificar la trayectoria municipal

    return trends

def create_trend_map(
    trends,
    municipality_geometry,
    output_path,
    dpi
):
    map_data = municipality_geometry.merge(
        trends[
            [
                "cod_mpio",
                "pendiente_log_rendimiento",
            ]
        ],
        on="cod_mpio",
        how="left",
        validate="one_to_one"
    )

    limit = float(
        np.nanmax(
            np.abs(
                map_data["pendiente_log_rendimiento"]
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    map_data.plot(
        column="pendiente_log_rendimiento",
        ax=ax,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        legend=True,
        linewidth=0.08,
        edgecolor="black",
        missing_kwds={
            "color": "lightgray",
            "label": "Sin dato"
        }
    ) # Representar espacialmente la tendencia temporal municipal

    ax.set_title(
        "Tendencia municipal de log_rendimiento. 2006-2018",
        fontsize=16
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Exportar mapa de tendencias

    plt.close(fig)

def create_summary(
    dataset,
    counts,
    vmin,
    vmax,
    trends,
    output_path
):
    summary = pd.DataFrame(
        {
            "indicador": [
                "periodo_inicial",
                "periodo_final",
                "numero_anios",
                "municipios_por_anio",
                "observaciones_municipio_anio",
                "min_log_rendimiento",
                "max_log_rendimiento",
                "media_log_rendimiento",
                "mediana_log_rendimiento",
                "municipios_aumento",
                "municipios_disminucion",
                "municipios_estables",
            ],
            "valor": [
                2006,
                2018,
                len(YEARS),
                EXPECTED_MUNICIPALITIES,
                len(dataset),
                vmin,
                vmax,
                dataset[TARGET_COLUMN].mean(),
                dataset[TARGET_COLUMN].median(),
                int(
                    (
                        trends["clasificacion_evolucion"]
                        == "Aumento"
                    ).sum()
                ),
                int(
                    (
                        trends["clasificacion_evolucion"]
                        == "Disminución"
                    ).sum()
                ),
                int(
                    (
                        trends["clasificacion_evolucion"]
                        == "Estable"
                    ).sum()
                ),
            ],
        }
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    ) # Guardar resumen de validación y tendencias


def create_zip(output_dir):
    zip_path = output_dir.with_suffix(".zip")

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(output_dir)
                ) # Incluir todos los productos en el ZIP

    return zip_path

def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annual_dir = output_dir / "mapas_anuales"
    change_dir = output_dir / "cambios"
    trend_dir = output_dir / "tendencias"
    table_dir = output_dir / "tablas"

    for directory in [
        annual_dir,
        change_dir,
        trend_dir,
        table_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        ) # Crear estructura de carpetas del atlas

    print("Cargando dataset científico...")
    dataset = load_dataset(input_path)

    print("Validando panel municipio-año...")
    counts = validate_panel(dataset)

    print("Construyendo catálogo espacial...")
    municipality_geometry = build_municipality_geometry(
        dataset
    )

    print("Calculando escala común...")
    vmin, vmax = calculate_common_scale(
        dataset
    )

    print(
        f"Escala común de {TARGET_COLUMN}: "
        f"{vmin:.6f} a {vmax:.6f}"
    )

    for year in YEARS:
        print(
            f"Generando mapa {year}..."
        )

        create_year_map(
            dataset=dataset,
            municipality_geometry=municipality_geometry,
            year=year,
            vmin=vmin,
            vmax=vmax,
            output_path=(
                annual_dir
                / f"Mapa_{year}_log_rendimiento.png"
            ),
            dpi=args.dpi
        )

    print("Generando mapa de cambio 2006-2018...")

    create_change_map(
        dataset=dataset,
        municipality_geometry=municipality_geometry,
        output_path=(
            change_dir
            / "Mapa_cambio_log_rendimiento_2006_2018.png"
        ),
        dpi=args.dpi
    )

    print("Calculando tendencias municipales...")

    trends = calculate_municipal_trends(
        dataset
    )

    trends = classify_trends(
        trends
    )

    trends.to_csv(
        table_dir / "tendencia_municipal.csv",
        index=False,
        encoding="utf-8-sig"
    ) # Exportar resultados de tendencia por municipio

    trends[
        [
            "cod_mpio",
            "municipio",
            "departamento",
            "clasificacion_evolucion",
        ]
    ].to_csv(
        table_dir / "clasificacion_evolucion.csv",
        index=False,
        encoding="utf-8-sig"
    ) # Exportar clasificación municipal

    create_trend_map(
        trends=trends,
        municipality_geometry=municipality_geometry,
        output_path=(
            trend_dir
            / "Mapa_tendencia_log_rendimiento_2006_2018.png"
        ),
        dpi=args.dpi
    )

    create_summary(
        dataset=dataset,
        counts=counts,
        vmin=vmin,
        vmax=vmax,
        trends=trends,
        output_path=(
            table_dir
            / "resumen_atlas.csv"
        )
    )

    zip_path = create_zip(
        output_dir
    )

    print("")
    print("Atlas generado correctamente.")
    print(
        f"Mapas temporales: {len(YEARS)}"
    )
    print(
        "Mapa de cambio: 1"
    )
    print(
        "Mapa de tendencia: 1"
    )
    print(
        f"Municipios por año: {EXPECTED_MUNICIPALITIES}"
    )
    print(
        f"Observaciones: {len(dataset):,}"
    )
    print(
        f"Rango de log_rendimiento: "
        f"{vmin:.6f} a {vmax:.6f}"
    )
    print(
        f"Archivos: {output_dir}"
    )
    print(
        f"ZIP: {zip_path}"
    )

if __name__ == "__main__":
    main()