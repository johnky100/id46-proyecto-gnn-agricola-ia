from pathlib import Path
import argparse
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkb, wkt


YEARS = list(range(2006, 2019)) # Definir los años científicos 2006-2018
EXPECTED_MUNICIPALITIES = 1121 # Definir el número oficial de municipios
CRS_EPSG = 4686 # Utilizar MAGNA-SIRGAS como sistema oficial del proyecto

CLIMATE_COLUMNS = [
    "precip_total_anual",
    "desviacion_precip_mensual",
    "coeficiente_variacion_precip",
    "precip_minima_mensual",
    "temp_punto_rocio_era5",
    "evaporacion_era5",
    "indice_area_foliar_vegetacion_alta",
    "evaporacion_potencial_era5",
    "escorrentia_total",
    "escorrentia_superficial",
    "radiacion_termica_descendente",
    "temperatura_aire_era5",
    "precipitacion_total_era5",
    "componente_u_viento",
    "componente_v_viento",
] # Variables climáticas oficiales del Dataset Científico

CLIMATE_LABELS = {
    "precip_total_anual": "Precipitación total anual",
    "desviacion_precip_mensual": "Desviación mensual de precipitación",
    "coeficiente_variacion_precip": "Coeficiente de variación de precipitación",
    "precip_minima_mensual": "Precipitación mínima mensual",
    "temp_punto_rocio_era5": "Temperatura del punto de rocío ERA5",
    "evaporacion_era5": "Evaporación ERA5",
    "indice_area_foliar_vegetacion_alta": "Índice de área foliar de vegetación alta ERA5",
    "evaporacion_potencial_era5": "Evaporación potencial ERA5",
    "escorrentia_total": "Escorrentía total",
    "escorrentia_superficial": "Escorrentía superficial",
    "radiacion_termica_descendente": "Radiación térmica descendente",
    "temperatura_aire_era5": "Temperatura del aire ERA5",
    "precipitacion_total_era5": "Precipitación total ERA5",
    "componente_u_viento": "Componente U del viento",
    "componente_v_viento": "Componente V del viento",
} # Nombres científicos para títulos y reportes

CLIMATE_UNITS = {
    "precip_total_anual": "mm",
    "desviacion_precip_mensual": "mm",
    "coeficiente_variacion_precip": "adimensional",
    "precip_minima_mensual": "mm",
    "temp_punto_rocio_era5": "K",
    "evaporacion_era5": "m",
    "indice_area_foliar_vegetacion_alta": "adimensional",
    "evaporacion_potencial_era5": "m",
    "escorrentia_total": "m",
    "escorrentia_superficial": "m",
    "radiacion_termica_descendente": "J/m²",
    "temperatura_aire_era5": "K",
    "precipitacion_total_era5": "m",
    "componente_u_viento": "m/s",
    "componente_v_viento": "m/s",
} # Unidades documentadas para facilitar la lectura cartográfica


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generar atlas cartográfico de las variables climáticas "
            "del Dataset Científico para 2006-2018."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al archivo dataset_gnn_certificado.parquet."
    )

    parser.add_argument(
        "--output",
        default="src/python/outputs/atlas_climatico_2006_2018",
        help="Carpeta de salida del atlas climático."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las imágenes PNG."
    )

    parser.add_argument(
        "--only-annual",
        action="store_true",
        help="Generar solamente los mapas anuales."
    )

    parser.add_argument(
        "--only-change",
        action="store_true",
        help="Generar solamente los mapas de cambio 2006-2018."
    )

    parser.add_argument(
        "--only-trend",
        action="store_true",
        help="Generar solamente los mapas de tendencia 2006-2018."
    )

    return parser.parse_args()


def convert_geometry(value):
    if value is None:
        return None

    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value)) # Convertir geometría WKB a geometría Shapely

    if isinstance(value, str):
        try:
            return wkt.loads(value) # Convertir geometría WKT a geometría Shapely
        except Exception:
            return None

    return value


def load_dataset(input_path):
    required_columns = [
        "cod_municipio",
        "municipio",
        "departamento",
        "anio",
        "geometry",
    ] + CLIMATE_COLUMNS

    dataset = pd.read_parquet(
        input_path,
        columns=required_columns
    ) # Cargar únicamente las columnas necesarias para el atlas

    missing = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing}"
        ) # Detener ejecución si faltan variables oficiales

    dataset["anio"] = pd.to_numeric(
        dataset["anio"],
        errors="coerce"
    ).astype("Int64") # Normalizar la dimensión temporal

    for column in CLIMATE_COLUMNS:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce"
        ) # Convertir cada variable climática a formato numérico

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
        crs=f"EPSG:{CRS_EPSG}"
    ) # Crear GeoDataFrame con el CRS oficial


def validate_panel(dataset):
    years_found = sorted(
        dataset["anio"].dropna().astype(int).unique().tolist()
    )

    if years_found != YEARS:
        raise ValueError(
            f"Periodo inesperado. Encontrado: {years_found}. "
            f"Esperado: {YEARS}."
        ) # Validar exactamente el periodo 2006-2018

    counts = (
        dataset
        .groupby("anio")["cod_municipio"]
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
            f"{EXPECTED_MUNICIPALITIES} municipios por año.\n"
            f"Conteos encontrados:\n{counts}"
        ) # Validar el panel balanceado

    if dataset.duplicated(
        subset=["cod_municipio", "anio"]
    ).any():
        raise ValueError(
            "Existen registros municipio-año duplicados."
        ) # Garantizar una observación por municipio y año

    return counts


def build_municipality_geometry(dataset):
    municipality_geometry = (
        dataset[
            [
                "cod_municipio",
                "municipio",
                "departamento",
                "geometry",
            ]
        ]
        .drop_duplicates("cod_municipio")
        .copy()
    ) # Construir una geometría única por municipio

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


def calculate_scales(dataset):
    scales = {}

    for variable in CLIMATE_COLUMNS:
        values = dataset[variable].dropna()

        if values.empty:
            raise ValueError(
                f"No existen valores válidos para {variable}."
            )

        vmin = float(values.min())
        vmax = float(values.max())

        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0 # Evitar una escala degenerada

        scales[variable] = {
            "vmin": vmin,
            "vmax": vmax,
        } # Calcular una escala común para cada variable climática

    return scales


def create_annual_map(
    dataset,
    municipality_geometry,
    variable,
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
                "cod_municipio",
                variable,
            ]
        ]
        .copy()
    ) # Extraer los valores de la variable para el año

    map_data = municipality_geometry.merge(
        values,
        on="cod_municipio",
        how="left",
        validate="one_to_one"
    ) # Asociar datos climáticos a geometrías municipales

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    map_data.plot(
        column=variable,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        legend=True,
        linewidth=0.08,
        edgecolor="black",
        missing_kwds={
            "color": "lightgray",
            "label": "Sin dato",
        },
    ) # Construir mapa coroplético municipal

    label = CLIMATE_LABELS[variable]
    unit = CLIMATE_UNITS[variable]

    ax.set_title(
        f"{label}. Año {year}",
        fontsize=16
    )

    ax.text(
        0.01,
        0.01,
        f"Unidad: {unit}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
    ) # Añadir unidad de la variable

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Guardar mapa anual

    plt.close(fig)


def create_change_map(
    dataset,
    municipality_geometry,
    variable,
    output_path,
    dpi
):
    first = (
        dataset[
            dataset["anio"].astype(int) == 2006
        ][
            [
                "cod_municipio",
                variable,
            ]
        ]
        .rename(
            columns={
                variable: "valor_2006",
            }
        )
    ) # Extraer valores de 2006

    last = (
        dataset[
            dataset["anio"].astype(int) == 2018
        ][
            [
                "cod_municipio",
                variable,
            ]
        ]
        .rename(
            columns={
                variable: "valor_2018",
            }
        )
    ) # Extraer valores de 2018

    change = first.merge(
        last,
        on="cod_municipio",
        how="inner",
        validate="one_to_one"
    )

    change["cambio_2006_2018"] = (
        change["valor_2018"]
        - change["valor_2006"]
    ) # Calcular cambio absoluto entre 2006 y 2018

    map_data = municipality_geometry.merge(
        change[
            [
                "cod_municipio",
                "cambio_2006_2018",
            ]
        ],
        on="cod_municipio",
        how="left",
        validate="one_to_one"
    )

    values = map_data["cambio_2006_2018"].dropna()

    if values.empty:
        raise ValueError(
            f"No existen cambios calculables para {variable}."
        )

    limit = float(
        np.nanmax(
            np.abs(values)
        )
    )

    if np.isclose(limit, 0):
        limit = 1.0 # Evitar una escala simétrica degenerada

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
            "label": "Sin dato",
        },
    ) # Construir mapa divergente de cambio temporal

    label = CLIMATE_LABELS[variable]
    unit = CLIMATE_UNITS[variable]

    ax.set_title(
        f"Cambio 2018 respecto a 2006. {label}",
        fontsize=16
    )

    ax.text(
        0.01,
        0.01,
        f"Unidad: {unit}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
    ) # Añadir unidad

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Guardar mapa de cambio

    plt.close(fig)

    return change


def calculate_trends(dataset, variable):
    rows = []

    for municipality_id, group in dataset.groupby("cod_municipio"):
        group = group.sort_values("anio").copy()

        x = group["anio"].astype(float).to_numpy()
        y = group[variable].astype(float).to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3:
            continue

        slope, intercept = np.polyfit(
            x[valid],
            y[valid],
            1
        ) # Estimar tendencia temporal utilizando los 13 años

        fitted = slope * x[valid] + intercept
        residuals = y[valid] - fitted

        ss_res = np.sum(
            residuals ** 2
        )

        ss_tot = np.sum(
            (
                y[valid]
                - np.mean(y[valid])
            ) ** 2
        )

        r2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        ) # Calcular R cuadrado de la tendencia

        rows.append(
            {
                "cod_municipio": municipality_id,
                "municipio": group["municipio"].iloc[0],
                "departamento": group["departamento"].iloc[0],
                "variable": variable,
                "pendiente": slope,
                "intercepto": intercept,
                "r2_tendencia": r2,
                "n_anios": int(valid.sum()),
            }
        )

    return pd.DataFrame(rows)


def create_trend_map(
    trends,
    municipality_geometry,
    variable,
    output_path,
    dpi
):
    map_data = municipality_geometry.merge(
        trends[
            [
                "cod_municipio",
                "pendiente",
            ]
        ],
        on="cod_municipio",
        how="left",
        validate="one_to_one"
    ) # Asociar tendencia temporal a la geometría

    values = map_data["pendiente"].dropna()

    if values.empty:
        raise ValueError(
            f"No existen tendencias calculables para {variable}."
        )

    limit = float(
        np.nanmax(
            np.abs(values)
        )
    )

    if np.isclose(limit, 0):
        limit = 1.0 # Evitar una escala simétrica degenerada

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    map_data.plot(
        column="pendiente",
        ax=ax,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        legend=True,
        linewidth=0.08,
        edgecolor="black",
        missing_kwds={
            "color": "lightgray",
            "label": "Sin dato",
        },
    ) # Representar espacialmente la tendencia climática

    label = CLIMATE_LABELS[variable]
    unit = CLIMATE_UNITS[variable]

    ax.set_title(
        f"Tendencia 2006-2018. {label}",
        fontsize=16
    )

    ax.text(
        0.01,
        0.01,
        f"Pendiente anual. Unidad: {unit}/año",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
    ) # Documentar la unidad de la pendiente

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    ) # Guardar mapa de tendencia

    plt.close(fig)


def create_variable_summary(
    dataset,
    variable,
    scale,
    trends,
    output_path
):
    rows = []

    for year in YEARS:
        values = dataset[
            dataset["anio"].astype(int) == year
        ][variable]

        rows.append(
            {
                "variable": variable,
                "anio": year,
                "media": values.mean(),
                "mediana": values.median(),
                "sd": values.std(),
                "min": values.min(),
                "max": values.max(),
                "n": values.notna().sum(),
            }
        )

    summary = pd.DataFrame(rows)

    trend_summary = pd.DataFrame(
        {
            "variable": [variable],
            "escala_min": [scale["vmin"]],
            "escala_max": [scale["vmax"]],
            "media_pendiente_municipal": [
                trends["pendiente"].mean()
            ],
            "mediana_pendiente_municipal": [
                trends["pendiente"].median()
            ],
            "municipios_pendiente_positiva": [
                int(
                    (
                        trends["pendiente"] > 0
                    ).sum()
                )
            ],
            "municipios_pendiente_negativa": [
                int(
                    (
                        trends["pendiente"] < 0
                    ).sum()
                )
            ],
            "municipios_pendiente_cero": [
                int(
                    (
                        trends["pendiente"] == 0
                    ).sum()
                )
            ],
        }
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    ) # Guardar resumen anual de la variable

    trend_summary.to_csv(
        output_path.with_name(
            f"{variable}_resumen_tendencia.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    ) # Guardar resumen de tendencia municipal


def create_master_summary(
    dataset,
    scales,
    all_trends,
    output_path
):
    rows = []

    for variable in CLIMATE_COLUMNS:
        values = dataset[variable]

        trend = all_trends.get(variable)

        rows.append(
            {
                "variable": variable,
                "nombre": CLIMATE_LABELS[variable],
                "unidad": CLIMATE_UNITS[variable],
                "min_panel": values.min(),
                "max_panel": values.max(),
                "media_panel": values.mean(),
                "sd_panel": values.std(),
                "escala_mapa_min": scales[variable]["vmin"],
                "escala_mapa_max": scales[variable]["vmax"],
                "media_pendiente": (
                    trend["pendiente"].mean()
                    if trend is not None
                    else np.nan
                ),
                "mediana_pendiente": (
                    trend["pendiente"].median()
                    if trend is not None
                    else np.nan
                ),
                "municipios_aumento": (
                    int(
                        (
                            trend["pendiente"] > 0
                        ).sum()
                    )
                    if trend is not None
                    else np.nan
                ),
                "municipios_disminucion": (
                    int(
                        (
                            trend["pendiente"] < 0
                        ).sum()
                    )
                    if trend is not None
                    else np.nan
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    ) # Guardar resumen maestro del atlas climático


def create_zip(output_dir):
    zip_path = output_dir.with_suffix(".zip")

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in sorted(
            output_dir.rglob("*")
        ):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(
                        output_dir
                    )
                ) # Empaquetar todos los productos del atlas

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
    change_dir = output_dir / "cambios_2006_2018"
    trend_dir = output_dir / "tendencias_2006_2018"
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
        ) # Crear estructura de carpetas del atlas climático

    print("Cargando Dataset Científico...")
    dataset = load_dataset(
        input_path
    )

    print("Validando panel 2006-2018...")
    counts = validate_panel(
        dataset
    )

    print("Construyendo catálogo espacial...")
    municipality_geometry = build_municipality_geometry(
        dataset
    )

    print("Calculando escalas comunes...")
    scales = calculate_scales(
        dataset
    )

    all_trends = {}

    generate_annual = not (
        args.only_change
        or args.only_trend
    )

    generate_change = not (
        args.only_annual
        or args.only_trend
    )

    generate_trend = not (
        args.only_annual
        or args.only_change
    )

    for variable in CLIMATE_COLUMNS:
        variable_dir = annual_dir / variable

        if generate_annual:
            variable_dir.mkdir(
                parents=True,
                exist_ok=True
            ) # Crear carpeta anual específica de la variable

            print(
                f"Generando mapas anuales: {variable}"
            )

            for year in YEARS:
                create_annual_map(
                    dataset=dataset,
                    municipality_geometry=municipality_geometry,
                    variable=variable,
                    year=year,
                    vmin=scales[variable]["vmin"],
                    vmax=scales[variable]["vmax"],
                    output_path=(
                        variable_dir
                        / f"{variable}_{year}.png"
                    ),
                    dpi=args.dpi
                )

        if generate_change:
            print(
                f"Generando mapa de cambio: {variable}"
            )

            create_change_map(
                dataset=dataset,
                municipality_geometry=municipality_geometry,
                variable=variable,
                output_path=(
                    change_dir
                    / f"{variable}_cambio_2006_2018.png"
                ),
                dpi=args.dpi
            )

        if generate_trend:
            print(
                f"Calculando tendencia: {variable}"
            )

            trends = calculate_trends(
                dataset,
                variable
            )

            all_trends[variable] = trends

            trends.to_csv(
                table_dir
                / f"{variable}_tendencia_municipal.csv",
                index=False,
                encoding="utf-8-sig"
            ) # Exportar pendiente municipal de la variable

            create_trend_map(
                trends=trends,
                municipality_geometry=municipality_geometry,
                variable=variable,
                output_path=(
                    trend_dir
                    / f"{variable}_tendencia_2006_2018.png"
                ),
                dpi=args.dpi
            )

            create_variable_summary(
                dataset=dataset,
                variable=variable,
                scale=scales[variable],
                trends=trends,
                output_path=(
                    table_dir
                    / f"{variable}_resumen_anual.csv"
                )
            )

    if generate_trend:
        create_master_summary(
            dataset=dataset,
            scales=scales,
            all_trends=all_trends,
            output_path=(
                table_dir
                / "resumen_maestro_variables_climaticas.csv"
            )
        )

    metadata = pd.DataFrame(
        {
            "indicador": [
                "periodo_inicial",
                "periodo_final",
                "numero_anios",
                "municipios_por_anio",
                "observaciones",
                "numero_variables_climaticas",
            ],
            "valor": [
                2006,
                2018,
                len(YEARS),
                EXPECTED_MUNICIPALITIES,
                len(dataset),
                len(CLIMATE_COLUMNS),
            ],
        }
    )

    metadata.to_csv(
        table_dir / "metadata_atlas_climatico.csv",
        index=False,
        encoding="utf-8-sig"
    ) # Guardar metadatos generales del atlas

    zip_path = create_zip(
        output_dir
    )

    annual_count = (
        len(CLIMATE_COLUMNS) * len(YEARS)
        if generate_annual
        else 0
    )

    change_count = (
        len(CLIMATE_COLUMNS)
        if generate_change
        else 0
    )

    trend_count = (
        len(CLIMATE_COLUMNS)
        if generate_trend
        else 0
    )

    print("")
    print("Atlas climático generado correctamente.")
    print(
        f"Periodo: {YEARS[0]}-{YEARS[-1]}"
    )
    print(
        f"Municipios por año: {EXPECTED_MUNICIPALITIES}"
    )
    print(
        f"Observaciones: {len(dataset):,}"
    )
    print(
        f"Variables climáticas: {len(CLIMATE_COLUMNS)}"
    )
    print(
        f"Mapas anuales: {annual_count}"
    )
    print(
        f"Mapas de cambio: {change_count}"
    )
    print(
        f"Mapas de tendencia: {trend_count}"
    )
    print(
        f"Salida: {output_dir}"
    )
    print(
        f"ZIP: {zip_path}"
    )


if __name__ == "__main__":
    main()
