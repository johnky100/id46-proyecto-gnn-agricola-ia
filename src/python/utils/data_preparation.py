import pandas as pd # Manipulación de datos


def prepare_tabular_features(
    dataset,
    target_variable,
    excluded_columns
):
    """
    Preparar las variables predictoras oficiales del proyecto.
    """

    feature_columns = (
        dataset.drop(
            columns = (
                excluded_columns
                + [target_variable]
            ),
            errors = "ignore"
        )
        .select_dtypes(
            include = [
                "number",
                "bool"
            ]
        )
        .columns
        .tolist()
    ) # Variables predictoras

    x_data = dataset[
        feature_columns
    ] # Matriz de entrada

    y_true = dataset[
        target_variable
    ].to_numpy() # Variable objetivo

    return {
        "feature_columns": feature_columns,
        "x_data": x_data,
        "y_true": y_true,
        "n_features": len(feature_columns),
        "n_samples": len(dataset)
    }