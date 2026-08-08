# build_combined_edges.py

# =============================================================================
# BUILD_COMBINED_EDGES.PY
# Objetivo: Construir las Aristas Combinadas Oficiales del proyecto mediante
# la fusión del Grafo Espacial y el Grafo Dinámico, generando la estructura
# de conectividad final utilizada por el GraphData.
#
# Arquitectura científica
# Entradas: edge_index_spatial y edge_index_dynamic.
# Producto: edge_index_final.
# Pregunta científica: ¿La estructura de conectividad final representa
# correctamente las relaciones espaciales y dinámicas entre los municipios?
# =============================================================================

# =============================================================================
# 1. IMPORTACIÓN DE DEPENDENCIAS
# =============================================================================
import torch # Construcción y manipulación de tensores para PyTorch Geometric

# =============================================================================
# 2. VALIDACIÓN DE LAS ARISTAS DE ENTRADA
# =============================================================================
# Objetivo:
# Verificar la integridad estructural de las Aristas Espaciales y las
# Aristas Dinámicas antes de construir las Aristas Combinadas Oficiales
# del proyecto.
#
# Entradas:
# - edge_index_spatial.
# - edge_index_dynamic.
#
# Producto:
# - Confirmación de la integridad estructural de las Aristas de Entrada.
#
# Responde:
# ¿Las Aristas Espaciales y Dinámicas poseen la estructura necesaria para
# construir correctamente las Aristas Combinadas Oficiales?
# =============================================================================

def validate_input_edges(
    edge_index_spatial: torch.Tensor,
    edge_index_dynamic: torch.Tensor,
) -> None:
    """
    Valida la integridad estructural de las Aristas de Entrada.
    """

    print("-" * 80)
    print("Validación de las Aristas de Entrada")
    print("-" * 80)

    input_edges = {
        "Espaciales": edge_index_spatial,
        "Dinámicas": edge_index_dynamic,
    } # Aristas de entrada del proyecto

    for edge_name, edge_index in input_edges.items():

        # ---------------------------------------------------------------------
        # Validación del tensor
        # ---------------------------------------------------------------------
        if edge_index is None:
            raise ValueError(
                f"Las Aristas {edge_name} no pueden ser None."
            )

        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                f"Las Aristas {edge_name} deben ser un tensor de PyTorch."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                f"Las Aristas {edge_name} deben tener tipo torch.long."
            )

        # ---------------------------------------------------------------------
        # Validación de dimensiones
        # ---------------------------------------------------------------------
        if edge_index.ndim != 2:
            raise ValueError(
                f"El tensor de Aristas {edge_name} debe tener dos dimensiones."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"La primera dimensión del tensor de Aristas {edge_name} debe ser igual a 2."
            )

        if edge_index.shape[1] == 0:
            raise ValueError(
                f"No existen Aristas {edge_name}."
            )

        # ---------------------------------------------------------------------
        # Validación de índices
        # ---------------------------------------------------------------------
        if edge_index.min().item() < 0:
            raise ValueError(
                f"Las Aristas {edge_name} contienen índices negativos."
            )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas de Entrada validadas correctamente.")
    print(f"Aristas espaciales       : {edge_index_spatial.shape[1]:,}")
    print(f"Aristas dinámicas        : {edge_index_dynamic.shape[1]:,}")
    print("Estado                   : CORRECTO")

# =============================================================================
# 3. CONSTRUCCIÓN DE LAS ARISTAS COMBINADAS
# =============================================================================
# Objetivo:
# Construir las Aristas Combinadas Oficiales del proyecto mediante la
# integración de las Aristas Espaciales y las Aristas Dinámicas.
#
# Entradas:
# - edge_index_spatial.
# - edge_index_dynamic.
#
# Producto:
# - edge_index_final.
#
# Responde:
# ¿Las Aristas Combinadas integran correctamente la estructura espacial
# permanente y las relaciones dinámicas entre los municipios?
# =============================================================================

def build_combined_edges(
    edge_index_spatial: torch.Tensor,
    edge_index_dynamic: torch.Tensor,
) -> torch.Tensor:
    """
    Construye las Aristas Combinadas Oficiales del proyecto.

    Parameters
    ----------
    edge_index_spatial : torch.Tensor
        Tensor con las Aristas Espaciales Oficiales.

    edge_index_dynamic : torch.Tensor
        Tensor con las Aristas Dinámicas Oficiales.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_final utilizado por PyTorch Geometric.
    """

    print("-" * 80)
    print("Construcción de las Aristas Combinadas")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Fusión de las Aristas de Entrada
    # -------------------------------------------------------------------------
    edge_index_final = torch.cat(
        [
            edge_index_spatial,
            edge_index_dynamic,
        ],
        dim=1,
    ) # Fusionar las Aristas Espaciales y Dinámicas

    # -------------------------------------------------------------------------
    # Eliminación de aristas duplicadas
    # -------------------------------------------------------------------------
    edge_index_final = torch.unique(
        edge_index_final.T,
        dim=0,
    ).T # Eliminar Aristas Combinadas duplicadas

    # -------------------------------------------------------------------------
    # Validación de la construcción
    # -------------------------------------------------------------------------
    if edge_index_final.numel() == 0:
        raise ValueError(
            "No fue posible construir las Aristas Combinadas."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas Combinadas construidas correctamente.")
    print(f"Aristas espaciales       : {edge_index_spatial.shape[1]:,}")
    print(f"Aristas dinámicas        : {edge_index_dynamic.shape[1]:,}")
    print(f"Aristas combinadas       : {edge_index_final.shape[1]:,}")
    print("Estado                   : CORRECTO")

    return edge_index_final

# =============================================================================
# 4. VALIDACIÓN DE LAS ARISTAS COMBINADAS
# =============================================================================
# Objetivo:
# Verificar la integridad estructural de las Aristas Combinadas Oficiales
# antes de su utilización en la construcción del GraphData.
#
# Entradas:
# - edge_index_final.
# - n_nodes.
#
# Producto:
# - Confirmación de la integridad estructural de las Aristas Combinadas.
#
# Responde:
# ¿Las Aristas Combinadas representan correctamente la estructura de
# conectividad final del proyecto?
# =============================================================================

def validate_combined_edges(
    edge_index_final: torch.Tensor,
    n_nodes: int,
) -> None:
    """
    Valida la integridad estructural de las Aristas Combinadas Oficiales.

    Parameters
    ----------
    edge_index_final : torch.Tensor
        Tensor con las Aristas Combinadas Oficiales.

    n_nodes : int
        Número total de nodos del grafo.

    Returns
    -------
    None
    """

    print("-" * 80)
    print("Validación de las Aristas Combinadas")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Validación del tensor
    # -------------------------------------------------------------------------
    if edge_index_final is None:
        raise ValueError(
            "Las Aristas Combinadas no pueden ser None."
        )

    if not isinstance(edge_index_final, torch.Tensor):
        raise TypeError(
            "Las Aristas Combinadas deben ser un tensor de PyTorch."
        )

    if edge_index_final.dtype != torch.long:
        raise TypeError(
            "Las Aristas Combinadas deben tener tipo torch.long."
        )

    # -------------------------------------------------------------------------
    # Validación de dimensiones
    # -------------------------------------------------------------------------
    if edge_index_final.ndim != 2:
        raise ValueError(
            "El tensor edge_index_final debe tener dos dimensiones."
        )

    if edge_index_final.shape[0] != 2:
        raise ValueError(
            "La primera dimensión de edge_index_final debe ser igual a 2."
        )

    if edge_index_final.shape[1] == 0:
        raise ValueError(
            "No existen Aristas Combinadas."
        )

    # -------------------------------------------------------------------------
    # Validación del número de nodos
    # -------------------------------------------------------------------------
    if n_nodes <= 0:
        raise ValueError(
            "El número de nodos debe ser mayor que cero."
        )

    # -------------------------------------------------------------------------
    # Validación de índices
    # -------------------------------------------------------------------------
    if edge_index_final.min().item() < 0:
        raise ValueError(
            "Existen índices negativos en las Aristas Combinadas."
        )

    if edge_index_final.max().item() >= n_nodes:
        raise ValueError(
            "Existen índices de nodo fuera del rango permitido."
        )

    # -------------------------------------------------------------------------
    # Validación de aristas duplicadas
    # -------------------------------------------------------------------------
    n_unique_edges = torch.unique(
        edge_index_final.T,
        dim=0,
    ).shape[0] # Número de aristas únicas

    if n_unique_edges != edge_index_final.shape[1]:
        raise ValueError(
            "Existen Aristas Combinadas duplicadas."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas Combinadas validadas correctamente.")
    print(f"Nodos                    : {n_nodes:,}")
    print(f"Aristas                  : {edge_index_final.shape[1]:,}")
    print("Estado                   : CORRECTO")

# =============================================================================
# 5. CONSTRUCCIÓN OFICIAL DE LAS ARISTAS COMBINADAS
# =============================================================================
# Objetivo:
# Centralizar la validación y construcción de las Aristas Combinadas
# Oficiales del proyecto para entregar una única interfaz al resto
# del pipeline científico.
#
# Entradas:
# - edge_index_spatial.
# - edge_index_dynamic.
# - n_nodes.
#
# Producto:
# - edge_index_final.
#
# Responde:
# ¿Las Aristas Combinadas Oficiales fueron construidas correctamente
# para representar la estructura de conectividad final del proyecto?
# =============================================================================

def prepare_combined_edges(
    edge_index_spatial: torch.Tensor,
    edge_index_dynamic: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Construye las Aristas Combinadas Oficiales del proyecto.

    Parameters
    ----------
    edge_index_spatial : torch.Tensor
        Tensor con las Aristas Espaciales Oficiales.

    edge_index_dynamic : torch.Tensor
        Tensor con las Aristas Dinámicas Oficiales.

    n_nodes : int
        Número total de nodos del grafo.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_final listo para ser utilizado por
        PyTorch Geometric.
    """

    # -------------------------------------------------------------------------
    # Validación de las Aristas de Entrada
    # -------------------------------------------------------------------------
    validate_input_edges(
        edge_index_spatial=edge_index_spatial,
        edge_index_dynamic=edge_index_dynamic,
    ) # Validar las Aristas de Entrada

    # -------------------------------------------------------------------------
    # Construcción de las Aristas Combinadas
    # -------------------------------------------------------------------------
    edge_index_final = build_combined_edges(
        edge_index_spatial=edge_index_spatial,
        edge_index_dynamic=edge_index_dynamic,
    ) # Construir las Aristas Combinadas

    # -------------------------------------------------------------------------
    # Validación de las Aristas Combinadas
    # -------------------------------------------------------------------------
    validate_combined_edges(
        edge_index_final=edge_index_final,
        n_nodes=n_nodes,
    ) # Validar las Aristas Combinadas

    # -------------------------------------------------------------------------
    # Retorno
    # -------------------------------------------------------------------------
    return edge_index_final