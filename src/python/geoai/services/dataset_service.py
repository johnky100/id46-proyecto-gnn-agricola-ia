# =============================================================================
# DATASET SERVICE
# =============================================================================

from pathlib import Path


class DatasetService:

    def get_num_municipios(self):
        return 1121

    def get_num_variables(self):
        return 36

    def get_periodo(self):
        return "2006 - 2018"

    def get_num_registros(self):
        return 14573


dataset_service = DatasetService()