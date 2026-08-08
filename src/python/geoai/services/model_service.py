class ModelService:

    def get_official_model(self):
        return "GraphSAGE"

    def get_training_status(self):
        return "Entrenado"

    def get_last_rmse(self):
        return 0.07597


model_service = ModelService()