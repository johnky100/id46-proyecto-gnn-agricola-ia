class ForecastService:

    def get_status(self):
        return "Disponible"

    def get_last_forecast_year(self):
        return 2025


forecast_service = ForecastService()