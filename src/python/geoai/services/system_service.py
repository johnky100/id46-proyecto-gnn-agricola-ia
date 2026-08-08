from datetime import datetime


class SystemService:

    def get_version(self):
        return "1.0"

    def get_last_update(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M")


system_service = SystemService()