import os
import json

class Settings:

    def __init__(self, json_path='config/data.json'):
        self.json_path = json_path
        
        # Cargar el archivo JSON si existe, de lo contrario, crear uno nuevo
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as file:
                self.data = json.load(file)
        else:
            self.data = {
                "actual_version": 0,
                "modelo_actual": "",
                "versiones": []
            }
            with open(self.json_path, 'w') as file:
                json.dump(self.data, file, indent=4)
            print(f"Archivo JSON creado en {self.json_path} con datos por defecto.")
        
        # Validar los datos cargados
        self.validate_json()
        
    def validate_json(self):
        if "actual_version" not in self.data:
            self.data["actual_version"] = 0
        if "modelo_actual" not in self.data:
            self.data["modelo_actual"] = ""
        if "versiones" not in self.data:
            self.data["versiones"] = []

    # ------------------------------------------------------
    
    def new_version(self, new_version):
        # Incrementa el contador de versiones
        self.data['actual_version'] += 1
        version_str = f"v{self.data['actual_version']}"
        new_version["version"] = version_str
        
        self.data["versiones"].append(new_version)
        self.data["modelo_actual"] = new_version["archivo"]

        self.save_settings()
        print(f"Nueva versión agregada: {version_str}")
        
    def downgrade_versions(self, no_versiones):
        versiones = self.data.get("versiones", [])
        total = len(versiones)

        if total < no_versiones:
            print("No hay suficientes versiones para eliminar.")
            return

        ## Verificar si quedó al menos una versión
        if self.data["versiones"]:
            nueva_ultima = self.data["versiones"][-1]
            self.data["modelo_actual"] = nueva_ultima["archivo"]
            self.data["actual_version"] = int(nueva_ultima["version"].replace("v", ""))
            print(f"Revertido a {self.data['modelo_actual']} (versión v{self.data['actual_version']}).")
        else:
            self.data["modelo_actual"] = ""
            self.data["actual_version"] = 0
            print("Todas las versiones fueron eliminadas. Estado reiniciado.")
        
        # Guardar
        self.save_settings()
        print(f"Revertido a {self.data['modelo_actual']} (versión v{self.data['actual_version']}).")

    def save_settings(self):
        with open(self.json_path, 'w') as file:
            json.dump(self.data, file, indent=4)
        print("Settings saved successfully.")
        
    def get_actual_version(self):
        return self.data.get("actual_version", 0)
    
    def get_modelo_actual(self):
        return self.data.get("modelo_actual", "")