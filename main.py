import os
import pandas as pd
import CNN.CNN as cnn
import config.settings as config
import Utils.evaluation as evaluate
import Utils.Use as use
import tensorflow as tf

class Main:
    def __init__(self):
        self.name = "Main Class"
        self.options = {
            '1': self.option1,
            '2': self.option2,
            '3': self.option3,
            '4': self.option4,
            '0': self.exit,
        }
        self.test_path = os.path.join(os.path.dirname(__file__), "Test2")
        self.training_path = os.path.join(os.path.dirname(__file__), "Training2")
        self.running = True
        self.settings = config.Settings()
        self.cnn = cnn.CNN()
        print("Cargando modelo...")
        self.model_path = os.path.join(os.path.dirname(__file__), "Model" + self.settings.get_modelo_actual())
        self.model = tf.keras.models.load_model(self.model_path)
        print("Modelo cargado correctamente.")
        print("Forma de entrada esperada:", self.model.input_shape) # Experimental
        self.class_names = {
            0: "Benign",
            1: "Malignant"
            }
        self.evaluate = evaluate.evaluate(self.model, self.test_path, self.class_names)
        
    def option1(self):
        # Código para iterar el modelo
        self.cnn.train_model()
        
    def option2(self):
        # Código para downgradear modelo
        n_versions = input("¿Cuántas versiones deseas eliminar? (0 para cancelar): ")
        if n_versions.isdigit():
            n_versions = int(n_versions)
            if n_versions > 0:
                self.settings.downgrade_versions(n_versions)
            else:
                print("Operación cancelada.")
        else:
            print("Entrada no válida. Se eliminarán 2 versiones.")
    
    def option3(self):
        # Código para probar el modelo
        use_model = use.UseModel(self.model)
        n = input("¿Cuántas imágenes deseas predecir? (0 para cancelar): ")
        if n.isdigit():
            n = int(n)
            if n > 0:
                use_model.predict(n)
            else:
                print("Operación cancelada.")
                
    def option4(self):
        # Código para crear métricas del modelo
        class_names = {
            0: "Benign",
            1: "Malignant"
        }
        evaluate_obj = evaluate.evaluate(self.model, self.test_path, class_names)
        evaluate_obj.evaluate_model()
        evaluate_obj.plot_training_history(self.cnn.history)
        evaluate_obj.matrix_confusion(self.evaluate.y_true, self.evaluate.y_pred)
        evaluate_obj.display_predictions(self.evaluate.X_test, self.evaluate.predictions, self.evaluate.y_test)
        
    def exit(self):
        self.running = False

    def show_menu(self):
        print("\n")
        print("1. Iterar modelo")
        print("2. Downgrade modelo")
        print("3. Probar modelo")
        print("4. Metricas modelo")
        print("0. Salir")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("Selecciona una opción: ")
            action = self.options.get(choice)
            if action:
                action()
            else:
                print("Opción no válida. Intenta de nuevo.")
    
    def process_verification_folder(self, folder_path, verifier):
        """Procesa todos los pares de fotos en una carpeta (adaptado para la clase)"""
        try:
            results = []
            file_pairs = []
            
            # Identificar archivos
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Encontrar pares (foto + INE)
            foto_files = [f for f in files if '_foto.' in f.lower()]
            
            for foto in foto_files:
                prefix = foto.split('_foto.')[0]
                ine_candidates = [f for f in files if f.startswith(f"{prefix}_ine.")]
                
                if ine_candidates:
                    file_pairs.append((foto, ine_candidates[0]))
            
            if not file_pairs:
                print("\nNo se encontraron pares válidos (formato esperado: '1_foto.jpg', '1_ine.jpg')")
                return None
            
            print(f"\nEncontrados {len(file_pairs)} pares de imágenes:")
            for i, (foto, ine) in enumerate(file_pairs, 1):
                print(f" {i}. {foto} ↔ {ine}")
            
            print("\nIniciando verificación...")
            
            for foto, ine in file_pairs:
                foto_path = os.path.join(folder_path, foto)
                ine_path = os.path.join(folder_path, ine)
                
                result = verifier.verify_faces(foto_path, ine_path, is_ine=True, return_details=True)
                
                row = {
                    'foto': foto,
                    'ine': ine,
                    'match': result['match'],
                    'probability': result['probability'],
                    'cosine_sim': result['similarity']['cosine'],
                    'error': result.get('error', '')
                }
                results.append(row)
                
                status = "✅" if row['match'] else "❌"
                print(f" {status} {foto:<15} ↔ {ine:<15} | Prob: {row['probability']:.1%} | Coseno: {row['cosine_sim']:.3f}")
            
            # Crear DataFrame
            df = pd.DataFrame(results)
            
            return df
        
        except Exception as e:
            print(f"\nError durante el procesamiento: {str(e)}")
            return None


if __name__ == "__main__":
    app = Main()
    app.run()