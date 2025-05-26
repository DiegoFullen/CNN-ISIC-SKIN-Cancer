import os
import pandas as pd
import CNN.CNN as cnn
import config.settings as config
import Utils.evaluation as evaluate
import Utils.Use as use
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

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
        print("WIP")
        
        """
        X_test, y_test = self.load_data(self.test_path)
        self.model.evaluate(X_test, y_test)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.evaluate.evaluate_model()
        self.evaluate.plot_training_history(self.cnn.history)
        self.evaluate.matrix_confusion(self.evaluate.y_true, self.evaluate.y_pred)
        self.evaluate.display_predictions(self.evaluate.X_test, self.evaluate.predictions, self.evaluate.y_test)
        """
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


if __name__ == "__main__":
    app = Main()
    app.run()