import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class evaluate:
    def __init__(self, model, test_ds, class_names):
        self.model = model
        self.test_ds = test_ds
        self.class_names = class_names

    def evaluate_model(self):
        # Evaluar el modelo en el conjunto de prueba
        test_loss, test_acc, test_auc = self.model.evaluate(self.test_ds)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de precisión
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Precisión del modelo')
        ax1.set_ylabel('Precisión')
        ax1.set_xlabel('Época')
        ax1.legend(['Entrenamiento', 'Validación'], loc='lower right')
        
        # Gráfico de pérdida
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Pérdida del modelo')
        ax2.set_ylabel('Pérdida')
        ax2.set_xlabel('Época')
        ax2.legend(['Entrenamiento', 'Validación'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('Graphics/training_history.png')
        plt.show()
        
    def matrix_confusion(self, y_true, y_pred):
        print("\nMatriz de confusión:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('Graphics/confusion_matrix.png')
        plt.show()
    
    def display_predictions(self, X_test, predictions, y_test, num_images=5):
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        plt.figure(figsize=(15, 10))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(X_test[i])
            color = "green" if predicted_classes[i] == true_classes[i] else "red"
            title = f"Pred: {self.class_names[predicted_classes[i]]}\nReal: {self.class_names[true_classes[i]]}"
            plt.title(title, color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('Graphics/predictions.png')
        plt.show()
            