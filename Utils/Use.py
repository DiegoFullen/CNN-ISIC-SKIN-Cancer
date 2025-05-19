import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix

class UseModel:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.close()
    
    # ------------- Evaluaciones mejoradas del modelo ------------------
    def evaluate_model(self):
        # Crear dataset de prueba
        test_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Hacer evaluación del modelo
        print("\nEvaluando modelo en conjunto de prueba...")
        evaluate_obj = evaluate(self.model, test_ds, self.train_df["benign_malignant"].cat.categories.tolist())
        evaluate_obj.evaluate_model()

        # Visualizar historial de entrenamiento con más métricas
        def plot_extended_training_history(history):
            # Lista de métricas para graficar
            metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
            
            for i, metric in enumerate(metrics):
                if metric in history.history:
                    axes[i].plot(history.history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history.history:
                    axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                
                axes[i].set_title(f'{metric.capitalize()} vs. Epochs')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.savefig('training_metrics_extended.png')
            plt.show()

        # Visualizar historial de entrenamiento del último bloque
        plot_extended_training_history(self.history_per_block[-1])

        # Hacer predicciones en datos de prueba
        predictions = self.model.predict(test_ds)
        true_classes = np.argmax(np.concatenate([y for x, y in test_ds], axis=0), axis=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Obtener los nombres de las clases
        class_names = self.train_df["benign_malignant"].cat.categories.tolist() 

        # Calcular métricas
        self.confusion_matrix = confusion_matrix(true_classes, predicted_classes)
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        
        # Almacenar métricas
        self.precision = report['weighted avg']['precision']
        self.recall_score = report['weighted avg']['recall']
        self.f1_score = report['weighted avg']['f1-score']
        
        # Mostrar la matriz de confusión
        evaluate_obj.matrix_confusion(true_classes, predicted_classes)

        # Encontrar umbral óptimo para mejorar clasificación
        def find_optimal_threshold(y_true, y_pred_proba):
            """
            Encontrar el umbral óptimo basado en la curva PR
            """
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
            
            # Calcular F1 para cada umbral
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Encontrar el umbral con el mejor F1-score
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
            best_f1 = f1_scores[best_f1_idx]
            
            print(f"Umbral óptimo encontrado: {best_threshold:.4f} (F1-score: {best_f1:.4f})")
            
            # Graficar curva ROC
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.savefig('roc_curve.png')
            plt.show()
            
            # Graficar curva Precision-Recall
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, label=f'PR curve (F1-Best = {best_f1:.4f})')
            plt.axvline(x=recall[best_f1_idx], color='r', linestyle='--', 
                    label=f'Threshold = {best_threshold:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid(alpha=0.3)
            plt.savefig('pr_curve.png')
            plt.show()
            
            return best_threshold

        # Mostrar predicciones con umbral optimizado
        y_pred_proba = self.model.predict(self.X_test)
        best_threshold = find_optimal_threshold(true_classes, y_pred_proba)

        # Aplicar umbral optimizado
        predicted_classes_optimized = (y_pred_proba[:, 1] >= best_threshold).astype(int)

        # Mostrar reporte con umbral optimizado
        print("\nReporte de clasificación con umbral optimizado:")
        print(classification_report(true_classes, predicted_classes_optimized, target_names=class_names))

        # Visualizar algunas predicciones
        num_images = 10 # Tomar 10 imágenes de prueba
        sample_images = self.X_test[:num_images]
        sample_predictions = self.model.predict(sample_images)

        # Mostrar el reporte de clasificación
        evaluate_obj.display_predictions(sample_images, sample_predictions, self.y_test[:num_images])

        # Función para visualizar ejemplos clasificados incorrectamente
        def visualize_misclassified(X_test, y_test, predictions, threshold=0.5, num_examples=10):
            """
            Visualizar ejemplos mal clasificados con el umbral optimizado
            """
            y_true = np.argmax(y_test, axis=1)
            y_pred = (predictions[:, 1] >= threshold).astype(int)
            
            # Encontrar índices donde la predicción es incorrecta
            incorrect_indices = np.where(y_true != y_pred)[0]
            
            if len(incorrect_indices) == 0:
                print("No se encontraron ejemplos mal clasificados con este umbral.")
                return
            
            # Seleccionar hasta num_examples ejemplos incorrectos
            selected_indices = incorrect_indices[:min(num_examples, len(incorrect_indices))]
            
            # Configurar visualización
            n_cols = min(5, len(selected_indices))
            n_rows = (len(selected_indices) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
            for i, idx in enumerate(selected_indices):
                # Obtener predicción y etiqueta real
                true_label = class_names[y_true[idx]]
                pred_label = class_names[y_pred[idx]]
                confidence = predictions[idx][y_pred[idx]]
                
                # Mostrar imagen
                axes[i].imshow(cv2.cvtColor(X_test[idx].astype('uint8'), cv2.COLOR_BGR2RGB))
                axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", color='red')
                axes[i].axis('off')
            
            # Ocultar ejes vacíos
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.savefig('misclassified_examples.png')
            plt.show()

        # Visualizar ejemplos mal clasificados
        visualize_misclassified(self.X_test, self.y_test, y_pred_proba, threshold=best_threshold)