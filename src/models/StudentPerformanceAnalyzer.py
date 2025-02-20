import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class StudentPerformanceAnalyzer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.best_model = None
        self.feature_importance = None
        self.r2_score = None
        self.rmse = None
        self.cv_scores = None

    def prepare_data(self, df):
        """
        Prepara los datos con la nueva distribución 80:20
        """
        processed_df = self.preprocessor.fit_transform(df)
        X = processed_df.drop('Exam_Score', axis=1)
        y = df['Exam_Score']

        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    def train_model(self, X_train, y_train):
        """
        Entrena múltiples modelos y selecciona el mejor
        """
        # Definir modelos y parámetros para búsqueda
        models = {
            'ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                }
            },
            'elasticnet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
        }

        best_score = -float('inf')

        # Validación cruzada con 5 folds
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model_info in models.items():
            # Búsqueda de hiperparámetros con validación cruzada
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=kfold,
                scoring='r2',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
                self.best_params = grid_search.best_params_

        # Calcular scores de validación cruzada para el mejor modelo
        self.cv_scores = cross_val_score(
            self.best_model, X_train, y_train,
            cv=kfold, scoring='r2'
        )

    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo con métricas adicionales
        """
        y_pred = self.best_model.predict(X_test)

        self.r2_score = r2_score(y_test, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calcular importancia de características
        self.feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(self.best_model.coef_)
        }).sort_values('importance', ascending=False)

        # Calcular residuos
        self.residuals = y_test - y_pred

    def analyze_performance(self, df):
        """
        Realiza el análisis completo con el modelo mejorado
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

        report = self._generate_report()

        return report

    def _generate_report(self):
        """
        Genera un reporte más detallado
        """
        report = {
            'model_info': {
                'type': self.best_model_name,
                'best_params': self.best_params
            },
            'model_performance': {
                'r2_score': self.r2_score,
                'rmse': self.rmse,
                'cv_mean_r2': self.cv_scores.mean(),
                'cv_std_r2': self.cv_scores.std()
            },
            'top_features': self.feature_importance.head(10).to_dict('records')
        }

        return report

    def plot_diagnostics(self):
        """
        Genera gráficos de diagnóstico del modelo
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Distribución de residuos
        sns.histplot(self.residuals, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribución de Residuos')

        # Top 10 características más importantes
        top_features = self.feature_importance.head(10)
        sns.barplot(data=top_features, x='importance',
                    y='feature', ax=axes[0, 1])
        axes[0, 1].set_title('Top 10 Características más Importantes')

        # Validación cruzada scores
        sns.boxplot(data=self.cv_scores, ax=axes[1, 0])
        axes[1, 0].set_title('Distribución de R² en Validación Cruzada')

        # Coeficientes por tipo de característica
        feature_types = self.feature_importance['feature'].apply(
            lambda x: 'Categórica' if '_' in x else 'Numérica'
        )
        sns.boxplot(
            x=feature_types,
            y=self.feature_importance['importance'],
            ax=axes[1, 1]
        )
        axes[1, 1].set_title(
            'Distribución de Importancia por Tipo de Característica')

        plt.tight_layout()
        return fig
