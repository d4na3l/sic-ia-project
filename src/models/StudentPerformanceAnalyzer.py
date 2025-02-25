import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class StudentPerformanceAnalyzer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.best_model = None
        self.feature_importance = None
        self.metrics = None
        self.cv_scores = None
        self.feature_types = None
        self.all_models_results = {}  # Para almacenar resultados de todos los modelos

        # Se agregan los atributos para almacenar X_test y y_test
        self.X_test = None
        self.y_test = None

    def _categorize_features(self, feature_names):
        """
        Categoriza las características según su tipo
        """
        feature_types = {}
        for feature in feature_names:
            if feature == 'Exam_Score':
                continue
            elif feature in self.preprocessor.numeric_features:
                feature_types[feature] = 'Numérica'
            elif any(feature.startswith(f"{nom}_") for nom in self.preprocessor.nominal_features):
                feature_types[feature] = 'Nominal'
            elif feature in self.preprocessor.ordinal_mappings:
                feature_types[feature] = 'Ordinal'
        return feature_types

    def prepare_data(self, df):
        """
        Prepara los datos procesados para el entrenamiento
        """
        processed_df = self.preprocessor.fit_transform(df)
        self.feature_types = self._categorize_features(processed_df.columns)

        X = processed_df.drop('Exam_Score', axis=1)
        y = processed_df['Exam_Score']

        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    def train_model(self, X_train, y_train):
        """
        Entrena múltiples modelos incluyendo regresión lineal
        """
        models = {
            'linear': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': np.logspace(-4, 4, 9),  # Rango más amplio
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                    'fit_intercept': [True, False]
                }
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': np.logspace(-4, 4, 9),
                    'selection': ['cyclic', 'random'],
                    'fit_intercept': [True, False]
                }
            },
            'elasticnet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': np.logspace(-4, 4, 9),
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'fit_intercept': [True, False]
                }
            }
        }

        best_score = -float('inf')
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Entrenar y evaluar cada modelo
        for name, model_info in models.items():
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=kfold,
                scoring=['r2', 'neg_mean_squared_error',
                         'neg_mean_absolute_error'],
                refit='r2',
                n_jobs=-1,
                verbose=1  # Añadido para mostrar progreso
            )

            print(f"\nEntrenando modelo: {name}")
            grid_search.fit(X_train, y_train)

            # Guardar resultados de cada modelo
            self.all_models_results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'model': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
                self.best_params = grid_search.best_params_

        # Calcular scores de validación cruzada para el mejor modelo
        self.cv_scores = {
            'r2': cross_val_score(self.best_model, X_train, y_train, cv=kfold, scoring='r2'),
            'rmse': np.sqrt(-cross_val_score(self.best_model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')),
            'mae': -cross_val_score(self.best_model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
        }

    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo con métricas expandidas
        """
        y_pred = self.best_model.predict(X_test)

        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'adjusted_r2': 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        }

        # Calcular importancia de características con tipos
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(self.best_model.coef_),
            'type': [self.feature_types.get(feat, 'Unknown') for feat in X_test.columns]
        })

        self.feature_importance = importance_df.sort_values(
            'importance', ascending=False)
        self.residuals = y_test - y_pred

    def analyze_performance(self, df):
        """
        Realiza el análisis completo y genera un reporte detallado
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Almacenar X_test e y_test en la instancia
        self.X_test = X_test
        self.y_test = y_test

        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

        report = self._generate_report()
        return report

    def _generate_report(self):
        """
        Genera un reporte más completo incluyendo comparación entre modelos
        """
        base_report = {}

        # Comparación entre modelos
        models_comparison = {
            name: {
                'best_score': results['best_score'],
                'best_params': results['best_params'],
                'cv_mean_r2': np.mean(results['cv_results']['split0_test_r2']),
                'cv_std_r2': np.std(results['cv_results']['split0_test_r2'])
            }
            for name, results in self.all_models_results.items()
        }
        base_report['models_comparison'] = models_comparison

        # Información del modelo seleccionado
        base_report['model_info'] = {
            'type': self.best_model_name,
            'best_params': self.best_params
        }

        # Métricas de rendimiento
        base_report['model_performance'] = {
            'test_metrics': self.metrics,
            'cross_validation': {
                'r2': {'mean': np.mean(self.cv_scores['r2']),
                       'std': np.std(self.cv_scores['r2'])},
                'rmse': {'mean': np.mean(self.cv_scores['rmse']),
                         'std': np.std(self.cv_scores['rmse'])},
                'mae': {'mean': np.mean(self.cv_scores['mae']),
                        'std': np.std(self.cv_scores['mae'])}
            }
        }

        # Importancia de las características
        overall_importance = self.feature_importance.to_dict(orient='records')
        by_type = {
            tipo: group.to_dict(orient='records')
            for tipo, group in self.feature_importance.groupby('type')
        }
        base_report['feature_importance'] = {
            'overall': overall_importance,
            'by_type': by_type
        }

        return base_report

    def plot_model_comparison(self):
        """
        Genera gráficos comparativos entre modelos
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Preparar datos para la comparación
        model_names = list(self.all_models_results.keys())
        r2_scores = [results['best_score']
                     for results in self.all_models_results.values()]
        cv_stds = [np.std(results['cv_results']['split0_test_r2'])
                   for results in self.all_models_results.values()]

        # Gráfico de barras de R²
        sns.barplot(x=model_names, y=r2_scores, ax=ax1)
        ax1.set_title('Comparación de R² entre Modelos')
        ax1.set_ylabel('R²')
        ax1.tick_params(axis='x', rotation=45)

        # Gráfico de desviación estándar en validación cruzada
        sns.barplot(x=model_names, y=cv_stds, ax=ax2)
        ax2.set_title('Estabilidad de Modelos (Desv. Est. en CV)')
        ax2.set_ylabel('Desviación Estándar')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def analyze_model_differences(self):
        """
        Analiza las diferencias entre los modelos en términos de predicciones
        """
        predictions = {}
        differences = {}

        # Obtener predicciones de cada modelo
        for name, results in self.all_models_results.items():
            model = results['model']
            predictions[name] = model.predict(self.X_test)

        # Calcular diferencias entre modelos
        for model1 in predictions:
            for model2 in predictions:
                if model1 < model2:
                    key = f"{model1}_vs_{model2}"
                    diff = np.abs(predictions[model1] - predictions[model2])
                    differences[key] = {
                        'mean_diff': np.mean(diff),
                        'max_diff': np.max(diff),
                        'std_diff': np.std(diff)
                    }

        return differences

    def plot_diagnostics(self):
        """
        Genera gráficos de diagnóstico mejorados
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig)

        # 1. Distribución de residuos
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(self.residuals, kde=True, ax=ax1)
        ax1.set_title('Distribución de Residuos')

        # 2. Q-Q plot de residuos
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot de Residuos')

        # 3. Top 10 características más importantes
        ax3 = fig.add_subplot(gs[1, :])
        sns.barplot(
            data=self.feature_importance.head(10),
            x='importance',
            y='feature',
            hue='type',
            ax=ax3
        )
        ax3.set_title('Top 10 Características más Importantes por Tipo')

        # 4. Distribución de importancia por tipo de característica
        ax4 = fig.add_subplot(gs[2, 0])
        sns.boxplot(
            data=self.feature_importance,
            x='type',
            y='importance',
            ax=ax4
        )
        ax4.set_title('Distribución de Importancia por Tipo de Característica')

        # 5. Distribución de métricas CV
        ax5 = fig.add_subplot(gs[2, 1])
        cv_data = pd.DataFrame({
            'R²': self.cv_scores['r2'],
            'RMSE': self.cv_scores['rmse'],
            'MAE': self.cv_scores['mae']
        })
        sns.boxplot(data=cv_data, ax=ax5)
        ax5.set_title('Distribución de Métricas en Validación Cruzada')

        plt.tight_layout()
        return fig
