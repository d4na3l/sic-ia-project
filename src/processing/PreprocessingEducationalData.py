import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PreprocessingEducationalData:
    def __init__(self):
        self.numeric_features = [
            'Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Tutoring_Sessions',
            'Physical_Activity', 'Exam_Score'
        ]

        self.categorical_features = [
            'Parental_Involvement', 'Access_to_Resources',
            'Extracurricular_Activities', 'Motivation_Level',
            'Internet_Access', 'Family_Income', 'Teacher_Quality',
            'School_Type', 'Peer_Influence', 'Learning_Disabilities',
            'Parental_Education_Level', 'Distance_from_Home', 'Gender'
        ]

        # Crear el preprocesador usando ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False),
                 self.categorical_features)
            ])

        self.feature_names = None

    def fit_transform(self, df):
        """
        Ajusta y transforma los datos aplicando estandarización y one-hot encoding.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame procesado
        """
        # Aplicar transformaciones
        data_transformed = self.preprocessor.fit_transform(df)

        # Obtener nombres de características
        numeric_features = self.numeric_features

        # Obtener nombres de características categóricas codificadas
        cat_features = []
        for i, feature in enumerate(self.categorical_features):
            categories = self.preprocessor.named_transformers_[
                'cat'].categories_[i][1:]
            cat_features.extend([f"{feature}_{cat}" for cat in categories])

        # Guardar todos los nombres de características
        self.feature_names = numeric_features + cat_features

        # Crear DataFrame con nombres de características
        return pd.DataFrame(data_transformed, columns=self.feature_names)

    def transform(self, df):
        """
        Transforma nuevos datos usando los parámetros ya ajustados.

        Args:
            df (pd.DataFrame): DataFrame a transformar

        Returns:
            pd.DataFrame: DataFrame procesado
        """
        data_transformed = self.preprocessor.transform(df)
        return pd.DataFrame(data_transformed, columns=self.feature_names)

    def get_feature_names(self):
        """
        Retorna los nombres de todas las características procesadas.

        Returns:
            list: Lista de nombres de características
        """
        return self.feature_names
