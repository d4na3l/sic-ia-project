import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PreprocessingEducationalData:
    def __init__(self):
        # Variables numéricas
        self.numeric_features = [
            'Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Tutoring_Sessions',
            'Physical_Activity', 'Exam_Score'
        ]

        # Variables nominales (para one-hot encoding)
        self.nominal_features = [
            'Extracurricular_Activities',
            'Internet_Access',
            'School_Type',
            'Learning_Disabilities',
            'Gender'
        ]

        # Variables ordinales (para label encoding con mapeo específico)
        self.ordinal_mappings = {
            'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
            'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
            'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
            'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
            # Nueva codificación
            'Peer_Influence': {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        }

        # Inicializar transformadores
        self.numeric_transformer = StandardScaler()
        self.nominal_transformer = OneHotEncoder(
            drop='first', sparse_output=False)

        # Almacenar transformadores para su uso posterior
        self.column_transformer = None
        self.feature_names = None

    def _create_ordinal_transformers(self):
        """
        Crea transformadores para variables ordinales usando los mapeos definidos
        """
        return {
            feature: lambda x, mapping=mapping: x.map(mapping)
            for feature, mapping in self.ordinal_mappings.items()
        }

    def fit_transform(self, df):
        """
        Ajusta y transforma los datos aplicando las transformaciones apropiadas
        para cada tipo de variable.
        """
        # Eliminar las columnas con valores nulos
        df_processed = df.drop(['Teacher_Quality',
                                'Parental_Education_Level',
                                'Distance_from_Home'], axis=1).copy()

        # Transformar variables ordinales
        ordinal_transformers = self._create_ordinal_transformers()
        for feature, transformer in ordinal_transformers.items():
            df_processed[feature] = transformer(df_processed[feature])

        # Crear ColumnTransformer para variables numéricas y nominales
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('nom', self.nominal_transformer, self.nominal_features)
            ],
            remainder='passthrough'
        )

        # Aplicar transformaciones
        transformed_array = self.column_transformer.fit_transform(df_processed)

        # Generar nombres de características
        nominal_feature_names = []
        if self.nominal_features:
            nominal_categories = self.column_transformer.named_transformers_[
                'nom'].categories_
            for feature, categories in zip(self.nominal_features, nominal_categories):
                nominal_feature_names.extend(
                    [f"{feature}_{cat}" for cat in categories[1:]])

        self.feature_names = (
            self.numeric_features +
            nominal_feature_names +
            list(self.ordinal_mappings.keys())
        )

        # Crear DataFrame con los nombres de características
        return pd.DataFrame(
            transformed_array,
            columns=self.feature_names
        )

    def transform(self, df):
        """
        Transforma nuevos datos usando los parámetros ya ajustados.
        """
        if self.column_transformer is None:
            raise ValueError(
                "El preprocesador debe ser ajustado primero con fit_transform")

        # Eliminar las columnas con valores nulos
        df_processed = df.drop(['Teacher_Quality',
                                'Parental_Education_Level',
                                'Distance_from_Home'], axis=1).copy()

        # Transformar variables ordinales
        ordinal_transformers = self._create_ordinal_transformers()
        for feature, transformer in ordinal_transformers.items():
            df_processed[feature] = transformer(df_processed[feature])

        # Aplicar transformaciones
        transformed_array = self.column_transformer.transform(df_processed)

        return pd.DataFrame(
            transformed_array,
            columns=self.feature_names
        )
