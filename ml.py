import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_model(df, logger):
    """
    Advanced ML model with:
    - Preprocessing pipeline
    - Model comparison
    - Cross validation
    - Feature importance
    - SHAP values
    """

    try:
        # ==============================
        # FEATURES & TARGET
        # ==============================

        X = df[[
            "gender",
            "parental level of education",
            "lunch",
            "test preparation course",
            "reading score",
            "writing score"
        ]]

        y = df["math score"]

        numeric_features = ["reading score", "writing score"]
        categorical_features = [
            "gender",
            "parental level of education",
            "lunch",
            "test preparation course"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                random_state=42
            )
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}
        best_model = None
        best_model_name = None
        best_r2 = -np.inf
        best_y_pred = None

        # ==============================
        # MODEL COMPARISON + CV
        # ==============================

        for name, model in models.items():

            pipeline = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            cv_scores = cross_val_score(
                pipeline, X, y, cv=5, scoring="r2"
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "cv_mean": cv_scores.mean()
            }

            if r2 > best_r2:
                best_r2 = r2
                best_model = pipeline
                best_model_name = name
                best_y_pred = y_pred

        logger.info("Advanced ML training + CV completed.")

        # ==============================
        # FEATURE IMPORTANCE (RF ONLY)
        # ==============================

        feature_importance = None
        shap_values = None

        if best_model_name == "Random Forest":

            rf_model = best_model.named_steps["model"]
            transformed_features = best_model.named_steps[
                "preprocessing"
            ].get_feature_names_out()

            feature_importance = pd.DataFrame({
                "feature": transformed_features,
                "importance": rf_model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            # ==============================
            # SHAP
            # ==============================

            X_processed = best_model.named_steps["preprocessing"].transform(X_test)

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_processed)

        return (
            best_model,
            best_model_name,
            results,
            y_test,
            best_y_pred,
            feature_importance,
            shap_values
        )

    except Exception as e:
        logger.error(f"Error during advanced model training: {e}")
        raise
