from sklearn.base import BaseEstimator


def build_model(model_type: str = "xgb") -> BaseEstimator:
    """
    Instantiate a configured regression model.

    The four model types form the heterogeneous ensemble described in the paper
    (Section 2.5): XGBoost and LightGBM capture non-linear relationships,
    Random Forest contributes stability through averaging, and ElasticNet serves
    as a regularised linear anchor that reduces variance in low-signal regimes.
    """
    if model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    if model_type == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_samples=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
    if model_type == "elasticnet":
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)),
        ])
    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Use 'xgb', 'lgbm', 'rf', or 'elasticnet'."
    )
