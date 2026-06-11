from sklearn.base import BaseEstimator


def build_model(model_type: str = "xgb") -> BaseEstimator:
    """
    Instantiate a configured regression model.

    XGBoost is the primary model: regularised (reg_alpha/lambda), depth-limited,
    and uses histogram-based splits for speed on tabular data.
    Random Forest serves as the second ensemble member.
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
    raise ValueError(f"Unsupported model_type '{model_type}'. Use 'xgb' or 'rf'.")
