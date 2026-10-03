"""
PURPOSE: Ye MAIN script hai jo poore pipeline ko END-TO-END chalata hai —
         data load se lekar final model save hone tak. Ye kisi bhi individual
         module ki NAYI functionality nahi hai, sirf saare pichle phases ke
         functions ko SAHI ORDER mein call karta hai.
"""

import logging

from src.config import (
    CV_FOLDS,
    MODEL_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_FIGURES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.loader import DataLoader
from src.data.splitter import get_split_summary, split_features_target, split_train_test
from src.data.validation import DataValidator
from src.evaluation.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.features.preprocessing import build_preprocessor, infer_feature_types
from src.models.selection import (
    compare_models,
    save_model,
    select_production_model,
)
from src.models.train import get_predictions
from src.models.tune import tune_logistic_regression, tune_random_forest
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["amount", "age", "channel", TARGET_COLUMN]

N_TUNING_ITERATIONS = 10

MIN_RECALL = 0.7
MIN_PRECISION = 0.5

MODEL_VERSION = "v1"


def main() -> None:
    setup_logging()

    df = DataLoader(RAW_DATA_PATH / "transactions.csv").load()

    validator = DataValidator(
        required_columns=REQUIRED_COLUMNS, target_column=TARGET_COLUMN
    )
    validator.validate(df)

    X, y = split_features_target(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    split_summary = get_split_summary(y_train, y_test)
    logger.info("Split summary: %s", split_summary)

    numerical_features, categorical_features = infer_feature_types(df, TARGET_COLUMN)
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    lr_search = tune_logistic_regression(
        X_train,
        y_train,
        preprocessor,
        random_state=RANDOM_STATE,
        cv_folds=CV_FOLDS,
        n_iter=N_TUNING_ITERATIONS,
    )
    rf_search = tune_random_forest(
        X_train,
        y_train,
        preprocessor,
        random_state=RANDOM_STATE,
        cv_folds=CV_FOLDS,
        n_iter=N_TUNING_ITERATIONS,
    )

    searches = [
        ("logistic_regression", lr_search),
        ("random_forest", rf_search),
    ]
    fitted_pipelines = {}
    model_results = {}
    predictions = {}

    for model_name, search in searches:
        best_pipeline = search.best_estimator_
        fitted_pipelines[model_name] = best_pipeline

        y_pred, y_proba = get_predictions(best_pipeline, X_test)
        predictions[model_name] = (y_pred, y_proba)
        model_results[model_name] = calculate_metrics(y_test, y_pred, y_proba)

    comparison_df = compare_models(model_results)
    logger.info("Model comparison:\n%s", comparison_df)
    print(comparison_df.to_string())

    best_model_name = select_production_model(
        model_results, min_recall=MIN_RECALL, min_precision=MIN_PRECISION
    )
    best_pipeline = fitted_pipelines[best_model_name]

    model_path = save_model(
        best_pipeline, best_model_name, version=MODEL_VERSION, model_dir=MODEL_DIR
    )

    for model_name, search in searches:
        y_pred, y_proba = predictions[model_name]
        plot_confusion_matrix(
            y_test,
            y_pred,
            REPORTS_FIGURES_DIR / f"confusion_matrix_{model_name}.png",
            model_name,
        )
        plot_roc_curve(
            y_test,
            y_proba,
            REPORTS_FIGURES_DIR / f"roc_curve_{model_name}.png",
            model_name,
        )

    print(
        f"\n=== FINAL SUMMARY ===\n"
        f"Selected model: {best_model_name}\n"
        f"Metrics: {model_results[best_model_name]}\n"
        f"Model saved at: {model_path}"
    )


if __name__ == "__main__":
    main()