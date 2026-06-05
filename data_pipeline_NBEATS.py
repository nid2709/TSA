from src.N_BEATS.data_processing import data_preprocessing
from src.N_BEATS.N_BEATS import model_pipeline
from src.N_BEATS.N_BEATSH5 import model_pipeline_hor5
from src.N_BEATS.N_BEATSH10 import model_pipeline_hor10


def run_pipeline():
    print("\n========== Libraries Importing ==========")

    feature_df = data_preprocessing()
    print("\n========== Completed Data Preprocessing ==========")
    
    model_pipeline(feature_df)
    print("\n========== N-BEATS Model Completed! ==========")

    #model_pipeline_hor5(feature_df)
    print("\n========== N-BEATS Model Horizon Step 5 Completed! ==========")

    #model_pipeline_hor10(feature_df)
    print("\n========== N-BEATS Model Horizon Step 10 Completed! ==========")


if __name__ == "__main__":
    run_pipeline()