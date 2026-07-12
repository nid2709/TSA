from src.N_BEATS.data_processing import data_preprocessing
from src.N_BEATS.nbeats import model_pipeline
from src.N_BEATS.nbeats_long import model_pipeline_nbeats_long
from src.N_BEATS.synbeats import model_pipeline_synbeats
from src.N_BEATS.nbeats_wavelet import model_pipeline_nbeats_wavelet
from src.N_BEATS.N_BEATSH5 import model_pipeline_hor5
from src.N_BEATS.N_BEATSH10 import model_pipeline_hor10
from src.N_BEATS.N_BEATS import model_pipeline_XAI_UQ

from src.N_BEATS.data_processing_AQI import data_preprocessing_AQI
from src.N_BEATS.N_BEATS_AQI import model_pipeline_AQI
from src.N_BEATS.N_BEATS_AQI import model_pipeline_AQI_hor5
from src.N_BEATS.N_BEATS_AQI import model_pipeline_AQI_hor10

from src.N_BEATS.data_processing_Temp import data_preprocessing_Temp
from src.N_BEATS.N_BEATS_Temp import model_pipeline_Temp
from src.N_BEATS.N_BEATS_Temp_long import model_pipeline_Temp_long


def run_pipeline():
    print("\n========== Libraries Importing ==========")

    print("\n========== CO2 Prediction ==========")

    feature_df = data_preprocessing()
    print("\n========== Completed Data Preprocessing ==========")

    model_pipeline(feature_df)
    print("\n========== N-BEATS Model Completed For Shortterm! ==========")

    # model_pipeline_nbeats_long(feature_df)
    # print("\n========== N-BEATS Model For Longterm Completed! ==========")

    # model_pipeline_synbeats(feature_df)
    # print("\n========== syN-BEATS Model For Longterm Completed! ==========")

    # model_pipeline_nbeats_wavelet(feature_df)
    # print("\n========== N-BEATS Wavelet Model For Longterm Completed! ==========")

    #model_pipeline_hor5(feature_df)
    #print("\n========== N-BEATS Model Horizon Step 5 Completed! ==========")

    #model_pipeline_hor10(feature_df)
    #print("\n========== N-BEATS Model Horizon Step 10 Completed! ==========")

    # model_pipeline_XAI_UQ(feature_df)
    # print("\n========== N-BEATS Model Completed with XAI and UQ! ==========")




    
    # print("\n========== AQI Prediction ==========")

    # feature_df = data_preprocessing_AQI()
    # print("\n========== Completed Data Preprocessing ==========")

    # model_pipeline_AQI(feature_df)
    # print("\n========== N-BEATS Model Completed! ==========")

    # model_pipeline_AQI_hor5(feature_df)
    # print("\n========== N-BEATS Model Horizon Step 5 Completed! ==========")

    # model_pipeline_AQI_hor10(feature_df)
    # print("\n========== N-BEATS Model Horizon Step 10 Completed! ==========")





    # print("\n========== Temprature Prediction ==========")
    
    # feature_df = data_preprocessing_Temp()
    # print("\n========== Completed Data Preprocessing ==========")

    # model_pipeline_Temp(feature_df)
    # print("\n========== N-BEATS Model Completed! ==========")

    # model_pipeline_Temp_long(feature_df)
    # print("\n========== N-BEATS Model for long-term Completed! ==========")

if __name__ == "__main__":
    run_pipeline()