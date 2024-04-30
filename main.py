from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import heartpy as hp
import pandas as pd
from io import BytesIO
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)



@app.post("/")
async def evaluate_hrv(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        if 'ecg' not in df.columns:
            raise ValueError("CSV file must contain 'ecg' column.")

        # Extract ECG data column from the CSV file
        hrdata = df['ecg'].values.tolist()

        # Process HRV
        fs = 200.137457
        working_data_hrv, measures_hrv = hp.process(hrdata, fs, report_time=True, calc_freq=True, high_precision=True, high_precision_fs=1000.0)
        
        return measures_hrv
    except Exception as e:
        return {"error": str(e)}


