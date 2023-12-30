import uvicorn
from fastapi import FastAPI,File,UploadFile,status,Response,Request
import pandas as pd
import io 
from fastapi.middleware.cors import CORSMiddleware
import random
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import json
app = FastAPI()
origins = [
    'http://127.0.0.1:5500',
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# app.mount("/Frontend", StaticFiles(directory="Frontend"), name="static")
app.mount("/static", StaticFiles(directory="../Frontend"), name="static")
templates = Jinja2Templates(directory="../Frontend")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("home.html",{"request":request,"key":"value"})
        
@app.get('/check_status',status_code=status.HTTP_201_CREATED)
def root(response:Response):
    try:
        return {"status":200,"message":"Article prediction is in Progress!!"}
    except Exception as ex:
        print(ex)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status":400,"message":str(ex)}
    
# Calling the API Gateway endpoint 
def gateway_api(json_data:dict):
    try:
        # API Gateway endpoint
        url = "https://nj7tqz9049.execute-api.us-west-1.amazonaws.com/Dev/api_gate_res"

        payload = json.dumps({
        "data":json_data
        })
        headers = {
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response
    except Exception as ex:
        raise Exception(str(ex))
    
@app.post('/get_prediction_for_csv',status_code=status.HTTP_201_CREATED)
def import_csv(file: UploadFile = File(...)):
    try:
        dataframe = pd.read_csv(io.StringIO(file.file.read().decode('utf-8')), delimiter=',')
        top_ten_df = [i.split('/')[-2] for i in dataframe['url'].head(10).tolist()]
        response = gateway_api(data)
        return {"status":status.HTTP_201_CREATED,"model":\
                [{"xg_boost":{"MAE":response['xg_boost']['MAE'],"RMS":response['xg_boost']['RMS']}},
                {"gradient_boosting":{"MAE":response['gradient_boosting']['RMS'],"RMS":response['gradient_boosting']['RMS']}}
                ],"data":response}
    except Exception as ex:
        print(ex)
        return {"status":status.HTTP_400_BAD_REQUEST,"message":str(ex)}
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
