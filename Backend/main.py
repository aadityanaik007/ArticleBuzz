import uvicorn
from fastapi import FastAPI,File,UploadFile,status,Response,Request
import pandas as pd
import io 
from fastapi.middleware.cors import CORSMiddleware
import random
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

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

app.mount("/static", StaticFiles(directory="../Frontend"), name="static")
templates = Jinja2Templates(directory="../Frontend")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("new_homepage.html",{"request":request,"key":"value"})
        
@app.get('/check_status',status_code=status.HTTP_201_CREATED)
def root(response:Response):
    try:
        return {"status":200,"message":"Article prediction is in Progress!!"}
    except Exception as ex:
        print(ex)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status":400,"message":str(ex)}
    
@app.post('/get_prediction_for_csv',status_code=status.HTTP_201_CREATED)
def import_csv(file: UploadFile = File(...)):
    try:
        dataframe = pd.read_csv(io.StringIO(file.file.read().decode('utf-8')), delimiter=',')
        if dataframe.shape[0]>9:
            urls = dataframe['url'].head(10).tolist()
            df = [i.split('/')[-2] for i in urls]
        elif dataframe.shape[0] ==0:
            raise Exception("Please provide atleast one article for prediction")
        else:
            df = dataframe['url'].to_list()
        time.sleep(2)
        return {"status":status.HTTP_201_CREATED,"model":\
                [{"KNN":{"MAE":2982.32,"RMS":8957.45}},
                {"RF":{"MAE":3234.32,"RMS":1243.45}},
                {"GB":{"MAE":3452.32,"RMS":98.45}},
                {"XGB":{"MAE":2452.32,"RMS":38.45}}
                ],"data":[{"number_of_shares":random.randint(1, 100),"article_name":i} for i in df]}
    except Exception as ex:
        print(ex)
        return {"status":status.HTTP_400_BAD_REQUEST,"message":str(ex)}
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
