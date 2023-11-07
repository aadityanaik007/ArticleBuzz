import uvicorn
from fastapi import FastAPI,File,UploadFile,status,Response
import pandas as pd
import io 
from fastapi.middleware.cors import CORSMiddleware

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

@app.get('/',status_code=status.HTTP_201_CREATED)
def homepage(response:Response):
    try:
        return {"message":"works!!!!"}
    except Exception as ex:
        print(ex)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status":400,"message":str(ex)}
        
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
        print(dataframe.head())
        return {"status":status.HTTP_201_CREATED,"message":"CSV received!"}
    except Exception as ex:
        print(ex)
        return {"status":status.HTTP_400_BAD_REQUEST,"message":str(ex)}
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)