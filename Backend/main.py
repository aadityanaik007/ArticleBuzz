import uvicorn
from fastapi import FastAPI,File,UploadFile,status,Response
import pandas as pd
import io 

app = FastAPI()

@app.get('/',status_code=status.HTTP_201_CREATED)
def homepage():
    return {"message":"works!!!!"}

@app.get('/check_status',status_code=status.HTTP_201_CREATED)
def root(response:Response):
    try:
        return "Article prediction is in Progress!!"
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
