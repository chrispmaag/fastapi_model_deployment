# Source: https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8
# https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f
# https://towardsdatascience.com/deployment-could-be-easy-a-data-scientists-guide-to-deploy-an-image-detection-fastapi-api-using-329cdd80400
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel

class Input(BaseModel):
    age: int
    sex: str

app = FastAPI()


# @app.post("/predict/image")
# async def predict_api(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format."
#     image = read_imagefile(await file.read())
#     prediction = predict(image)

#     return prediction

# Get request version
# @app.get("/predict")
# def predict_model(age: int, sex:str):
#     if age < 10 or sex=='F':
#         return {'survived': 1}
#     return {'survived': 0}

@app.put("/predict")
def predict_model(d: Input):
    if d.age < 10 or d.sex=='F':
        return {'survived': 1}
    return {'survived': 0}
