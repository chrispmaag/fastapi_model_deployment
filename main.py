# Source: https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8
# https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f
# https://towardsdatascience.com/deployment-could-be-easy-a-data-scientists-guide-to-deploy-an-image-detection-fastapi-api-using-329cdd80400
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format."
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
