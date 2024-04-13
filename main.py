import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.post("/model/create")
async def model_create():
    pass


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
