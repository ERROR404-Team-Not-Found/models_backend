import uvicorn
from fastapi import fastapi

app = FastAPI()


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
