import uvicorn
from models import Layers
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from minio_client import Client
from model_utils import layers, activation_functions, get_layer

app = FastAPI()

minio_client = None

@app.on_event("startup")
def start():
    global minio_client
    minio_client = Client()


@app.get("/model/layers")
async def model_layers():
    return layers


@app.get("/model/activations")
async def model_activations():
    return activation_functions



@app.post("/model/create")
async def model_create(layers: Layers):
    model_code = "import torch.nn as nn\n\n"
    model_code += f"class {layers.name}(nn.Module):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n"

    for i, layer in enumerate(layers.layers):
        model_code += f"        self.layer{i} = nn.{get_layer(layer.name, layer.params)}\n"
    
    model_code += "    \ndef forward(self, x):\n"
    for i, layer in enumerate(layers.layers):
        model_code += f"        x = self.layer{i}(x)\n"
    
    model_code += "        return x\n"

    create_bucket_result = minio_client.create_user_bucket(layers.user_id)
    if not create_bucket_result:
        return JSONResponse("Could not create user bucket!", status_code=500)
    
    upload_model = minio_client.upload_file(layers.user_id, layers.name, f'{layers.name.lower()}.py', model_code)

    if upload_model is None:
        return JSONResponse("Could not upload model!", status_code=500)
    
    return JSONResponse("Model created successfully!", status_code=201)

    

    


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
