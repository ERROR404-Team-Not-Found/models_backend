from hashlib import sha256
import os
import io
import shutil
import zipfile
import uvicorn
import subprocess
import pandas as pd
from PIL import Image
from pathlib import Path
from base64 import b64encode
from minio_client import Client
from models import Layers, Train
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from model_utils import layers, activation_functions, get_layer, read_log_file

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

minio_client = None

@app.on_event("startup")
def start():
    global minio_client
    minio_client = Client()
    minio_client.get_last_model_version("something", "Detector")


@app.get("/")
async def main():
    return JSONResponse("Server works!", status_code=200)


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


@app.get("/model/architecture")
async def model_architecture(user_id: str, model_name: str):
    result = minio_client.get_object(user_id, model_name, f"{model_name}.py")

    if result is None:
        return JSONResponse("Could not fetch model architecture!", status_code=500)

    return JSONResponse(result.decode("utf-8"), status_code=200)


@app.post("/model/dataset")
async def model_dataset(user_id: str, model_name: str, file: UploadFile, label_column: str = None):
    result = minio_client.create_user_bucket(user_id)

    if not result:
        return JSONResponse("Could not create user bucket!", status_code=500)

    if file.content_type == "application/zip":
        comment = f'# {{"dataset_path": "{user_id}/{model_name}/dataset.csv", "dataset_type": "images", "label_column": "label_values"}}\n'
        with open("./datasets.py", 'r') as f:
            content = f.read()

        buffer = io.StringIO()

        buffer.write(comment + content)

        string_content = buffer.getvalue()

        bytes_content = string_content.encode('utf-8')

        bytes_io = io.BytesIO()
        bytes_io.write(bytes_content)

        bytes_io.seek(0)

        result_dataset_loader = minio_client.upload_file(user_id, model_name, "dataset.py", bytes_io) 

        with zipfile.ZipFile(io.BytesIO(await file.read()), 'r') as zip_ref:
            temp_dir = "temp_images"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            zip_ref.extractall(temp_dir)
    
        for path in Path(temp_dir).glob("*.csv"):
            csv_path = str(path)
        df = pd.read_csv(csv_path)

        required_columns = {'image', 'label_value', 'label_name'}
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        for idx, row in df.iterrows():
            image_path = os.path.join(temp_dir, row['image'])
            try:
                with Image.open(image_path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format=img.format)
                    img_base64 = b64encode(buffered.getvalue()).decode("utf-8")
                df.at[idx, 'image'] = f"data:image/png;base64,{(img_base64)}"
            except IOError:
                df.at[idx, 'image'] = None

        shutil.rmtree(temp_dir)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        result = minio_client.upload_file(user_id, model_name, "dataset.csv", buffer) 
        
        if result is None or result_dataset_loader is None:
            return JSONResponse("Couldn't upload the dataset!", status_code=500)

        return JSONResponse("Uploaded dataset successfully!", status_code=201)

    elif file.content_type == 'text/csv':
        buffer = await file.read()

        result = minio_client.upload_file(user_id, model_name, "dataset.csv", io.BytesIO(await file.read()))

        comment = f'# {{"dataset_path": "{user_id}/{model_name}/dataset.csv", "dataset_type": "csv", "label_column": "{label_column}"}}\n'
        with open("./datasets.py", 'r') as f:
            content = f.read()

        buffer = io.StringIO()

        buffer.write(comment + content)

        string_content = buffer.getvalue()

        bytes_content = string_content.encode('utf-8')

        bytes_io = io.BytesIO()
        bytes_io.write(bytes_content)

        bytes_io.seek(0)

        result_dataset_loader = minio_client.upload_file(user_id, model_name, "dataset.py", bytes_io) 

        if result is None or result_dataset_loader is None:
            return JSONResponse("Couldn't upload the dataset!", status_code=500)
        
        return JSONResponse("Uploaded dataset successfully!", status_code=201)

    return JSONResponse("Dataset type can only be CSV and zip!", status_code=400)


@app.get("/model/dataset")
async def model_dataset(user_id: str, model_name: str):
    result = minio_client.get_object(user_id, model_name, "dataset.csv")

    df = pd.read_csv(io.StringIO(result.decode('utf-8')))

    if "image" in df.columns:
        file_type = "images"
    else:
        file_type = "csv"

    if result is None:
        return JSONResponse("Couldn't fetch the dataset!", status_code=500)
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_data = buffer.getvalue()

    data = {
        "dataset_type": file_type,
        "csv_data": csv_data
    }

    return JSONResponse(data, status_code=200)


@app.get("/models")
async def models(user_id: str):
    result = minio_client.get_models(user_id)

    if result is None:
        JSONResponse("Couldn't fetch the models!", status_code=500)

    return JSONResponse(result, status_code=200)


@app.post("/model/train")
async def model_train(train: Train):
    latest_version = minio_client.get_last_model_version(train.user_id, train.model_name)
    cmd = [
        "nohup", "python3", os.getenv("TRAIN_PATH"),
        "-u", str(train.user_id),
        "-m", str(train.model_name),
        "-l", str(train.learning_rate),
        "-o", str(train.optimizer),
        "-e", str(train.epochs),
        "-b", str(train.batch_size),
        "-v", str(latest_version + 1),
        "&"
    ]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    


@app.get("/train/status")
async def train_status():
    log_entries = read_log_file("./model.log")
    for entry in log_entries:
        print(entry)


@app.get("/model/versions")
async def model_versions(user_id: str, model_name: str):
    result = minio_client.get_model_versions(user_id, model_name)

    if result is None:
        JSONResponse("Couldn't fetch the model versions!", status_code=500)

    return result


@app.get("/model/weight")
async def model_weight(user_id: str, model_name: str, version: str, hash: str):
    result = minio_client.get_model_version(user_id, model_name, version)

    if result is None:
        JSONResponse("Couldn't fetch the model versions!", status_code=500)

    sha256_hash = sha256()

    def iterfile():
        for chunk in result.stream(32*1024):
            sha256_hash.update(chunk)
            yield chunk

    computed_hash = sha256_hash.hexdigest()
    if computed_hash != hash:
        raise JSONResponse(status_code=500, content="Hash does not match, model weights are altered.")

    headers = {
        "Content-Disposition": f"attachment; filename={version}.pth"
    }

    return StreamingResponse(iterfile(), headers=headers)


@app.get("/model/version/last")
async def model_versions(user_id: str, model_name: str):
    result = minio_client.get_last_model_version(user_id, model_name)

    if result is None:
        JSONResponse("Couldn't fetch the model versions!", status_code=500)

    return result


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
