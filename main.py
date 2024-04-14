import os
import io
import shutil
import tempfile
import zipfile
import subprocess
from fastapi.responses import FileResponse
import pandas as pd
from PIL import Image
from minio import Minio
from pathlib import Path
from hashlib import sha256
from base64 import b64encode
from minio_client import Client
from models import Layers, Train
from fastapi import BackgroundTasks, FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from model_utils import layers, activation_functions, get_layer, read_log_file
from dotenv import load_dotenv

load_dotenv()

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
    model_code = "import torch.nn as nn\n"
    model_code += "import torch.nn.functional as F\n"
    model_code += f"class {layers.name}(nn.Module):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n"

    for i, layer in enumerate(layers.layers):
        model_code += f"        self.layer{i} = nn.{get_layer(layer.name, layer.params)}\n"

    model_code += f"        self.fc1 = nn.Linear(64 * 54 * 54, 128)\n"
    model_code += f"        self.fc2 = nn.Linear(128, {layers.num_classes})\n"
    
    model_code += "    def forward(self, x):\n"
    for i, layer in enumerate(layers.layers):
        if layers.activation_function is not None:
            model_code += f"        x = F.{layers.activation_function}(self.layer{i}(x))\n"
            model_code += f"        x = F.max_pool2d(x, 2)\n"

    model_code += f"        x = x.view(-1, 64* 54 * 54)\n"
    model_code += f"        x = F.{layers.activation_function}(self.fc1(x))\n"
    model_code += f"        x = self.fc2(x)\n"
    
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

    if file.filename.split(".")[-1] == "zip":
        comment = f'# {{"dataset_path": "{user_id}/{model_name}/dataset.csv", "dataset_type": "images", "label_column": "label_value"}}\n'
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

    elif file.filename.split(".")[-1] == "csv":
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
async def model_train(train: Train, background_tasks: BackgroundTasks):
    latest_version = minio_client.get_last_model_version(train.user_id, train.model_name)
    latest_version = int(latest_version) + 1 if latest_version is not None else 0
    print("Version: ", latest_version)
    cmd = f"""
    python3 /home/ubuntu-machine/train/train.py \
    -u {train.user_id} \
    -m {train.model_name} \
    -l {train.learning_rate} \
    -o {train.optimizer} \
    -e {train.epochs} \
    -b {train.batch_size} \
    -v {latest_version}
    """
    working_directory = "/home/ubuntu-machine/train"
    background_tasks.add_task(subprocess.run, cmd, cwd=working_directory, shell=True, check=True, )
    return {"message": "Training started!"}
    


@app.get("/train/status")
async def train_status(user_id: str):
    log_entries = read_log_file(os.getenv("TRAIN_PATH") + f"/temp/{user_id}/" + "model.log")
    if log_entries is None or len(log_entries) == 0:
        return JSONResponse("Model not training!", status_code=500)
    elif "error" in log_entries[-1]["message"]:
        return JSONResponse("Model did not finished training!", status_code=500)
    elif "finished" in log_entries[-1]["message"]:
        for entry in log_entries:
            if "hash" in entry["message"]:
                hash = entry["message"].split(":")[-1].strip()
                shutil.rmtree(f'/home/ubuntu-machine/train/temp/{user_id}')

                return JSONResponse({
                    "message": "Training finished!",
                    "hash": hash
                }, status_code=200)
    
    return JSONResponse("Still training!", status_code=400)
            


@app.get("/model/versions")
async def model_versions(user_id: str, model_name: str):
    result = minio_client.get_model_versions(user_id, model_name)

    if result is None:
        JSONResponse("Couldn't fetch the model versions!", status_code=500)

    return result


@app.get("/model/weight")
async def model_weight(user_id: str, model_name: str, version: str, hash: str):
    client = Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY")
    )
    with tempfile.NamedTemporaryFile() as tmp:
        result = client.fget_object(user_id, f"{model_name}/model_weights/{version}.pth", tmp.name)
        if result is None:
            return JSONResponse("Couldn't fetch the model versions!", status_code=500)
        
        with open(tmp.name, "rb") as f:
            data = f.read()
            hash_sha256 = sha256()
            hash_sha256.update(data)
            computed_hash = hash_sha256.hexdigest()

        if computed_hash != hash:
            return JSONResponse("Model weights have been altered!", status_code=500)
        
    temp_name = "model.pth"
        
    result = client.fget_object(user_id, f"{model_name}/model_weights/{version}.pth", temp_name)

    if result is None:
        return JSONResponse("Couldn't fetch the model versions!", status_code=500)

    return FileResponse(temp_name, media_type='application/octet-stream', headers={
            "Content-Disposition": f"attachment; filename={version}.pth"
        })

@app.get("/model/version/last")
async def model_versions(user_id: str, model_name: str):
    result = minio_client.get_last_model_version(user_id, model_name)

    if result is None:
        JSONResponse("Couldn't fetch the model versions!", status_code=500)

    return result
