import hashlib
import os
import sys
import json
import torch
import GPUtil
import shutil
import inspect
import logging
import argparse
import tempfile
import importlib
import torch.nn as nn
from minio import Minio
from pathlib import Path
import torch.optim as optim
from minio.error import S3Error
from dotenv import load_dotenv
from torch.utils.data import random_split, DataLoader

load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument("-u", "--user_id", type=str, help="ID of the user.", required=True)
parser.add_argument("-m", "--model_name", type=str, help="Name for the model.", required=True)
parser.add_argument("-l", "--learning_rate", type=float, help="Learning for the model.", required=True)
parser.add_argument("-o", "--optimizer", type=str, help="Optimizer implementation to use", required=True)
parser.add_argument("-e", "--epochs", type=str, help="Number of epoches to run.", required=True)
parser.add_argument("-b", "--batch_size", type=str, help="Batch size for data.", required=True)
parser.add_argument("-v", "--version", type=str, help="Version of weights.", required=True)

args = parser.parse_args()

logging.basicConfig(filename=f'temp/{args.user_id}/model.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


gpu_list = [((i.memoryUsed / i.memoryTotal) * 100, i.id) for i in GPUtil.getGPUs() if (i.memoryUsed / i.memoryTotal) * 100 < 10.0]

gpu_id = sorted(gpu_list, key=lambda x: x[0])[-1][1]


client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY")
)


try:
    response = client.get_object(args.user_id, f"{args.model_name}/{args.model_name}.py")
    os.makedirs(f'temp/{args.user_id}', exist_ok=True)

    with open(f"temp/{args.user_id}/{args.model_name}.py", "wb") as f:
        for data in response.stream(32*1024):
            f.write(data)
except S3Error:
    logging.error("error")
    exit(1)
finally:
    response.close()
    response.release_conn()


try:
    response = client.get_object(args.user_id, f"{args.model_name}/dataset.py")

    with open(f"temp/{args.user_id}/dataset.py", "wb") as f:
        for data in response.stream(32*1024):
            f.write(data)
except S3Error:
    logging.error("error")
    exit(1)
finally:
    response.close()
    response.release_conn()

module_name = args.model_name
module_file_path = f'./temp/{args.user_id}/{args.model_name}.py'

spec = importlib.util.spec_from_file_location(module_name, module_file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

for name, obj in inspect.getmembers(module, inspect.isclass):
    if obj.__module__ == module_name:
        model_instance = obj() 
        break

with open(f'./temp/{args.user_id}/dataset.py') as f:
    json_data = f.readline().strip().replace("#", "")
    dataset_dict = json.loads(json_data)

spec = importlib.util.spec_from_file_location("dataset",  f'./temp/{args.user_id}/dataset.py')
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module) 

dataset_instance = getattr(module, "CustomDataset")(**dataset_dict)

train_size = int(0.8 * len(dataset_instance))
valid_size = len(dataset_instance) - train_size
train_dataset, valid_dataset = random_split(dataset_instance, [train_size, valid_size])


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

device = f"cuda:{gpu_id}" if isinstance(gpu_id, int) else "cpu"


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_instance.parameters(), lr=args.learning_rate) if args.optimizer == "SGD" else optim.Adam(model_instance.parameters(), lr=args.learning_rate)

for epoch in range(args.epochs):
    model_instance.train()

    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        outputs, _ = model_instance(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    logging.info(f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}')

    model_instance.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output, _ = model_instance(inputs)

            _, predicted = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        current_accuracy = round(100 * correct / total, 2)
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Validation Accuracy: {current_accuracy}%')


with tempfile.NamedTemporaryFile(delete=False) as tmp:
    torch.save(model_instance.state_dict(), tmp.name)

    hash_sha256 = hashlib.sha256()
    with open(tmp.name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    logging.info(f"hash: {hash_sha256.hexdigest()}")

    try:
        client.fput_object(
            args.user_id, f'{args.model_name}/model_weights/{args.version}.pth', tmp.name
        )
    except S3Error as exc:
        logging.error("error")

logging.info('finished')

shutil.rmtree(f'./temp/{args.user_id}')