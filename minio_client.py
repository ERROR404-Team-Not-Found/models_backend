import os
from typing import Any
from minio import Minio
from minio.api import S3Error
from dotenv import load_dotenv
from io import BytesIO, StringIO

load_dotenv()



class Client:
    def __init__(self):
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY")
        )

    
    def create_user_bucket(self, username: str) -> bool:
        if self.client.bucket_exists(username):
            return True
        
        try:
            self.client.make_bucket(username)
        except S3Error:
            return False
        return True
    
    def upload_file(self, bucket_name: str, path: str, object_name: str, content: str) -> None | str:
        try:
            if isinstance(content, BytesIO) or isinstance(content, StringIO):
                bytes_content = content
            else:
                bytes_content = BytesIO(content.encode('utf-8'))

            result = self.client.put_object(bucket_name, f"{path.lower()}/{object_name}", bytes_content, -1, part_size=1024*1024*5)
            return result.object_name
        except S3Error as e:
            return None
        
    def get_object(self, bucket_name: str, path: str, object_name: str) -> None | Any:
        response = None
        try:
            response = self.client.get_object(bucket_name, f"{path}/{object_name.lower()}")
            return response.data
        except S3Error as e:
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()
            
    def get_last_model_version(self, bucket_name: str, model_name: str) -> None | str:
        try:
            response = self.client.list_objects(bucket_name, prefix=f"{model_name}/model_weights", recursive=True)
            return [l.object_name for l in list(response)][-1].split("/")[-1].split(".")[0]
        except S3Error:
            return None
    
    def get_model_versions(self, bucket_name: str, model_name: str) -> None | str:
        try:
            response = self.client.list_objects(bucket_name, prefix=f"{model_name}/model_weights", recursive=True)
            return [l.object_name.split("/")[-1].split(".")[0] for l in list(response)]
        except S3Error:
            return None
        
    def get_model_version(self, bucket_name: str, model_name: str, model_version: str) -> None | str:
        response = None
        try:
            response = self.client.get_object(bucket_name, f"{model_name}/model_weights/{model_version}.pth")
            return response
        except S3Error:
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()
    
    def get_models(self, bucket_name: str) -> None | str:
        try:
            response = self.client.list_objects(bucket_name)
            return [i.object_name.split("/")[0] for i in list(response)]
        except S3Error:
            return None

