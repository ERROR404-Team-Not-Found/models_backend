import os
from io import BytesIO
from minio import Minio
from minio.api import S3Error
from dotenv import load_dotenv

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
            bytes_content = BytesIO(content.encode('utf-8'))

            result = self.client.put_object(bucket_name, f"{path.lower()}/{object_name}", bytes_content, -1, part_size=1024*1024*5)
            return result.object_name
        except S3Error as e:
            return None
