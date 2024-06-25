# Harbor Object Detection Project

This project is designed to detect different types of ships and vessels from aerial images using computer vision techniques. The project involves three main steps: preprocessing, training, and prediction. This README provides instructions to get started with running the project on Microsoft Azure and using Weights and Biases (wandb) for experiment tracking. Additionally, it includes guidelines for creating a Kubeflow pipeline to automate the entire process.

## Prerequisites
- Microsoft Azure account with access to Azure Storage and Azure Virtual Machines
- Python 3.8+
- Azure CLI

## Setup Instructions

### Request Access to Azure Storage
To access the test dataset stored in an Azure Storage account, request access from your Azure administrator. Generate temporary credentials using the Azure CLI and configure the Azure Storage client.

1. **Authenticate with Azure:**
    ```bash
    az login
    ```

2. **Set the Subscription:**
    ```bash
    az account set --subscription YOUR_SUBSCRIPTION_ID
    ```

3. **Configure Azure Storage Client:**
    ```python
    from azure.storage.blob import BlobServiceClient

    connection_string = "your_connection_string"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client("your-container")
    ```

### Setup MLFlow
1. **Check for Existing MLFlow Server:**
   Before setting up a new MLFlow server, check with your team if there is already a shared MLFlow server available.

2. **Setup MLFlow Server on a Separate VM Instance:**

    1. **Launch an VM Instance:**
        - Choose an instance type
        - Configure security group and key pair.

    2. **Connect to the Instance:**
        ```bash
        ssh -i "your-key-pair.pem" user@your-mlflow-instance-public-dns
        ```

    3. **Install Dependencies:**
        ```bash
        sudo yum install git -y
        sudo yum install -y python3
        pip3 install mlflow
        ```

    4. **Start MLFlow Server:**
        ```bash
        mlflow server \
          --backend-store-uri sqlite:///mlflow.db \
          --default-artifact-root azure://your-azure-storage/mlflow/ \
          --host 0.0.0.0
        ```

![MLFlow](../../images/mlflow.png)

### Provision a Virtual Machine
1. **Launch a Virtual Machine:**
    - Choose a machine type with GPU (e.g., `Standard_NC6`).
    - Configure network security group and SSH keys.

2. **Connect to the Instance:**
    ```bash
    ssh -i "your-key-pair.pem" your-username@your-vm-public-ip
    ```

### Clone the Repository
1. **Install Git:**
    ```bash
    sudo apt-get install git -y
    ```

2. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/harbor-object-detection.git
    cd harbor-object-detection
    ```

### Setup Azure CLI
1. **Install Azure CLI:**
    ```bash
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    ```

## Running the Project

After you've SSH'd into the the machine and cloned your Git commit there, you can run your jobs.

### Configure MLFlow Tracking:


```bash
az account set --subscription <subscription>
az configure --defaults workspace=<workspace> group=<resource-group> location=<location>
az ml workspace show --query mlflow_tracking_uri
```

### Preprocessing
```bash
python preprocessing.py --csv_path data/raw/labels.csv --images_path data/raw/images.zip --output_path data/processed/dataset.npz
```

### Training
```bash
python training.py --input_path data/processed/dataset.npz --epochs 50 --learning_rate 0.001 --batch_size 32 --dataset_name harbor_dataset
```

### Prediction
```bash
python prediction.py --model_path models/harbor_model.h5 --test_data data/processed/dataset_test.npz --output_path predictions/
```

## Save Artifacts to Azure Blob Storage via MLFlow

```python
import mlflow
import os

os.environ["AZURE_TENANT_ID"] = "<AZURE_TENANT_ID>"
os.environ["AZURE_CLIENT_ID"] = "<AZURE_CLIENT_ID>"
os.environ["AZURE_CLIENT_SECRET"] = "<AZURE_CLIENT_SECRET>"

mlflow.set_tracking_uri(mlflow_tracking_uri)

with mlflow.start_run():
    mlflow.log_param("epochs", 50)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_artifact("models/harbor_model.h5")
    mlflow.log_artifact("predictions/")
```

## Chaining jobs and pipelining

## Airflow Pipeline

![Airflow](../../images/airflow.png)

1. **Install Airflow:**
    ```bash
    pipx install apache-airflow
    ```

2. **Create a DAG:**
    ```python
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    from datetime import datetime
    import boto3
    import os

    def download_data():
        s3 = boto3.client('s3')
        s3.download_file('your-s3-bucket', 'data/raw/labels.csv', '/path/to/labels.csv')
        s3.download_file('your-s3-bucket', 'data/raw/images.zip', '/path/to/images.zip')

    def preprocess():
        import subprocess
        subprocess.run(["python", "preprocess.py", "--csv_path", "/path/to/labels.csv", "--images_path", "/path/to/images.zip", "--output_path", "data/processed/dataset.npz"])

    def train():
        import subprocess
        subprocess.run(["python", "train_model.py", "--input_path", "data/processed/dataset.npz", "--epochs", "50", "--learning_rate", "0.001", "--batch_size", "32", "--dataset_name", "harbor_dataset"])

    def predict():
        import subprocess
        subprocess.run(["python", "predict.py", "--model_path", "models/harbor_model.h5", "--test_data", "data/processed/dataset_test.npz", "--output_path", "predictions/"])

    default_args = {
        'owner': 'airflow',
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
    }

    dag = DAG('harbor_detection_pipeline', default_args=default_args, schedule_interval='@daily')

    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        dag=dag,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess,
        dag=dag,
    )

    train_task = PythonOperator(
        task_id='train',
        python_callable=train,
        dag=dag,
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag,
    )

    download_task >> preprocess_task >> train_task >> predict_task
    ```

3. **Deploy the DAG:**
    - Save the DAG file (e.g., `harbor_detection_dag.py`) to the Airflow DAGs folder.
    - Start the Airflow web server and scheduler.