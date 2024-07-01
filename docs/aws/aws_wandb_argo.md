# Harbor Vessel Detection

This project is designed to detect different types of ships and vessels from aerial images using computer vision techniques. The project involves three main steps: preprocessing, training, and prediction. This README provides instructions to get started with running the project on AWS and using MLFlow for experiment tracking. Additionally, it includes guidelines for creating an Argo pipeline to automate the entire process.

## Prerequisites
- AWS account with access to S3 and EC2
- Python 3.8+
- AWS CLI
- Boto3
- Weights & Biases
- Argo Workflows

## Setup Instructions

### Setup AWS CLI, Boto3, WandB and Argo
1. **Install AWS CLI:**
    ```bash
    sudo apt-get install awscli
    ```

2. **Install Boto3:**
    ```bash
    pipx install boto3
    ```

3. **Install wandb:**
    ```bash
    pip install wandb
    wandb login
    ```

    Add the following configuration to your scripts:
    ```python
    import wandb

    wandb.init(project='harbor-object-detection')
    ```

4. **Install Argo CLI:**
    ```bash
    curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.1.0/argo-linux-amd64.gz
    gunzip argo-linux-amd64.gz
    chmod +x argo-linux-amd64
    mv ./argo-linux-amd64 /usr/local/bin/argo
    ```

### Request Access to S3 Bucket
To access the test dataset stored in an S3 bucket, request access by submitting a JIRA ticket

Generate temporary credentials using the AWS Security Token Service (STS) and configure Boto3.

1. **Generate Temporary Credentials:**
    ```bash
    aws sts get-session-token --duration-seconds 36000
    ```

2. **Configure Boto3:**
    ```python
    import boto3

    session = boto3.Session(
        aws_access_key_id='YOUR_ACCESS_KEY',
        aws_secret_access_key='YOUR_SECRET_KEY',
        aws_session_token='YOUR_SESSION_TOKEN'
    )
    s3 = session.resource('s3')
    ```

### Provision an EC2 GPU Machine
1. **Launch an EC2 Instance:**
    - Choose an instance type with GPU (e.g., `p3.2xlarge`).
    - Configure security group and key pair.
        - Remember to allow access to port 22 from our internal network.

2. **Connect to the Instance:**
    ```bash
    ssh -i "your-key-pair.pem" ec2-user@your-ec2-instance-public-dns
    ```
![AWS EC2](../../images/aws_ec2.png)

### Clone the Repository
1. **Install Git:**
    ```bash
    sudo yum install git -y
    ```

2. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/harbor-object-detection.git
    cd harbor-object-detection
    ```

> Remember to commit your changes to Git periodically!

> Remember to shut down the machine, when you no longer needed!

## Log Artifacts to Weights and Biases
1. **Log Artifacts:**
    ```python
    import wandb

    wandb.init(project='harbor-object-detection')
    wandb.config.update({
        "epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 32
    })
    wandb.log_artifact('models/harbor_model.h5')
    wandb.log_artifact('predictions/')
    ```

![MLFlow](../../images/wandb.png)

## Running the Project

### Preprocessing
```bash
python preprocess.py --csv_path data/raw/labels.csv --images_path data/raw/images.zip --output_path data/processed/dataset.npz
```

### Training
```bash
python train_model.py --input_path data/processed/dataset.npz --epochs 50 --learning_rate 0.001 --batch_size 32 --dataset_name harbor_dataset
```

### Prediction
```bash
python predict.py --model_path models/harbor_model.h5 --test_data data/processed/dataset_test.npz --output_path predictions/
```

## Save Artifacts to S3 via MLFlow
1. **Log Artifacts:**
    ```python
    import mlflow

    with mlflow.start_run():
        mlflow.log_param("epochs", 50)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_artifact("models/harbor_model.h5")
        mlflow.log_artifact("predictions/")
    ```

2. **Save Model to S3:**
    ```bash
    mlflow artifacts download -r your-run-id -o s3://your-s3-bucket/mlflow-artifacts/
    ```

## Argo Workflow

![Argo Workflow](../../images/argo-workflow.png)

### Define the Workflow
1. **Create the Workflow YAML:**
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: Workflow
    metadata:
      generateName: harbor-detection-pipeline-
    spec:
      entrypoint: main
      templates:
      - name: main
        dag:
          tasks:
          - name: download-data
            template: download-data
          - name: preprocess
            template: preprocess
            dependencies: [download-data]
          - name: train
            template: train
            dependencies: [preprocess]
          - name: predict
            template: predict
            dependencies: [train]

      - name: download-data
        container:
          image: amazonlinux
          command: [sh, -c]
          args: ["aws s3 cp s3://your-s3-bucket/data/raw/labels.csv /data/ && aws s3 cp s3://your-s3-bucket/data/raw/images.zip /data/"]
          volumeMounts:
            - name: data
              mountPath: /data

      - name: preprocess
        container:
          image: python:3.8
          command: ["python", "preprocessing.py"]
          args: ["--csv_path", "/data/labels.csv", "--images_path", "/data/images.zip", "--output_path", "/data/processed/dataset.npz"]
          volumeMounts:
            - name: data
              mountPath: /data

      - name: train
        container:
          image: python:3.8
          command: ["python", "training.py"]
          args: ["--input_path", "/data/processed/dataset.npz", "--epochs", "50", "--learning_rate", "0.001", "--batch_size", "32", "--dataset_name", "harbor_dataset"]
          volumeMounts:
            - name: data
              mountPath: /data

      - name: predict
        container:
          image: python:3.8
          command: ["python", "prediction.py"]
          args: ["--model_path", "/data/models/harbor_model.h5", "--test_data", "/data/processed/dataset_test.npz", "--output_path", "/data/predictions/"]
          volumeMounts:
            - name: data
              mountPath: /data

      volumes:
      - name: data
        emptyDir: {}
    ```

2. **Submit the Workflow:**
    ```bash
    argo submit -n argo --watch harbor-detection-pipeline.yaml
    ```

By following these instructions, you can set up and run the Harbor Object Detection Project on AWS with MLFlow tracking and automate the workflow using Argo Workflows.