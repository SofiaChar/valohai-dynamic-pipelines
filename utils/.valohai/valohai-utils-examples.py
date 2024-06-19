import valohai
from valohai import Pipeline

preprocess_inputs = {
    "dataset": "s3://valohai-demo-library-data/dynamic-pipelines/train/images.zip",
    "labels": "s3://valohai-demo-library-data/dynamic-pipelines/train/*.csv"
}

preprocess_parameters = {
    "validation_split": 0.3,
    "dataset_names": "[harbor_A]"
}

valohai.prepare(step="preprocess-dataset",
                image="docker.io/noorai/dynamic-pipelines-demo:0.1",
                environment="aws-eu-west-1-g4dn-xlarge",
                default_inputs=preprocess_inputs,
                default_parameters=preprocess_inputs)

# Now run:
# vh yaml step valohai-utils-examples.py

def main(config) -> Pipeline:

    # Create a pipeline called "utilspipeline".
    pipe = Pipeline(name="train-inference-pipeline", config=config)

    # Define the pipeline nodes.
    preprocess = pipe.execution("preprocess")
    train = pipe.execution("train-model")
    inference = pipe.execution("batch-inference")

    # Configure the pipeline, i.e. define the edges.
    preprocess.output("*.npz").to(train.input("data"))
    preprocess.output("model*.pkl").to(inference.input("model"))

    return pipe

# Now run
# vh yaml pipeline valohai-utils-examples.py
