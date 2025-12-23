import json
import torch


def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps({"name": "rizin", "age": 2}),
    }
