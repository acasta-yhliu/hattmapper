import json
from treemapper import eval

def lambda_handler(event, context):
    eval()
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
