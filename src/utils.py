import boto3
import requests
import os
from requests import ConnectionError, HTTPError
from botocore.exceptions import ClientError, NoCredentialsError


def walk(root_dir):
    flac_files = {}
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(root_dir):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == '.flac':
                flac_files[file] = os.path.join(root, file)
    return flac_files


def create_role(vault_server_addr: str, vault_role_name: str, aws_role_name: str, token: str):
    url = vault_server_addr + '/v1/aws/roles/' + vault_role_name

    try:
        resp = requests.request('POST', url, headers={"X-Vault-Token": token}, data= \
            {"role_arns": aws_role_name,
             "credential_type": "assumed_role"},
                                timeout=2)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        raise e
    except ConnectionError as e:
        print('We failed to reach a server.')
        raise e
    return resp


def read_sts_creds(vault_server_addr: str, client_token: str, vault_role_name: str, logging):
    # First get client authorization token

    url = vault_server_addr + '/v1/aws/sts/' + vault_role_name
    try:
        resp = requests.request('POST', url, headers={"X-Vault-Token": client_token}, timeout=1)
        # Can also use this:
        # resp = requests.request('GET', url, headers={"Authorization": "Bearer " + client_token}, timeout=1)
    except HTTPError as e:
        logging.info('The vault server couldn\'t fulfill the request.')
        raise e
    except ConnectionError as e:
        logging.info('We failed to reach the vault server.')
        raise e
    if resp.ok:
        return resp.json()


def create_s3_client(logging):
    sts_client = boto3.client('sts')
    aws_role_arn = os.getenv('AWS_ROLE_ARN')
    region = 'us-east-1'
    try:
        assumedRoleObject = sts_client.assume_role(
            RoleArn=aws_role_arn, RoleSessionName="AssumeRoleSession1")
        logging.info("Successfully assumed role: {0}".format(aws_role_arn))
        access_key = assumedRoleObject["Credentials"]["AccessKeyId"]
        secret_key = assumedRoleObject["Credentials"]["SecretAccessKey"]
        session_token = assumedRoleObject["Credentials"]["SessionToken"]
        return boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token
        )
    except (NoCredentialsError, ClientError) as e:
        logging.info("unable to assume role: {0}, attempting to get credentials from vault".format(aws_role_arn))
        vault_server_addr = os.getenv('VAULT_ADDR')
        vault_root_token = os.getenv('VAULT_ROOT_TOKEN')
        vault_role_name = os.getenv('VAULT_S3_ROLE_NAME')


        # This is a one time operation performed during onboarding
        # res = create_role(vault_server_addr, vault_role_name, aws_role_arn, vault_root_token)
        try:
            sts_creds = read_sts_creds(vault_server_addr, vault_root_token, vault_role_name, logging)
            access_key = sts_creds["data"]["access_key"]
            secret_key = sts_creds["data"]["secret_key"]
            session_token = sts_creds["data"]["security_token"]
            return boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token
            )
        except Exception as e:
            logging.info("unable to assume role and obtain creds from vault")
            raise e


def boto_response_ok(resp):
    return resp["ResponseMetadata"]["HTTPStatusCode"] == 200




