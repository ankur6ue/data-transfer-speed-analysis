
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import os
import ray
import tempfile
from .utils import create_s3_client, boto_response_ok, create_role, create_sqs_client, poll_sqs, exit_spin_loop
from pathlib import Path
import logging
import shutil
import time


@ray.remote(num_returns=2)
def load_model(model_name, bucket_name, cache_location=True):
    logging.basicConfig(level=logging.INFO)
    logging.info("reading data about model {0} from bucket {1}".format(model_name, bucket_name))
    model_data, s3_download_time = load_model_impl(model_name, bucket_name, cache_location)
    return model_data, s3_download_time


def load_model_impl(model_name, bucket_name, cache_location=True):
    # Logic: first see if we previously downloaded the model to a temporary directory. If so, a file in .cache directory
    # will have the location of the model. If not, download the model from S3 to a temporary directory, add it to cache
    # and read the model from the temporary directory.

    cache_dir = os.path.join(Path.home(), '.cache', model_name)
    if os.path.exists(os.path.join(cache_dir, 'cache')):
        with open(os.path.join(cache_dir, 'cache'), 'r') as f:
            dest_dir = f.readline()
            logging.info('found previously downloaded model location {0} in the cache'.format(dest_dir))
    else:
        logging.info('no cached model found, attempting to load {0} from S3 bucket {1}'.format(model_name, bucket_name))
        s3_client = create_s3_client()
        if s3_client:
            resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/' + model_name + "/", Delimiter="/", MaxKeys=100)
            if boto_response_ok(resp) and resp.get("Contents"):
                tmpdirname = tempfile.mkdtemp()
                dest_dir = tmpdirname
                logging.info('creating temporary directory: {0}'.format(tmpdirname))
                # write to cache so we can avoid downloading next time
                if cache_location:
                    try:
                        logging.info("creating directory {0}".format(cache_dir))
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise ValueError("error creating temporary directory {0}".format(exc))
                    logging.info('created temporary directory: {0}'.format(tmpdirname))

                    with open(os.path.join(cache_dir, 'cache'), 'w') as f:
                        f.write(tmpdirname)
                        logging.info('wrote tmp directory path to .cache')
                start = time.time()
                for item in resp["Contents"]:
                    _, filename = os.path.split(item['Key'])
                    dest_path = os.path.join(tmpdirname, filename)

                    s3_client.download_file(bucket_name, item['Key'], dest_path)
                s3_download_time = time.time() - start
                logging.info("s3 download time: {0}".format(s3_download_time))

            else:
                raise ValueError("error downloading {0} from s3".format(model_name))
        else:
            raise ValueError("error creating S3 client while trying to download speech2text model")
    if os.path.exists(dest_dir):
        objs = {}
        objs['dest_dir'] = dest_dir
        # Iterate over all the files in directory
        for folder_name, subfolders, filenames in os.walk(dest_dir):
            for filename in filenames:
                # create complete filepath of file in directory
                filepath = os.path.join(folder_name, filename)
                with open(filepath, mode='rb') as file:  # b is important -> binary
                    fileContent = file.read()
                    objs[filename] = fileContent
        if not cache_location: # if we are not using caching, delete the temporary directory
            shutil.rmtree(dest_dir)
        return objs, s3_download_time

    else:
        raise ValueError("error loading model {0}".format(model_name))
