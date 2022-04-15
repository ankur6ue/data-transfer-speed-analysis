from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import os
import io
import time
import ray
import csv
import tempfile
import logging
import sys
import gc
from urllib.parse import urlparse
from .utils import create_s3_client, boto_response_ok, create_role, create_sqs_client, poll_sqs, exit_spin_loop
from pathlib import Path
import shutil

# use memory aware scheduling, so these actors aren't scheduled on the same worker pod, leading to OOM
# see: https://docs.ray.io/en/latest/ray-core/memory-management.html
#@ray.remote(num_cpus=1, memory=5000 * 1024 * 1024, max_task_retries=0)
@ray.remote
class Speech2Text(object):
    def __init__(self, id):
        self.start = time.time()
        self.id = id

    def load_model_from_obj_store(self, model_cfg):
        logging.basicConfig(level=logging.INFO)
        logging.info("starting actor {0}".format(self.id))
        timing = {}
        start = time.time() - self.start
        model_data_ref = model_cfg.get("model_data_ref")
        s3_download_time_ref = model_cfg.get("s3_download_time_ref")
        model_data = ray.get(model_data_ref)
        s3_download_time = ray.get(s3_download_time_ref)

        logging.info("s3_download_time: {0}".format(s3_download_time))
        timing["data_replication {0}".format(self.id)] = (start, time.time() - self.start - s3_download_time)
        # generate temporary directory name
        tmp_dir = tempfile.mkdtemp()
        logging.info('creating temporary directory: {0}'.format(tmp_dir))
        # create tmp_dir
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        for k, v in model_data.items():
            if k is not "dest_dir":
                filepath = os.path.join(tmp_dir, k)
                with open(filepath, mode='wb') as file:  # b is important -> binary
                    file.write(v)

        self.model = Speech2TextForConditionalGeneration.from_pretrained(tmp_dir)
        self.processor = Speech2TextProcessor.from_pretrained(tmp_dir)
#        timing["data_replication {0}".format(self.id)] = (start, time.time() - self.start)
        shutil.rmtree(tmp_dir)
        logging.info('successfully loaded model')
        self.test_model(logging)
        return timing

    def load_model_from_s3(self, model_cfg):
        logging.basicConfig(level=logging.INFO)
        logging.info("starting actor {0}".format(self.id))
        timing = {}
        start = time.time() - self.start
        s3_client = create_s3_client()
        bucket_name = model_cfg.get("bucket_name")
        model_name = model_cfg.get("model_name")

        if s3_client:
            resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/' + model_name + "/", Delimiter="/",
                                             MaxKeys=100)
            if boto_response_ok(resp) and resp.get("Contents"):
                tmp_dir = tempfile.mkdtemp()
                logging.info('creating temporary directory: {0}'.format(tmp_dir))
                for item in resp["Contents"]:
                    _, filename = os.path.split(item['Key'])
                    dest_path = os.path.join(tmp_dir, filename)
                    s3_client.download_file(bucket_name, item['Key'], dest_path)
                timing["data_download_from_s3 {0}".format(self.id)] = (start, time.time() - self.start)

                self.model = Speech2TextForConditionalGeneration.from_pretrained(tmp_dir)
                self.processor = Speech2TextProcessor.from_pretrained(tmp_dir)
                # delete tmpdir to prevent disk space from getting filled up
                shutil.rmtree(tmp_dir)
                # os.rmdir(tmp_dir)
                # timing["data_download_from_s3 {0}".format(self.id)] = (start, time.time() - self.start)

                # Run the model on test file for sanity
                self.test_model(logging)
                return timing

    def test_model(self, logging):
        sp = sf.read("2022-03-05-11:46:38.wav")
        inputs = self.processor(sp[0], sampling_rate=16_000, return_tensors="pt")
        generated_ids = self.model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids)[0]
        logging.info(transcription)

