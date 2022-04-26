from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import os
import time
import ray
import tempfile
import logging
from .utils import create_s3_client, boto_response_ok
from pathlib import Path
import shutil

# use memory aware scheduling, so these actors aren't scheduled on the same worker pod, leading to OOM
# see: https://docs.ray.io/en/latest/ray-core/memory-management.html
@ray.remote(num_cpus=1, memory=2000 * 1024 * 1024, max_task_retries=0)
#@ray.remote
class Speech2Text(object):
    def __init__(self, id):
        self.start = time.time()
        self.id = id
        logging.basicConfig(level=logging.INFO)

    def load_model_from_obj_store(self, model_cfg):
        """
        Loads speech2text model data from plasma store into Actor RAM, saves it to a temporary location on the disk
        and initializes the speech2text model. Runs the model on a test audio file to verify all went well
        :param model_cfg:
        :return: timing info about transferring data across plasma stores
        """
        logging.info("starting actor {0}".format(self.id))
        timing = {}
        start = time.time() - self.start
        model_data_ref = model_cfg.get("model_data_ref")
        s3_download_time_ref = model_cfg.get("s3_download_time_ref")
        model_data = ray.get(model_data_ref)
        s3_download_time = ray.get(s3_download_time_ref)

        logging.info("s3_download_time: {0}".format(s3_download_time))
        # We subtract the download from s3 time, because we are only want to measure the time to transfer
        # data across plasma stores
        timing["data_replication {0}".format(self.id)] = (start, time.time() - self.start - s3_download_time)
        # generate temporary directory name to save model data
        tmp_dir = tempfile.mkdtemp()
        logging.info('creating temporary directory: {0}'.format(tmp_dir))
        # create tmp_dir
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        for k, v in model_data.items():
            if k is not "dest_dir":
                filepath = os.path.join(tmp_dir, k)
                with open(filepath, mode='wb') as file:  # b is important -> binary
                    file.write(v)

        # Load model and processor
        self.model = Speech2TextForConditionalGeneration.from_pretrained(tmp_dir)
        self.processor = Speech2TextProcessor.from_pretrained(tmp_dir)
        # Delete temporary directory
        shutil.rmtree(tmp_dir)
        logging.info('successfully loaded model')
        # Test the model
        self.test_model(logging)
        return timing

    def load_model_from_s3(self, model_cfg):
        """
        Loads speech2text model data from S3 into to a temporary path on the disk
        and initializes the speech2text model. Runs the model on a test audio file to verify all went well
        :param model_cfg:
        :return: timing info about downloading data from S3
        """
        logging.info("starting actor {0}".format(self.id))
        timing = {}
        start = time.time() - self.start
        s3_client = create_s3_client(logging)
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
        else:
            raise ValueError("Error creating s3 boto client")

    def test_model(self, logging):
        # sp = sf.read("2022-03-05-11:46:38.wav")
        sp = sf.read("1089-134686-0000.flac")
        inputs = self.processor(sp[0], sampling_rate=16_000, return_tensors="pt")
        generated_ids = self.model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids)[0]
        logging.info(transcription)

