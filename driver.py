import ray
import logging
from src.speech2text import Speech2Text
from src.sync import Speech2TextSyncActor, NERSyncActor
from dotenv import dotenv_values
from src.load_model import load_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import logging
from ray.util.queue import Queue
# Initiate a driver.
# ray.init(address="auto")

# runtime_env={"working_dir": working_dir,
#                           "excludes":["/models/*", "/scratch/*", ".cache/", "*.pkl", "*.flac"],
#                           "env_vars":config}

config = dotenv_values("config.env")
# ray.init(num_cpus=8, runtime_env={"env_vars":config})
working_dir = "/home/ankur/dev/apps/ML/learn/ray/dataload-time-test/"

# os.environ["RAY_LOG_TO_STDERR"] = "1"
ray.init(address="ray://54.89.13.234:31912",
         runtime_env={"working_dir": working_dir,
                      "excludes":["/models/*", "/scratch/*", ".cache/", "*.pkl", "*.flac"],
                      "env_vars":config})

@ray.remote
def run_workflow(num_actors, method):
    run_workflow_impl(num_actors, method)


def run_workflow_impl(num_actors, method):
    model_name = 's2t-medium-librispeech-asr'
    bucket_name = config.get("S3_BUCKET_NAME")
    # schedule load model on head node
    model_data_ref, s3_download_time_ref = load_model.options(resources={"worker_pod_a": 1}).remote(model_name, bucket_name, cache_location=False)
    model_cfg = {}
    model_cfg["model_name"] = model_name
    model_cfg["bucket_name"] = bucket_name
    model_cfg["model_data_ref"] = model_data_ref
    model_cfg["s3_download_time_ref"] = s3_download_time_ref

    # method 1: in this case, models will be loaded via object replication aross plasma stores
    # actors to process speechtotext
    actor_resources = [{"worker_pod_b": 1}, {"worker_pod_c": 1}, {"worker_pod_d": 1}, {"worker_pod_e": 1},
                       {"worker_pod_f": 1}, {"worker_pod_e": 1}, {"worker_pod_f": 1}]

    actors = [Speech2Text.options(resources=actor_resources[i]).remote(i) for i in range(num_actors)]
    # actors = [Speech2Text.options("speech2text").remote(i) for i in range(num_actors)]
    actor_pool = ray.util.ActorPool(actors)

    if method == 'load_from_obj_store':
        while actor_pool.has_free():
            actor_pool.submit(lambda a, v: a.load_model_from_obj_store.remote(v), model_cfg)
        res = []

        while actor_pool.has_next():
            res.append(actor_pool.get_next_unordered())

    if method == 'load_from_s3':
        while actor_pool.has_free():
            actor_pool.submit(lambda a, v: a.load_model_from_s3.remote(v), model_cfg)
        res = []
        while actor_pool.has_next():
            res.append(actor_pool.get_next_unordered())

    return res


def plot_results(res):
    colors = ['deepskyblue', 'limegreen', 'magenta']
    labels = []
    y_start = 1
    num_plots = len(res)
    ticks = [i + 0.25 for i in range(1, num_plots+1)]
    for i in range(len(res)):
        name, rng = list(res[i].keys())[0], list(res[i].values())[0]
        plt.broken_barh([rng], (y_start + i, 0.5), color=colors[i%3])
        labels.append(name)
    plt.title('Data Transfer Time (sec)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(ticks=ticks,
               labels=labels, fontsize=12)
    plt.tick_params(left=False)
    plt.show()


num_actors = 5
method = 'load_from_obj_store'
method = 'load_from_s3'
res = run_workflow_impl(num_actors, method)
plot_results(res)
avg = 0
for elem in res:
    for k, v in elem.items():
        avg += v[1] - v[0]
avg = avg/len(res)
print("average data load time using method {0}: {1}".format(method, avg))
# df = ray.get(run_workflow.remote(num_actors, method))
print("ok")
# model_data_ref = ray.get(load_model.options(resources={"worker_pod_a": 1}).remote(model_name, bucket_name))