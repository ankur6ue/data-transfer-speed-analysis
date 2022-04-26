import ray
import logging
from src.speech2text import Speech2Text
import os
from dotenv import dotenv_values
from src.download_model import download_model
import matplotlib.pyplot as plt


@ray.remote
def run_workflow(num_actors, method):
    return run_workflow_impl(num_actors, method)


def run_workflow_impl(num_actors, method):
    model_name = 's2t-medium-librispeech-asr'
    bucket_name = os.getenv("S3_BUCKET_NAME")
    actor_resources = [{"worker_pod_b": 1}, {"worker_pod_c": 1}, {"worker_pod_d": 1}, {"worker_pod_e": 1},
                       {"worker_pod_f": 1}, {"worker_pod_e": 1}, {"worker_pod_f": 1}]

    actors = [Speech2Text.options(resources=actor_resources[i]).remote(i) for i in range(num_actors)]
    # actors = [Speech2Text.options("speech2text").remote(i) for i in range(num_actors)]
    actor_pool = ray.util.ActorPool(actors)

    model_cfg = {}
    model_cfg["model_name"] = model_name
    model_cfg["bucket_name"] = bucket_name

    # method 1: in this case, speech2text model will first be downloaded from S3 to memory of the downloading task.
    # Then, the model data will be copied to the RAM of each actor that needs it, via object replication
    # across plasma stores

    if method == 'load_from_obj_store':
        # Start download model task. This can also be an actor.
        model_data_ref, s3_download_time_ref = download_model.options(resources={"worker_pod_a": 1}).remote(model_name,
                                                                                bucket_name, cache_location=False)
        # Because this is a remote task, it returns immediately with references to the return values. Ray will
        # schedule the execution of the remote function on a worker pod that matches the resource specification
        model_cfg["model_data_ref"] = model_data_ref
        model_cfg["s3_download_time_ref"] = s3_download_time_ref
        # Now start actors to load the speech2text model data from the object store of the pod download task
        # and run the model on a test audio file and log the transcription
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


def main():
    # read env vars
    config = dotenv_values("config.env")
    # set environment variables from config.env
    for k, v in config.items():
        os.environ[k] = v

    method = os.getenv("METHOD")
    env = os.getenv("ENV")
    working_dir = os.getenv("WORKING_DIR")
    num_actors = (int)(os.getenv("NUM_ACTORS"))
    ray_cluster_url = os.getenv("RAY_CLUSTER_URL")
    if env == "local":
        # create a local ray cluster
        resources = {'num_cpus': 5, "worker_pod_b": 1, "worker_pod_c": 1, "worker_pod_d": 1, "worker_pod_e": 1, "worker_pod_f": 1,
                     "worker_pod_a": 1}
        ray.init(resources=resources)
    else:
        ray.init(address=ray_cluster_url,
                 runtime_env={"working_dir": working_dir,
                              "excludes": ["/models/*"],
                              "env_vars": config})

    # returns a list of dictionaries, where each element of the list is a key value pair
    # key (string): method actor_number, value (tuple): start and end processing times
    # eg., data_replication 0: (0.1, 5.1)
    res = run_workflow_impl(num_actors, method)
    # plot a bar chart of processing time for each actor
    plot_results(res)
    # calculate the average processing time by summing the total processing time for each actor and
    # dividing by the number of actors
    avg = 0
    for elem in res:
        for k, v in elem.items():
            avg += v[1] - v[0]
    avg = avg / len(res)
    print("average data load time using method {0}: {1}".format(method, avg))


if __name__ == "__main__":
    main()


