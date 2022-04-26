# Data Transfer Speed Comparison: Ray Plasma Store vs. S3
This project describe the results of some experiments regarding data transfer speed between Ray actors via object replication across plasma stores of Ray workers vs. downloading data from S3 into the process memory of Ray actors. It uses a speech2text application that runs the Huggins Face [S2T-MEDIUM-LIBRISPEECH-ASR](https://huggingface.co/facebook/s2t-medium-librispeech-asr) on an audio file as the test bed. See this [post](https://www.telesens.co/2022/04/23/data-transfer-speed-comparison-ray-plasma-store-vs-s3/) for details

The project assumes a solid understanding of Ray and kubernetes and is designed to be run on a Ray-kubernetes cluster. To run the project on the cloud, you'll need to set up a Ray kubernetes cluster and an S3 bucket with the speech2text model files along with IAM Roles and bucket policies so that code running on the nodes of a kubernetes cluster can obtain STS credentials for the bucket. However, to make it easier to get started, I've added a `local run` option. To use this option,

- Download the [S2T-MEDIUM-LIBRISPEECH-ASR](https://huggingface.co/facebook/s2t-medium-librispeech-asr) model files to models/librispeech-medium-asr

- Install the packages listed in requirements.txt to your python environment
- Create a directory called .cache in your home directory. In this directory, create a file called cache, containing the absolute path to the mdoel directory included with this repo. The download_model.py file will look for this file before it tries to download the model from S3. 

To run the system on the cloud, you must (this assumes a solid working knowledge of how IAM works on AWS):
- Create an S3 bucket and IAM role (role 1) with an attached bucket read policy
- Upload the contents of the model directory to a prefix in this bucket
- Add a trust relationship between the IAM role attached to the worker nodes of your kubernetes cluster and role 1, so that code running on your worker nodes is able to assume role 1 and pull data from the S3 bucket you created. 
- Populate AWS_ROLE_ARN and S3_BUCKET_NAME in config.env with these values
- Expose the service created by the Ray kubernetes controller as a nodeport service. You may also need to open the corresponding port in the security group attached to the worker nodes of your kubernetes cluster. 
- Use the IP address of any worker node of your cluster and the nodeport of the service created above for the value of the RAY_CLUSTER_URL variable in config.env (eg., RAY_CLUSTER_URL=ray://node_ip:node_port). This is the address of the Ray cluster used by driver.py when running in cloud mode. 

I may be able to provide some limited support if these instructions don't work. 