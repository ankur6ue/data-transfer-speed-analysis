This project describe the results of some experiments regarding data transfer speed between Ray actors via object replication across plasma stores of Ray workers vs. downloading data from S3 into the process memory of Ray actors. 

The project assumes a solid understanding of Ray and kubernetes and is designed to be run on a Ray-kubernetes cluster and requires setting up an S3 bucket with model files and setting up IAM so that code running on the nodes of a kubernetes cluster can obtain STS credentials for the bucket. However, to make it easier to get started, I have included the model files with this repo and added a local run option. 

You'll need to install the packages listed in requirements.txt to your python environment, and create a directory called .cache in your home directory. In this directory, create a file called cache, containing the absolute path to the mdoel directory included with this repo. The download_model.py will look for this file before it tries to download the model from S3. 

To run the system on the cloud, you must (this assumes a solid working knowledge of how IAM works on AWS):
- Create an S3 bucket and IAM role (role 1) with an attached bucket read policy
- Upload the contents of the model directory to a prefix in this bucket
- Add a trust relationship between the IAM role attached to the worker nodes of your kubernetes cluster and role 1, so that code running on your worker nodes is able to assume role 1 and pull data from the S3 bucket you created. 
- Populate AWS_ROLE_ARN and S3_BUCKET_NAME in config.env with these values
- Expose 