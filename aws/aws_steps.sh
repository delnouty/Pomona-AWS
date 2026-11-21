# P8 - AWS commands, step by step (same numbering as the textbook).
# Copy the commands from this file, not from the PDF (line breaks in a PDF can break a command).
# Replace only the text written <like-this>.
# Do NOT run this whole file at once: run each step, then compare with 'what you should see' in the book.

######################################################################
# Step 6 - Install the AWS command line (AWS CLI)
######################################################################
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
aws --version

######################################################################
# Step 7 - Connect the AWS CLI to your account
######################################################################
aws configure

aws sts get-caller-identity

######################################################################
# Step 8 - Write your settings in one file
######################################################################
mkdir -p ~/p8
cat > ~/p8/aws.env <<'EOF'
export AWS_REGION=eu-west-3
export AWS_DEFAULT_REGION=eu-west-3
export BUCKET=p8-fruits-<your-initials>-2026
EOF
source ~/p8/aws.env
echo $BUCKET $AWS_REGION

######################################################################
# Step 9 - Create the bucket in Paris
######################################################################
aws s3api create-bucket --bucket $BUCKET \
    --create-bucket-configuration LocationConstraint=eu-west-3

aws s3api get-bucket-location --bucket $BUCKET
aws s3api get-public-access-block --bucket $BUCKET
aws s3api get-bucket-encryption --bucket $BUCKET

aws s3api put-public-access-block --bucket $BUCKET --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

######################################################################
# Step 10 - Upload the images
######################################################################
cd ~/p8/my-p8
TEST=$(find data -type d -name Test | head -1); echo "$TEST"
find "$TEST" -name '*.jpg' | wc -l                         # number of local images
aws s3 sync "$TEST" s3://$BUCKET/Test --only-show-errors   # several minutes, prints nothing
aws s3 ls s3://$BUCKET/Test/ --recursive --summarize | tail -2

######################################################################
# Step 11 - Upload the bootstrap script
######################################################################
cd ~/p8/my-p8
cp ../p8-notebook-road/bootstrap-emr.sh .
cat bootstrap-emr.sh                        # read it: 5 commands
aws s3 cp bootstrap-emr.sh s3://$BUCKET/bootstrap-emr.sh

######################################################################
# Step 12 - The SSH key and the EMR roles
######################################################################
mkdir -p ~/.ssh
aws ec2 create-key-pair --key-name p8-emr-key --key-type rsa --key-format pem \
    --query KeyMaterial --output text > ~/.ssh/p8-emr-key.pem
chmod 400 ~/.ssh/p8-emr-key.pem
head -1 ~/.ssh/p8-emr-key.pem
aws emr create-default-roles

git add bootstrap-emr.sh && git commit -m "Add the EMR bootstrap script"

######################################################################
# Step 13 - Create the cluster
######################################################################
source ~/p8/aws.env
cat > ~/p8/emr-config.json <<EOF
[
  {"Classification": "jupyter-s3-conf",
   "Properties": {"s3.persistence.enabled": "true", "s3.persistence.bucket": "$BUCKET"}},
  {"Classification": "spark-defaults",
   "Properties": {"spark.sql.execution.arrow.maxRecordsPerBatch": "256"}}
]
EOF
cat ~/p8/emr-config.json                    # your bucket name must appear

aws emr list-release-labels --filters Prefix=emr-7 --query 'ReleaseLabels[0]' --output text

CLUSTER_ID=$(aws emr create-cluster \
  --name p8-fruits \
  --release-label <release> \
  --applications Name=Hadoop Name=Spark Name=Livy Name=JupyterHub \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
                    InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
  --use-default-roles \
  --ec2-attributes KeyName=p8-emr-key \
  --bootstrap-actions Path=s3://$BUCKET/bootstrap-emr.sh,Name=install-python-libs \
  --configurations file://$HOME/p8/emr-config.json \
  --log-uri s3://$BUCKET/logs/ \
  --auto-termination-policy IdleTimeout=3600 \
  --query ClusterId --output text)
echo $CLUSTER_ID
echo "export CLUSTER_ID=$CLUSTER_ID" > ~/p8/cluster.env

######################################################################
# Step 14 - Wait until the cluster is ready
######################################################################
aws emr wait cluster-running --cluster-id $CLUSTER_ID
aws emr describe-cluster --cluster-id $CLUSTER_ID --query Cluster.Status.State --output text

aws emr describe-cluster --cluster-id $CLUSTER_ID --query Cluster.Status.StateChangeReason

aws s3 ls s3://$BUCKET/logs/$CLUSTER_ID/node/ --recursive | grep bootstrap-actions
aws s3 cp s3://$BUCKET/logs/$CLUSTER_ID/node/<machine-id>/bootstrap-actions/1/stderr.gz - | gunzip | tail -20

######################################################################
# Step 15 - Open the cluster's firewall to your address only
######################################################################
SG=$(aws emr describe-cluster --cluster-id $CLUSTER_ID \
     --query Cluster.Ec2InstanceAttributes.EmrManagedMasterSecurityGroup --output text)
MYIP=$(curl -s https://checkip.amazonaws.com)
echo "security group: $SG   my address: $MYIP"
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $MYIP/32

######################################################################
# Step 16 - Open the SSH tunnel
######################################################################
DNS=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --query Cluster.MasterPublicDnsName --output text)
echo $DNS
ssh -i ~/.ssh/p8-emr-key.pem -N -L 9443:localhost:9443 -L 18080:localhost:18080 hadoop@$DNS

######################################################################
# Step 21 - Collect the results
######################################################################
aws s3 ls s3://$BUCKET/ --recursive | grep _SUCCESS           # the 3 result folders are complete
aws s3 cp --recursive s3://$BUCKET/Results_PCA_variance ~/p8/my-p8/data/variance_emr

######################################################################
# Step 23 - Terminate the cluster
######################################################################
source ~/p8/aws.env ~/p8/cluster.env
aws emr terminate-clusters --cluster-ids $CLUSTER_ID
aws emr wait cluster-terminated --cluster-id $CLUSTER_ID
aws emr list-clusters --active
aws ec2 describe-instances --filters Name=instance-state-name,Values=running \
    --query 'Reservations[].Instances[].InstanceId'

######################################################################
# End of the project (after the defence)
######################################################################
aws s3 rm s3://$BUCKET --recursive          # keep a local copy of what you need first
aws s3api delete-bucket --bucket $BUCKET
aws ec2 delete-key-pair --key-name p8-emr-key && rm ~/.ssh/p8-emr-key.pem
