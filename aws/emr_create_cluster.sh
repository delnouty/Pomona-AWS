#!/bin/bash
# Crée le cluster EMR (région européenne : RGPD) avec arrêt automatique.
# Reproduit la configuration du mode opératoire P8 (docs/mode_operatoire/).
#
# Prérequis (une seule fois) :
#   aws configure                  # clés IAM, jamais dans le dépôt
#   aws emr create-default-roles   # EMR_DefaultRole et EMR_EC2_DefaultRole
#   bucket S3 dans la même région, avec aws/emr_bootstrap.sh uploadé sur s3://$BUCKET/
#
# Usage : BUCKET=mon-bucket-eu KEY_NAME=ma-cle-ec2 bash aws/emr_create_cluster.sh
set -euo pipefail

REGION="${REGION:-eu-west-3}"            # Paris, comme le mode opératoire
BUCKET="${BUCKET:?définir BUCKET}"
KEY_NAME="${KEY_NAME:?définir KEY_NAME}"

# emr-6.3.0 : Spark 3.1.1, JupyterHub 1.2.0, application TensorFlow 2.4.1.
# À vérifier : que cette version soit encore proposée sur le compte ; l'application
# TensorFlow n'existe plus dans EMR 7.x (il faudrait alors pip install tensorflow).
RELEASE="${RELEASE:-emr-6.3.0}"

# Les notebooks JupyterHub sont enregistrés sur S3 plutôt que sur le disque des nœuds
CONF="$(mktemp)"
cat > "$CONF" <<EOF
[{"Classification":"jupyter-s3-conf","Properties":{"s3.persistence.bucket":"$BUCKET","s3.persistence.enabled":"true"}}]
EOF

aws emr create-cluster \
  --region "$REGION" \
  --name "fruits-emr" \
  --release-label "$RELEASE" \
  --applications Name=Hadoop Name=Spark Name=JupyterHub Name=TensorFlow \
  --configurations "file://$CONF" \
  --instance-groups \
      InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
      InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
  --use-default-roles \
  --ec2-attributes KeyName="$KEY_NAME" \
  --bootstrap-actions Path="s3://$BUCKET/bootstrap-emr.sh" \
  --log-uri "s3://$BUCKET/logs/" \
  --auto-termination-policy IdleTimeout=3600
# IdleTimeout=3600 : filet de sécurité budget (1 h d'inactivité).
# Le bootstrap doit être uploadé sous ce nom :
#   aws s3 cp aws/emr_bootstrap.sh s3://$BUCKET/bootstrap-emr.sh
# Pour accéder à JupyterHub : ouvrir le port 22 du groupe de sécurité du nœud maître
# (de préférence "Mon IP"), tunnel SSH `ssh -i <cle>.pem -D 5555 hadoop@<dns-maitre>` + FoxyProxy.
# Toujours terminer après la démo :
#   aws emr terminate-clusters --region $REGION --cluster-ids <j-XXXX>
