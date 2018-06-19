export MODEL_NAME="Potro" # change to your model name
export CURRENT_DATE=`date +%Y%m%d_%H%M%S`
export JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}
export BUCKET_NAME=stevenwho
export CLOUD_CONFIG=config/config.yaml
export JOB_DIR=gs://stevenwho/Potro/jobs/$JOB_NAME
export MODULE=trainer.task
export PACKAGE_PATH=./trainer
export REGION=us-central1
export RUNTIME=1.8
export DATA_DIR=gs://stevenwho/Potro/input
export OUTPUT_FILE=out_${CURRENT_DATE}



gcloud ml-engine jobs submit training ${JOB_NAME} \
    --package-path=${PACKAGE_PATH} \
    --staging-bucket gs://${BUCKET_NAME} \
    --module-name=${MODULE} \
    --runtime-version=${RUNTIME} \
    --region=${REGION} \
    --config=${CLOUD_CONFIG} \
    -- \
    --data-dir=${DATA_DIR}\
    --output-name=${OUTPUT_FILE}
 
    # --test-file=${TEST_FILE} \
    # -output-file=${OUTPUT_FILE}-
# export OUTPUT_FILE=gs://stevenwho/Potro/out.csv
# export TRAIN_FILE=gs://stevenwho/train.csv
# export TEST_FILE=gs://stevenwho/test.csv
# --job-dir $JOB_DIR \
# --train-file ./input/train.csv --test-file ./input/test.csv --output_file ./model/out.csv
export MODULE=trainer.task
export PACKAGE_PATH=./trainer
export DATA_DIR=./input


gcloud ml-engine local train \
    --package-path=${PACKAGE_PATH} \
    --module-name=${MODULE} \
    -- \
    --data-dir=${DATA_DIR} 

  