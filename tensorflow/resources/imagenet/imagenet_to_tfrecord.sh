export IMAGENET_HOME=/home/cc/dataset_backup
# # Setup folders
# mkdir -p $IMAGENET_HOME/validation
# mkdir -p $IMAGENET_HOME/train

# # Extract validation and training
# tar xf /home/cc/dataset_backup/ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
# tar xf /home/cc/dataset_backup/ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train

# # Extract and then delete individual training tar files This can be pasted
# # directly into a bash command-line or create a file and execute.
cd $IMAGENET_HOME/train

for f in *.tar; do
	d=`basename $f .tar`
	mkdir $d
	tar xf $f -C $d
done

cd $IMAGENET_HOME # Move back to the base folder

      # [Optional] Delete tar files if desired as they are not needed
      rm $IMAGENET_HOME/train/*.tar

      # Download labels file.
      wget -O $IMAGENET_HOME/synset_labels.txt \
	      https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt

      # Process the files. Remember to get the script from github first. The TFRecords
      # will end up in the --local_scratch_dir. To upload to gcs with this method
      # leave off `nogcs_upload` and provide gcs flags for project and output_path.
      python3 /home/cc/monarch-cc/tensorflow/resources/imagenet/imagenet_to_gcs_v2.py \
	      --raw_data_dir=$IMAGENET_HOME \
	      --local_scratch_dir=$IMAGENET_HOME/tf_records \
	      --nogcs_upload
