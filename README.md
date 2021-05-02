# bert_for_multitasking


## Command to train bert as a sequence classifier:

#### python bert_for_multitask.py --model_name_or_path bert-base-uncased --train_file /path/to/data/train.tsv --test_file /path/to/data/test.tsv --validation_file /path/to/data/valid.tsv --max_length 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 30 --output_dir /path/to/output/dir --seed 42

## Command to train bert as a sequence classifier and regression task:

#### python bert_for_multitask.py --model_name_or_path bert-base-uncased --va REGRESSION_OUTPUT_SIZE --train_file /path/to/data/train.tsv --test_file /path/to/data/test.tsv --validation_file /path/to/data/valid.tsv --max_length 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 30 --output_dir /path/to/output/dir --seed 42
