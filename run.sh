python preprocess.py --mode train --behaviors_path ./train_behaviors.tsv --news_path ./train_news.tsv --outputs ./result_for_train.csv

python preprocess.py --mode test --behaviors_path ./test_behaviors.tsv --news_path ./test_news.tsv --outputs ./result_for_test.csv

python train.py --mode train --train_path ./result_for_train.csv

python train.py --mode test --test_path ./result_for_test.csv --outputs ./test_output.csv

