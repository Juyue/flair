
echo 'Filename,Start,End,Type,Score,Surface' > solution_train.csv
python3 merge_result_lines.py train_output.csv | grep -E 'PER|LOC|ORG|MIS' >> solution_train.csv
python scorer.py /home/jupyter/data_ner/truth.csv solution_train.csv ''

echo 'Filename,Start,End,Type,Score,Surface' > solution.csv
python3 merge_result_lines.py dev_output.csv | grep -E 'PER|LOC|ORG|MIS' >> solution.csv

cp solution.csv /home/jupyter/submission_file/solution/solution.csv

