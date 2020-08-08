python eval_less_memory.py -g 0 -s test -y 2017 -D ../data/
cd ../user_data/tmp_data/
zip -r summit.zip *
cp -r summit.zip  ../../prediction_result/
rm -rf *
