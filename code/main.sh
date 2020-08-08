wget -O STM_weights.pth "https://www.dropbox.com/s/mtfxdr93xc3q55i/STM_weights.pth?dl=1"
mv STM_weights.pth ../user_data/model_data/

python eval.py -g 0 -s test -y 2017 -D ../data/
cd ../user_data/tmp_data/
zip -r summit.zip *
cp -r summit.zip  ../../prediction_result/
rm -rf *
