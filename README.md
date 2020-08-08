# 采取的方案： Video Object Segmentation using Space-Time Memory Networks  
pdf： https://arxiv.org/abs/1904.00607
github： https://github.com/seoungwugoh/STM (参考的代码)
通过时空内存网络来获取帧间的信息， 当前帧可以利用之前存储的信息来补充分割的结果，从而得到一个更加精确的分割结果。
# 目前提供的方案用的是ＳＴＭ官方的开源方案，　为了适应当前的人像细致分割的数据集，我也将STM的pretrained模型加载到网络中，利用提供的人像数据进行finetune。　
# 由于本次比赛分为初赛semi-supervised 和复赛unsupervised的流程，　我简单评估了目前进入复赛的条件，　
# 只需要提供STM pretrained model　进行eval就可以取得排行榜上的42名的结果 /0.876/0.86/0.93/0.08/0.89/0.94/0.08，就能稳当的进入复赛。
# 因此，我提交了两次结果在semi-supervised 的初赛上，就把精力用在后续的复赛的unsupervised的赛程中。如前所述,我调研了unsupervised方案, 目前可能比较好的方案是combine 单帧的实例分割网络和STM模型,整个网络特别重,系统看起来特别的繁杂,不elegant,
# 我通过分析并做试验提出一个简洁而高效的unsurpervised框架, 具体的情况复赛再见分晓.
# 因此此次提交的方案目前只提供semi-supervised的可以线上测试和训练代码的代码，训练代码是直接使用STM的模型来进行finetune,没有做过多的尝试,详情如下:
#

# environment
python 3.8, ubuntu16, pytorch1.5, CUDA Version 10.2.89, numpy, opencv, pillow, 
缺少什么lib，就pip install 一下。


# test
# 之前提交官网时实在GPU V100 32G上跑的， 可以完整的复现比赛结果：42 /0.876/0.86/0.93/0.08/0.89/0.94/0.08， 执行的程序如下所示
cd code
python eval.py -g 0 -s test -y 2017 -D ../data/
cd ../user_data/tmp_data/
zip -r summit.zip *
cp -r summit.zip  ../../prediction_result/
rm -rf *
即直接运行code目录下的 ./main.sh
# 如果只能在titan xp 12G上跑， 可以运行以下的代码,目前在main.sh 上提供的可能会跟线上结果有所差距：
cd code
python eval_less_memory.py -g 0 -s test -y 2017 -D ../data/
cd ../user_data/tmp_data/
zip -r summit.zip *
cp -r summit.zip  ../../prediction_result/
rm -rf *
即直接运行code目录下的 ./main_less_memory.sh

# train
# STM 官方开源没有提供数据的组织和训练代码, 自己复现了一把
python train.py -g 0 -s train -y 2017 --loadepoch 0


# 个人联系方式：
wechat & phone: 13032893650 
email: 2755289083@qq.com  
如果有任何验证的问题，欢迎致电与我联系，　谢谢。



