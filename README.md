# Prototypical Networks for Few-shot Learning
Updated implementation of ProtoNet. Using this code, you will get **80.0%** acc on MiniImagenet 5-shot task with ResNet12. Because someone only achieved 72% acc, this repo will be very helpful to those guys. The key ingredient of achieving such high acc is using SGD+StepLR.

## create the dataset  
plz refer to [CloserLookAtFewShotLearning](https://github.com/wyharveychen/CloserLookFewShot)
## train the Baseline
python train.py --method_type baseline --name baseline
## train the ProtoNet
python train.py --method_type protonet --name baseline --warmup output/baseline_baseline_ResNet12_5shot
## test
python train.py --method_type protonet --name baseline --test
