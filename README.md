# RobuFlight

To run the code:

```
python aquila/scripts/train_trackVer5.py

tensorboard --logdir=/home/core/wangzimo/Aquila/runs --port=6012
python aquila/scripts/test_trackVer5.py

python aquila/scripts/test_trackVer5_iris_onboard.py
python aquila/scripts/test_trackVer5_iris_comparing.py

tensorboard --logdir=/home/zim/VTT/Aquila/aquila/test_runs --port=6012
```