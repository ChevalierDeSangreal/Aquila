# RobuFlight

To run the code:

```
python aquila/scripts/train_trackVer5.py
python aquila/scripts/train_trackVer9.py
python aquila/scripts/train_trackVer10.py

nohup python aquila/scripts/train_trackVer9.py &

python aquila/scripts/train_hoverVer0.py
nohup python aquila/scripts/train_hoverVer0.py &

tensorboard --logdir=/home/core/wangzimo/Aquila/runs --port=6012
python aquila/scripts/test_trackVer5.py
python aquila/scripts/test_trackVer9.py
python aquila/scripts/test_hoverVer0.py

python aquila/scripts/test_trackVer5_iris_onboard.py
python aquila/scripts/test_trackVer5_iris_comparing.py

python aquila/scripts/convert_to_tflite.py

tensorboard --logdir=/home/zim/VTT/Aquila/aquila/test_runs --port=6012
```

```
pip install tensorflow==2.17.1
```