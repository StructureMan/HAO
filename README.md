# HAO
Hyperbolic Adaptive Spatial Awareness for Multivariate Time Series Anomaly Detection
# How to run?
1. Set up the HAO operating environment.
2. Download the required dataset.
3. run cmd : python main.py --dataset ASD --model HAO --windowsize 5 --epoch 1 --space Euclidea
  | 0/1 [00:00<?, ?it/s]
  Training HAO on ASD
  Epoch 1,        MSE = 0.09183857994980507
  Epoch 1,        MSE = 0.06797662206278485
  Epoch 1,        MSE = 0.06071583362423921
  Epoch 1,        MSE = 0.0574725832384837
  Epoch 1,        MSE = 0.05532288017284749
  Epoch 1,        MSE = 0.05369831801269238
    0%| |0/1 [00:30<?, ?it/s]
  Testing HAO on ASD"
  {'precision': 1.0, 'recall': 0.9424657534246575, 'f1': 0.9703808180535965, 'accuracy': 0.9950225171841669, 'roc_auc': 0.9712328767123288}
