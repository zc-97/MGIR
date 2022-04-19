# MGIR
Ensure that all data set files have been unzipped.

- you can use the jupyter notebook in `MGIR\dataprocess\process\eval.ipynb`:

- for Tmall:

```python
%run getdata1.py --dataset='Tmall' --va=0
%run getSubGraph2.py --dataset='Tmall' --freq 10 --cold_ratio 0.85 --seq_ratio 1.0
%run getDegreeMatrix3.py --dataset='Tmall'
%run data4.py --dataset='Tmall'
%run  ../../Final/FinalModel/main.py --dataset='Tmall' --dropout_gcn=0.2 --dropout_local=0.2 --device='cuda:0'
```
- for RetailRocket:

```python
%run getdata1.py --dataset='retailrocket' --va=0
%run getSubGraph2.py --dataset='retailrocket' --freq 10 --cold_ratio 0.9 --seq_ratio 1.0 
%run getDegreeMatrix3.py --dataset='retailrocket'
%run data4.py --dataset='retailrocket'
%run  ../../Final/FinalModel/main.py --dataset='retailrocket' --dropout_gcn=0.1 --dropout_local=0.5 --l2=1e-6 --alpha=0.8 --device='cuda:0'
```
- for Last.fm

```python
%run getdata1.py --dataset='LastFM' --va=0
%run getSubGraph2.py --dataset='LastFM'  --freq 70 --cold_ratio 0.95 --seq_ratio 1.0 
%run getDegreeMatrix3.py --dataset='LastFM'
%run data4.py --dataset='LastFM'
%run ../../Final/FinalModel/main.py --dataset='LastFM' --dropout_gcn=0.2 --dropout_local=0.3 --l2=1e-6 --alpha=0.4 --batch_size=512 --device='cuda:0'
```







