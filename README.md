# MultiVAE-for-MovieLens

boostcamp AI Tech 5기에서 진행된 "Movie Recommendation" 대회에서 빠른 실험을 위해 제작한 MultiVAE(MultiDAE) 모델의 PyTorch template 입니다.
해당 대회에서는 Task에 맞춰서 가공한 MovieLens 데이터를 사용했습니다. (Implicit feedback)

<br>

Weights & Biases를 활용하기 위해 **src/wandb.py**을 본인의 환경에 맞게 수정해야 합니다.  
Hyperparameter Tuning은 **src/args.py**를 참고합니다.

<br>

## Environment
`Python 3.8.5`  
`PyTorch 1.10.2`


<br>

## Training & Tuning

튜닝을 위해 데이터를 train/valid/test로 split 합니다.  
submission 파일은 생성하지 않습니다.

```bash
python train.py --model MultiVAE --mode tuning
```

```bash
python train.py --model MultiDAE --mode tuning
```

<br>

## Training & Inference

제출 파일 생성을 위해 Training에 전체 데이터를 사용합니다.  
각 유저별 아이템의 Top-10을 뽑아서 submission 파일을 생성합니다.

```bash
python train.py --model MultiVAE --mode submission
```

```bash
python train.py --model MultiDAE --mode submission
```
