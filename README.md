# Temporal_Fusion_Transformer(PyTorch)
\# Time Series   
\# Multi-horizon Time Series Forecasting
# 현황
Google research 깃허브의 tf v1 TFT 코드를 pytorch로 구현, train, inference까지 동작 확인 완료.

NLP 외 분야에서 임베딩, 어텐션을 포함한 트랜스포머 구조가 어떻게 사용되는지 확인하기 위해 도전해보았지만 시계열, Multi-horizon Time Series Forecasting Task는 현재 익숙하지가 않아 타 데이터셋에 적용, 기존 머신러닝 모델과의 성능 비교 등은 추후 작업 예정.

# Folder tree
./   
 ┣ data_formatters   
 ┃ ┣ base.py   
 ┃ ┣ electricity.py   
 ┃ ┗ volatility.py   
 ┣ utils   
 ┃ ┣ data_downloader.py   
 ┃ ┣ hyperparam_opt.py   
 ┃ ┣ loss.py   
 ┃ ┗ utils.py   
 ┣ LICENSE   
 ┣ model.py   
 ┣ README.md   
 ┗ tft_test.ipynb   


# Paper
https://arxiv.org/pdf/1912.09363v3.pdf

# Usage example
tft_test.ipynb

# Reference
https://paperswithcode.com/paper/temporal-fusion-transformers-for

https://github.com/google-research/google-research/tree/master/tft (Official)

https://github.com/jdb78/pytorch-forecasting

https://github.com/mattsherar/Temporal_Fusion_Transform
