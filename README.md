
<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Auto Trading</h3>

  <p align="center">
    DSAI HW2
    <br />
    <a href="https://github.com/Shulammiteya/Auto-Trading"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.7.13

### Usage

1. Install packages
   ```sh
   pip install -r requirements.txt
   ```
2. Run trader.py （To train the model, uncomment line 119）
   ```JS
   python trader.py --training training.csv --testing testing.csv --output output.csv
   ```
<br />


## Data Scaling

* 使用 sklearn 的 MinMaxScaler
   ```JS
   
   from sklearn.preprocessing import MinMaxScaler
   
   scaler = MinMaxScaler(feature_range=(0, 1))
    
   ```


## Model Training

* 使用 LSTM 模型進行多步預測，利用前 10 天的股票資訊，輸出未來 2 天的開盤價預測。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1m6O477NjdQRENFTbX6RiYKL3S-F0Nl9R" alt="model structure">
</p>

* 模型訓練資訊。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1s27Bj9yD0b3LA6_OdspvTxaak9QxzKJv" alt="training history">
</p>
<br />


## Model Validation

* 圖為 test dataset 的 Ground Truth 與模型預測值（train：validate：test 比例為 0.76：0.19：0.05）。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1pjbBkBHSv-T2kFzmvhkYZ1-RcH2t4wLK" alt="model validation">
</p>
<br />


## Trading Policy
* 若明天的預測開盤價 < 後天預測開盤價，則代表明天為低點，因此決定買入；反之則賣出。
<br />


<!-- CONTACT -->
## Contact

About me: [Hsin-Hsin, Chen](https://www.facebook.com/profile.php?id=100004017297228) - shulammite302332@gmail.com
