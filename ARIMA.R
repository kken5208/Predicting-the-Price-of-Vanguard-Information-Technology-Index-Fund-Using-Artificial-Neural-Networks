
library(data.table)
library(ggplot2)
library(xtable)
library(forecast)
library(strucchange)
library(tseries)
library(TSstudio)
library(lubridate)
# library(tsoutliers)

### Seasonal Data in Zhang and Qi (2005) ###

data.folder = "/Users/kken5/桌面/論文/資料/"
data.file = "vgt.csv"
file.header = TRUE
log_transform = TRUE

data.file = paste(data.folder, data.file, sep="")
df_ = read.csv(data.file, header=file.header, stringsAsFactors = TRUE)
colnames(df_)=column.names
ts_ = ts(data = df_$Price)
# ts_ = ts(data = df_$x, frequency = data.frequency, start = data.start)
if (log_transform) {
  ts_ = log(ts_)
}
autoplot(ts_)


# lt = tslm(urate_ts ~ trend)
# urate_ts = lt$residuals
# autoplot(urate_ts)
# new.data.file = paste(data.folder, "zhangqi_data10_resid.csv", sep="")
# write.csv(urate_ts, new.data.file)


# du = diff(urate_ts, lag=12, differences=1)
# autoplot(du)
# urate_ts <- du - mean(du)

ts_ = diff(ts_)

ts_len = 4607
train_size = round(ts_len * 0.8)
test_size = ts_len - train_size


train_r_ts = ts(data=ts_[1:train_size])
# # auto arima
train_fit = auto.arima(train_r_ts, stepwise=TRUE)
summary(train_fit)

# fit.arima = Arima(urate_ts, order=c(model.p, model.d, 0),
#                   seasonal=list(order=c(model.P, model.D, 0),
#                                 period=model.period))



f_ahead = 1
forecast = rep(0, test_size)
for (c in 1:test_size){
  sample_ts = ts(data=ts_[1:(train_size+c-f_ahead)])
  sample.fit = Arima(sample_ts, model=train_fit)   # without re-estimation
  forecast[c] = predict(sample.fit, n.ahead=f_ahead)$pred[f_ahead]
}

test_target = ts_[(train_size+1):(train_size+test_size)]
RMSE = sqrt(sum((forecast - test_target)^2)/test_size)
