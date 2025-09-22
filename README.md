# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 22-09-24



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv("C:\\Users\\admin\\Documents\\time series dataset\\archive (3).zip")

# Check columns
print("Columns in dataset:", data.columns)

# Declare required variables
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]  # plt.rcParams is a dictionary-like object in Matplotlib

# Use 'Price' column as time series (Y)
Y = data['Price']

# Plot original data
plt.plot(Y)
plt.title('Original House Price Data')
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.show()

# Plot ACF and PACF of Original Data
plt.subplot(2, 1, 1)
plot_acf(Y, lags=int(len(Y) / 4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(Y, lags=int(len(Y) / 4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()


# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(Y, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

print("ARMA(1,1) Parameters:")
print("phi1 =", phi1_arma11)
print("theta1 =", theta1_arma11)

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()


# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(Y, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

print("ARMA(2,2) Parameters:")
print("phi1 =", phi1_arma22)
print("phi2 =", phi2_arma22)
print("theta1 =", theta1_arma22)
print("theta2 =", theta2_arma22)

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt
```
#OUTPUT:
![WhatsApp Image 2025-09-22 at 15 45 58_9a34fae8](https://github.com/user-attachments/assets/bf4c5109-ab93-448d-ad02-25584644604d)
![WhatsApp Image 2025-09-22 at 15 46 35_9983078d](https://github.com/user-attachments/assets/3b6dde26-d652-4439-beaf-06142538aeb7)
![WhatsApp Image 2025-09-22 at 15 46 50_ebfbd1df](https://github.com/user-attachments/assets/771266b2-4ed0-4e3f-99cc-32770bcd6dd0)
![WhatsApp Image 2025-09-22 at 15 47 04_f84eefca](https://github.com/user-attachments/assets/856cf53a-4d14-47af-b684-422096b9e839)
![WhatsApp Image 2025-09-22 at 15 47 16_ac8a168f](https://github.com/user-attachments/assets/5fd2ad15-47b0-4f38-97be-905382e2037d)


![WhatsApp Image 2025-09-22 at 15 47 32_e98a4227](https://github.com/user-attachments/assets/e85a4235-ed57-4e5b-91d8-f6608f0b317f)






#RESULT:
Thus, a python program is created to fir ARMA Model successfully.
