import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Dados de exemplo
data = pd.date_range(start='1/1/2022', end='1/31/2022')
values = np.random.randint(low=0, high=100, size=len(data))

# Criar um DataFrame com os dados
df = pd.DataFrame({'Date': data, 'Value': values})

# Converter a coluna Date para numérica
df['Date'] = pd.to_numeric(df['Date'])

# Adicionar a constante 1 para a regressão linear
df['Constant'] = 1

# Separar os dados de entrada (X) e saída (y)
X = df[['Date', 'Constant']]
y = df['Value']

# Criar o modelo de regressão linear
model = sm.OLS(y, X)

# Ajustar o modelo aos dados
results = model.fit()

# Extrair os coeficientes da regressão
coef = results.params['Date']
intercept = results.params['Constant']

# Calcular os valores previstos pela regressão
y_pred = coef * df['Date'] + intercept

# Plotar a série temporal original e a linha de regressão
plt.plot(data, values, label='Original')
plt.plot(data, y_pred, label='Regression')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series with Linear Regression')
plt.legend()
plt.show()

# Imprimir os resultados da regressão
print(results.summary())
