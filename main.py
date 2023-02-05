"""
Projeto Integrador Python 2022.1.
Projeto Final
Grupos de até 1 ou 2 integrantes. Data de entrega: 01/06/2022 23:59

Entrega do programa via compartilhamento do notebook do Google Colab
para o email pjasimoes@gmail.com

Coloque nome, TIA e semestre dos integrantes no topo do notebook
usando bloco de comentário.

Objetivos
1. Leitura dos dados do giroscópio https://raw.githubusercontent.com/pjasimoes/PIPythonData/main/giroscopio1.csv
2. Produzir gráficos das velocidades angulares das 3 coordenadas
3. Encontrar quantos graus de giro o sensor detectou em cada eixo
4. Obter a aceleração angular em função do tempo de cada coordenada e mostrar seus gráficos
5. Encontrar e mostrar em gráficos a variação da posição angular em função do tempo
6. Responda:
    6a. qual foi a ordem de movimento dos eixos?
    6b. qual foi a maior velocidade angular e em qual eixo?
    6c. qual foi o maior deslocamento angular (em graus) e em qual eixo?
    6d. qual eixo teve a maior aceleração angular?

Dicas:
    a) Remova o ruído dos sinais; ver pandas.dataframe.rolling() ou outro filtro
    b) Remova o erro sistemático dos sinais: a média em cada coordenada para t<10s deve ser zero
    c) Para fazer os gráficos, use plt.subplots
    d) para o item (3), faça a integral numérica via scipy.integrate.trapz()
    e) para o item (4), faça a derivada numérica via numpy.gradient()
    f) para o item (5), faça a integral numérica via scipy.integrate.cumtrapz()

"""


## módulos necessários
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.integrate import cumtrapz

## ordem do movimento: z=90,x=45,y=90

## leitura do arquivo csv de dados do giroscópio
url = 'https://raw.githubusercontent.com/pjasimoes/PIPythonData/main/giroscopio1.csv'
df = pd.read_csv(url)
keys = df.keys()

## verificação do tempo de leitura
dt = np.round((df['Time (s)'] - df['Time (s)'].shift(1)).median(), 3)
# fig=plt.figure()
# print('tempo de leitura = ',dt)
# plt.plot(df['Time (s)'],df['Time (s)'] - df['Time (s)'].shift(1),'.')
# plt.plot([df['Time (s)'].iloc[0],df['Time (s)'].iloc[-1]],[dt]*2)

## definição da janela de tempo para filtro passa-baixa para tirar ruído
## de alta frequência 
tint = 1  ## 1 seg
n = int(tint / dt)
t = df['Time (s)']

## correção de offset (erro sistemático)
meanx = df['Gyroscope x (rad/s)'].iloc[np.where(t < 8)].mean()
meany = df['Gyroscope y (rad/s)'].iloc[np.where(t < 8)].mean()
meanz = df['Gyroscope z (rad/s)'].iloc[np.where(t < 8)].mean()
meant = df['Absolute (rad/s)'].iloc[np.where(t < 8)].mean()

df['Gyroscope x (rad/s)'] -= meanx
df['Gyroscope y (rad/s)'] -= meany
df['Gyroscope z (rad/s)'] -= meanz
df['Absolute (rad/s)'] -= meant

## remoção de ruído de alta frequência
wx = df['Gyroscope x (rad/s)'].rolling(n, min_periods=0, center=True).mean()
wy = df['Gyroscope y (rad/s)'].rolling(n, min_periods=0, center=True).mean()
wz = df['Gyroscope z (rad/s)'].rolling(n, min_periods=0, center=True).mean()

## cálculo do deslocamento angular total em cada coordenada
x = np.rad2deg(trapz(wx, t))
y = np.rad2deg(trapz(wy, t))
z = np.rad2deg(trapz(wz, t))

## cálculo da posição angular em função do tempo
px = np.rad2deg(cumtrapz(wx, t))
py = np.rad2deg(cumtrapz(wy, t))
pz = np.rad2deg(cumtrapz(wz, t))

### cálculo da aceleração angular
acx = np.gradient(wx, t)
acy = np.gradient(wy, t)
acz = np.gradient(wz, t)

## gráficos
fig, ax = plt.subplots(3, 3, figsize=(12, 8))

## dados do giroscópio nos 3 eixos
## velocidade angular
ax[0][0].plot(t, df['Gyroscope x (rad/s)'])
ax[0][0].plot(t, wx)

ax[0][1].plot(t, df['Gyroscope y (rad/s)'])

ax[0][1].plot(t, wy)

ax[0][2].plot(t, df['Gyroscope z (rad/s)'])

ax[0][2].plot(t, wz)

## posição angular
ax[1][0].plot(t[1::], px)
ax[1][1].plot(t[1::], py)
ax[1][2].plot(t[1::], pz)

## aceleração angular
ax[2][0].plot(t, acx)
ax[2][1].plot(t, acy)
ax[2][2].plot(t, acz)

## prepação dos eixos dos gráficos
####################################

s = ax.shape
for i in range(s[0]):
    for j in range(s[1]):
        ax[i][j].set_xlabel(keys[0])
        ax[i][j].grid()
        if i == 1:
            ax[i][j].set_ylim(-10, 110)
        if i == 0:
            ax[i][j].plot(t, df['Absolute (rad/s)'], alpha=0.3, color='black')

ax[0][0].set_ylabel(keys[1])
ax[0][1].set_ylabel(keys[2])
ax[0][2].set_ylabel(keys[3])

ax[1][0].set_ylabel('x (deg)')
ax[1][1].set_ylabel('y (deg)')
ax[1][2].set_ylabel('z (deg)')

ax[2][0].set_ylabel('ax (rad/s^2)')
ax[2][1].set_ylabel('ay (rad/s^2)')
ax[2][2].set_ylabel('az (rad/s^2')

plt.tight_layout()

eixos = ['x', 'y', 'z']
## velocidades máximas
w = np.array([wx.max(), wy.max(), wz.max()])
## acelerações máximas
ac = np.array([acx.max(), acy.max(), acz.max()])

plt.show()

print('=====================================================================')
print('Movimento dos eixos: ')
print('z = ' + str(int(z)) + ' graus')
print('x = ' + str(int(x)) + ' graus')
print('y = ' + str(int(y)) + ' graus (maior deslocamento)')
print('')
print('Maior velocidade angular:')
print('Eixo ' + eixos[w.argmax()] + ': w = ' + str(np.round(w.max(), 1)) + ' rad/s')
print('')
print('Maior aceleração angular:')
print('Eixo ' + eixos[ac.argmax()] + ': w = ' + str(np.round(ac.max(), 1)) + ' rad/s^2')
print('')

# 6a. qual foi a ordem de movimento dos eixos?
#     6b. qual foi a maior velocidade angular e em qual eixo?
#     6c. qual foi o maior deslocamento angular (em graus) e em qual eixo?
#     6d. qual eixo teve a maior aceleração angular?