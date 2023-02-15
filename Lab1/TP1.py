# import des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# récupération des données et lecture du fichier
name_file = './PreLab1/data_lab1.txt'
columns = ['x', 'y']
data_in = pd.read_csv(name_file, names=columns, sep=' ')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])

# création du graphique
plt.figure()
# titre du graphique
plt.title('Graphique de la sortie du fichier data_lab1.txt')
plt.plot(x, y, 'ro')
# nom des axes
plt.xlabel('x')
plt.ylabel('y')
# affichage du graphique
plt.show()

# Question 2 Division of the datas
#first 70% of total data
X_train = x[0:70]
Y_train = y[0:70]

#last 30% of total data
X_test = x[70:100]
Y_test = y[70:100]

plt.plot(X_train, Y_train, 'ro', label='Training data')
plt.plot(X_test, Y_test, 'bo',  label='Test data')
plt.legend()
plt.show()

# Question 3 - BGD
# initialisation des paramètres
alpha = pow(10, -3)
theta = 0
itera = 0
threshold = 10
error = Verror = pow(10, 3)

# boucle de descente de gradient
while Verror[itera] > threshold:
    itera += 1
    for i in range(0, 1):
        theta[itera] = theta[itera-1] - learning_rate * \
            sum((np.transpose(X_train[i])*theta-Y_train[i])*X_train[i])

    error[itera] = (1/2)*sum(pow(Y_test[i] - Y_train[i], 2))
    Verror[itera] = error[itera] - error[itera-1]
    error[itera - 1] = error[itera]
    np.reshape()
