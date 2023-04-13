import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#predict the CO2 emission where the weight is 2300g, and the volume is 1300ccm:
predictedemmision = regr.predict([[2300, 1300]])

print("Please ignore the warning given above")
print()
print()


print("The predicted emission of a car whose volume is 1300 cm cubed and weight is 2300g")

print(predictedemmision)
'''
print(regr.coef_)
print(type(regr.coef_))
print(regr.coef_[0])
'''
print("if the weight of the engine increases by 1 kg, then the co2 emission increases by", regr.coef_, "grams")
print("if the volume of the engine increases by 1cm cube, then the co2 emission increases by", regr.coef_[1], "grams")