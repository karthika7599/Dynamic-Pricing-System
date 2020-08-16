# Dynamic-Pricing-System
An AI model to predict the price of a commodity dynamically based on expiry date, inventory level, sales and profit margin.

The dynamic pricing system model predicts price with respect to 4 factors namely Expiry Date, Profit gain, Inventory level and Sales History.
-As the expiry date of the item becomes closer the price of the item is to be decreased.
-The profit margin is to be set according to the shopkeeper. This is a constant value below which the price of the item can never go.
-Inventory level is the amount of goods that’s left with us. So if the number of items remaining is high we need to decrease the amount of the item.
-If the sales of the previous day is high the product price is to be increased.

Here linear regression model is mixed with polynomial regression to get accurate results using a pipeline as here we have 4 factors and the points are scattered in the space. 


