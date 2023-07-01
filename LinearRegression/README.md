# Linear Regression
We make the assumption that a dataset has a linear pattern $\hat{y} = wx + b$.

To determine $w$ and $b$ we calculate the Mean Squared Error and look for $w$ and $b$ that would give us the minimum MSE according the following formula.
$MSE = J(w, b) = \frac{1}{N}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$ 
$y_i$ is the true value $\hat{y_i}$ is the predicted value.

To update the parameters we need to calculate the derivative of MSE (Gradient Descent)
$\frac{\partial f}{\partial w} = \partial w =\frac{1}{N}\sum_{i=1}^{n}{-2x_i(y_i - (wx_i + b))} = \frac{1}{N}\sum_{i=1}^{n}{-2x_i(y_i - \hat{y})} = \frac{1}{N}\sum_{i=1}^{n}{2x_i(\hat{y} - y_i)}$ 
$\frac{\partial f}{\partial b} = \partial b =\frac{1}{N}\sum_{i=1}^{n}{-2(y_i - (wx_i + b))} = \frac{1}{N}\sum_{i=1}^{n}{-2(y_i - \hat{y})} = \frac{1}{N}\sum_{i=1}^{n}{2(\hat{y} - y_i)}$ 
with gradient decent we calculate at a given parameter which direction to go to minimize the MSE.

$w=w - \alpha . \partial w$

$b=b - \alpha . \partial b$

## Training
 - Initialize weight as zero.
 - Initialize bias as zero.
 - Given a data point
   - Predict result by using $\hat{y} = wx + b$
   - Calculate error.
   - Use gradient descent to figure out new weight and bias values.
   - Repeat n times.