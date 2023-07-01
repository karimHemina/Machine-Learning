# Logistic Regression
in logistic regression and in contrary to Linear Regression instead of trying  to get a specific value, we create probabilities. we do this by putting our value obtained from the equation $\hat{y} = wx + b$  into sigmoid function to get probability between 0 and 1.

sigmoid function is calculated as follows $s(x) = \frac{1}{1 + e^{-x}}$ we can also write $\hat{y}=h_0(x)=\frac{1}{1 + e^{-(wx+b)}}$ 

instead of using mean squared error as we do in Linear Regression, in logistic regression we use Cross Entropy.

$J(w,b) = J(\theta) = \frac{1}{N}\sum_{i=1}^{n}[y^i\log(h_0(x^i)) + (1 - y^i)\log(1 - h_0(x^i))]$ 

to update the parameters we need to calculate the derivative.

$\frac{\partial f}{\partial w} = ... = \frac{1}{N}\sum_{i=1}^{n}{2x_i(\hat{y} - y_i)}$ 

$\frac{\partial f}{\partial b} = ... = \frac{1}{N}\sum_{i=1}^{n}{2(\hat{y} - y_i)}$ 

## Training
 - Initialize weight as zero.
 - Initialize bias as zero.
 - Given a data point
   - Predict result by using  $\hat{y} = \frac{1}{1 + e^{-(wx+b)}}$ 
   - Calculate error.
   - Use gradient descent to figure out new weight and bias values.
   - Repeat n times.

## Testing
- Given a data point
	- Put in the values from the data point into the equation $\hat{y} = \frac{1}{1 + e^{-(wx+b)}}$ 
	- Choose the label based on the probability