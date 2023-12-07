# islp

# Support Vector Machines

## Conceptual

### Question 1

> This problem involves hyperplanes in two dimensions.
>
> a. Sketch the hyperplane $1 + 3X_1 − X_2 = 0$. Indicate the set of points for
>    which $1 + 3X_1 − X_2 > 0$, as well as the set of points for which
>    $1 + 3X_1 − X_2 < 0$.

```{r}
library(ggplot2)
xlim <- c(-10, 10)
ylim <- c(-30, 30)
points <- expand.grid(
  X1 = seq(xlim[1], xlim[2], length.out = 50), 
  X2 = seq(ylim[1], ylim[2], length.out = 50)
)
p <- ggplot(points, aes(x = X1, y = X2)) + 
  geom_abline(intercept = 1, slope = 3) +  # X2 = 1 + 3X1 
  theme_bw()
p + geom_point(aes(color = 1 + 3*X1 - X2 > 0), size = 0.1) + 
  scale_color_discrete(name = "1 + 3X1 − X2 > 0")
```

> b. On the same plot, sketch the hyperplane $−2 + X_1 + 2X_2 = 0$. Indicate the
>    set of points for which $−2 + X_1 + 2X_2 > 0$, as well as the set of points
>    for which $−2 + X_1 + 2X_2 < 0$.

```{r}
p + geom_abline(intercept = 1, slope = -1/2) +  # X2 = 1 - X1/2
  geom_point(
    aes(color = interaction(1 + 3*X1 - X2 > 0, -2 + X1 + 2*X2 > 0)), 
    size = 0.5
  ) + 
  scale_color_discrete(name = "(1 + 3X1 − X2 > 0).(−2 + X1 + 2X2 > 0)")
```

### Question 2

> We have seen that in $p = 2$ dimensions, a linear decision boundary takes the
> form $\beta_0 + \beta_1X_1 + \beta_2X_2 = 0$. We now investigate a non-linear
> decision boundary.
>
> a. Sketch the curve $$(1+X_1)^2 +(2−X_2)^2 = 4$$.

```{r}
points <- expand.grid(
  X1 = seq(-4, 2, length.out = 100), 
  X2 = seq(-1, 5, length.out = 100)
)
p <- ggplot(points, aes(x = X1, y = X2, z = (1 + X1)^2 + (2 - X2)^2 - 4)) + 
  geom_contour(breaks = 0, colour = "black") + 
  theme_bw()
p
```

> b. On your sketch, indicate the set of points for which
>    $$(1 + X_1)^2 + (2 − X_2)^2 > 4,$$ as well as the set of points for which
>    $$(1 + X_1)^2 + (2 − X_2)^2 \leq 4.$$

```{r}
p + geom_point(aes(color = (1 + X1)^2 + (2 - X2)^2 - 4 > 0), size = 0.1)
```

> c. Suppose that a classifier assigns an observation to the blue class if $$(1
>    + X_1)^2 + (2 − X_2)^2 > 4,$$ and to the red class otherwise. To what class
>    is the observation $(0, 0)$ classified? $(−1, 1)$? $(2, 2)$? $(3, 8)$?

```{r}
points <- data.frame(
  X1 = c(0, -1, 2, 3),
  X2 = c(0, 1, 2, 8)
)
ifelse((1 + points$X1)^2 + (2 - points$X2)^2 > 4, "blue", "red")
```

> d. Argue that while the decision boundary in (c) is not linear in terms of
>    $X_1$ and $X_2$, it is linear in terms of $X_1$, $X_1^2$, $X_2$, and
>    $X_2^2$.

The decision boundary is $$(1 + X_1)^2 + (2 − X_2)^2 -4 = 0$$ which we can expand 
to:
$$1 + 2X_1 + X_1^2 + 4 − 4X_2 + X_2^2 - 4 = 0$$
which is linear in terms of $X_1$, $X_1^2$, $X_2$, $X_2^2$.

### Question 3

> Here we explore the maximal margin classifier on a toy data set.
>
> a. We are given $n = 7$ observations in $p = 2$ dimensions. For each
>    observation, there is an associated class label.
>    
>    | Obs. | $X_1$ | $X_2$ | $Y$  |
>    |------|-------|-------|------|
>    | 1    | 3     | 4     | Red  |
>    | 2    | 2     | 2     | Red  |
>    | 3    | 4     | 4     | Red  |
>    | 4    | 1     | 4     | Red  |
>    | 5    | 2     | 1     | Blue |
>    | 6    | 4     | 3     | Blue |
>    | 7    | 4     | 1     | Blue |
>    
>    Sketch the observations.

```{r}
data <- data.frame(
  X1 = c(3, 2, 4, 1, 2, 4, 4),
  X2 = c(4, 2, 4, 4, 1, 3, 1),
  Y  = c(rep("Red", 4), rep("Blue", 3))
)
p <- ggplot(data, aes(x = X1, y = X2, color = Y)) + 
  geom_point(size = 2) + 
  scale_colour_identity() +
  coord_cartesian(xlim = c(0.5, 4.5), ylim = c(0.5, 4.5))
p
```

> b. Sketch the optimal separating hyperplane, and provide the equation for this
>    hyperplane (of the form (9.1)).

```{r}
library(e1071)

fit <- svm(as.factor(Y) ~ ., data = data, kernel = "linear", cost = 10, scale = FALSE)

# Extract beta_0, beta_1, beta_2
beta <- c(
  -fit$rho,
  drop(t(fit$coefs) %*% as.matrix(data[fit$index, 1:2]))
)
names(beta) <- c("B0", "B1", "B2")
p <- p + geom_abline(intercept = -beta[1] / beta[3], slope = -beta[2] / beta[3], lty = 2)
p
```

> c. Describe the classification rule for the maximal margin classifier. It
>    should be something along the lines of "Classify to Red if $\beta_0 +
>    \beta_1X_1 + \beta_2X_2 > 0$, and classify to Blue otherwise." Provide the
>    values for $\beta_0, \beta_1,$ and $\beta_2$.

Classify to red if $\beta_0 + \beta_1X_1 + \beta_2X_2 > 0$ and blue otherwise
where $\beta_0 = `r round(beta[1])`$,  $\beta_1 = `r round(beta[2])`$,
$\beta_2 = `r round(beta[3])`$.

> d. On your sketch, indicate the margin for the maximal margin hyperplane.

```{r}
p <- p + geom_ribbon(
  aes(x = x, ymin = ymin, ymax = ymax),
  data = data.frame(x = c(0, 5), ymin = c(-1, 4), ymax = c(0, 5)),
  alpha = 0.1, fill = "blue",
  inherit.aes = FALSE
)
p
```

> e. Indicate the support vectors for the maximal margin classifier.

```{r}
p <- p + geom_point(data = data[fit$index, ], size = 4)
p
```

The support vectors (from the svm fit object) are shown above. Arguably, 
there's another support vector, since four points exactly touch the margin.

> f. Argue that a slight movement of the seventh observation would not affect
>    the maximal margin hyperplane.

```{r}
p + geom_point(data = data[7, , drop = FALSE], size = 4, color = "purple")
```

The 7th point is shown in purple above. It is not a support vector, and not 
close to the margin, so small changes in its X1, X2 values would not affect the
current calculated margin. 

> g. Sketch a hyperplane that is _not_ the optimal separating hyperplane, and
>    provide the equation for this hyperplane.

A non-optimal hyperline that still separates the blue and red points would 
be one that touches the (red) point at X1 = 2, X2 = 2 and the (blue) point at
X1 = 4, X2 = 3. This gives line $y = x/2 + 1$  or, when  $\beta_0 = -1$, 
$\beta_1 = -1/2$, $\beta_2 = 1$.

```{r}
p + geom_abline(intercept = 1, slope = 0.5, lty = 2, col = "red")
```

> h. Draw an additional observation on the plot so that the two classes are no
>    longer separable by a hyperplane.

```{r}
p + geom_point(data = data.frame(X1 = 1, X2 = 3, Y  = "Blue"), shape = 15, size = 4)
```

## Applied

### Question 4

> Generate a simulated two-class data set with 100 observations and two features
> in which there is a visible but non-linear separation between the two classes.
> Show that in this setting, a support vector machine with a polynomial kernel
> (with degree greater than 1) or a radial kernel will outperform a support
> vector classifier on the training data. Which technique performs best on the
> test data? Make plots and report training and test error rates in order to
> back up your assertions.

```{r}
set.seed(10)
data <- data.frame(
  x = runif(100),
  y = runif(100)
)
score <- (2*data$x-0.5)^2 + (data$y)^2 - 0.5
data$class <- factor(ifelse(score > 0, "red", "blue"))

p <- ggplot(data, aes(x = x, y = y, color = class)) + 
  geom_point(size = 2) + scale_colour_identity()
p

train <- 1:50
test <- 51:100

fits <- list(
  "Radial" = svm(class ~ ., data = data[train, ], kernel = "radial"),
  "Polynomial" = svm(class ~ ., data = data[train, ], kernel = "polynomial", degree = 2),
  "Linear" = svm(class ~ ., data = data[train, ], kernel = "linear")
)

err <- function(model, data) {
  out <- table(predict(model, data), data$class)
  (out[1, 2] + out[2, 1]) / sum(out)
}
plot(fits[[1]], data)
plot(fits[[2]], data)
plot(fits[[3]], data)
sapply(fits, err, data = data[train, ])
sapply(fits, err, data = data[test, ])
```

In this case, the radial kernel performs best, followed by a linear kernel with
the 2nd degree polynomial performing worst. The ordering of these models is the
same for the training and test data sets.

### Question 5

> We have seen that we can fit an SVM with a non-linear kernel in order to
> perform classification using a non-linear decision boundary. We will now see
> that we can also obtain a non-linear decision boundary by performing logistic
> regression using non-linear transformations of the features.
>
> a. Generate a data set with $n = 500$ and $p = 2$, such that the observations
>    belong to two classes with a quadratic decision boundary between them. For
>    instance, you can do this as follows:
>    
>    ```r
>    > x1 <- runif(500) - 0.5
>    > x2 <- runif(500) - 0.5
>    > y <- 1 * (x1^2 - x2^2 > 0)
>    ```

```{r}
set.seed(42)
train <- data.frame(
  x1 = runif(500) - 0.5,
  x2 = runif(500) - 0.5
)
train$y <- factor(as.numeric((train$x1^2 - train$x2^2 > 0)))
```

> b. Plot the observations, colored according to their class labels. Your plot
>    should display $X_1$ on the $x$-axis, and $X_2$ on the $y$-axis.

```{r}
p <- ggplot(train, aes(x = x1, y = x2, color = y)) + 
  geom_point(size = 2)
p
```

> c. Fit a logistic regression model to the data, using $X_1$ and $X_2$ as 
>    predictors.

```{r}
fit1 <- glm(y ~ ., data = train, family = "binomial")
```

> d. Apply this model to the _training data_ in order to obtain a predicted class
>    label for each training observation. Plot the observations, colored
>    according to the _predicted_ class labels. The decision boundary should be
>    linear.

```{r}
plot_model <- function(fit) {
  if (inherits(fit, "svm")) {
    train$p <- predict(fit)
  } else {
    train$p <- factor(as.numeric(predict(fit) > 0))
  }
  ggplot(train, aes(x = x1, y = x2, color = p)) + 
    geom_point(size = 2)
}

plot_model(fit1)
```

> e. Now fit a logistic regression model to the data using non-linear functions
>    of $X_1$ and $X_2$ as predictors (e.g. $X_1^2, X_1 \times X_2, \log(X_2),$
>    and so forth).

```{r}
fit2 <- glm(y ~ poly(x1, 2) + poly(x2, 2), data = train, family = "binomial")
```

> f. Apply this model to the _training data_ in order to obtain a predicted
>    class label for each training observation. Plot the observations, colored
>    according to the _predicted_ class labels. The decision boundary should be
>    obviously non-linear. If it is not, then repeat (a)-(e) until you come up
>    with an example in which the predicted class labels are obviously
>    non-linear.

```{r}
plot_model(fit2)
```

> g. Fit a support vector classifier to the data with $X_1$ and $X_2$ as
>    predictors. Obtain a class prediction for each training observation. Plot
>    the observations, colored according to the _predicted class labels_.

```{r}
fit3 <- svm(y ~ x1 + x2, data = train, kernel = "linear")
plot_model(fit3)
```

> h. Fit a SVM using a non-linear kernel to the data. Obtain a class prediction
>    for each training observation. Plot the observations, colored according to
>    the _predicted class labels_.

```{r} 
fit4 <- svm(y ~ x1 + x2, data = train, kernel = "polynomial", degree = 2)
plot_model(fit4)
```

> i. Comment on your results.

When simulating data with a quadratic decision boundary, a logistic model with
quadratic transformations of the variables and an svm model with a quadratic
kernel both produce much better (and similar fits) than standard linear methods.

### Question 6

> At the end of Section 9.6.1, it is claimed that in the case of data that is
> just barely linearly separable, a support vector classifier with a small
> value of `cost` that misclassifies a couple of training observations may
> perform better on test data than one with a huge value of `cost` that does not
> misclassify any training observations. You will now investigate this claim.
>
> a. Generate two-class data with $p = 2$ in such a way that the classes are
>    just barely linearly separable.

```{r}
set.seed(2)

# Simulate data that is separable by a line at y = 2.5
data <- data.frame(
  x = rnorm(200),
  class = sample(c("red", "blue"), 200, replace = TRUE)
)
data$y <- (data$class == "red") * 5 + rnorm(200)

# Add barley separable points (these are simulated "noise" values)
newdata <- data.frame(x = rnorm(30))
newdata$y <- 1.5*newdata$x + 3 + rnorm(30, 0, 1)
newdata$class = ifelse((1.5*newdata$x + 3) - newdata$y > 0, "blue", "red")

data <- rbind(data, newdata)

# remove any that cause misclassification leaving data that is barley linearly
# separable, but along an axis that is not y = 2.5 (which would be correct
# for the "true" data.
data <- data[!(data$class == "red") == ((1.5*data$x + 3 - data$y) > 0), ]
data <- data[sample(seq_len(nrow(data)), 200), ]

p <- ggplot(data, aes(x = x, y = y, color = class)) + 
  geom_point(size = 2) + scale_colour_identity() + 
  geom_abline(intercept = 3, slope = 1.5, lty = 2)
p
```

> b. Compute the cross-validation error rates for support vector classifiers
>    with a range of `cost` values. How many training errors are misclassified
>    for each value of `cost` considered, and how does this relate to the
>    cross-validation errors obtained?

How many training errors are misclassified for each value of cost?

```{r}
costs <- 10^seq(-3, 5)

sapply(costs, function(cost) {
    fit <- svm(as.factor(class) ~ ., data = data, kernel = "linear", cost = cost)
    pred <- predict(fit, data)
    sum(pred != data$class)
})
```

Cross-validation errors

```{r}
out <- tune(svm, as.factor(class) ~ ., data = data, kernel = "linear", ranges = list(cost = costs))
summary(out)
data.frame(
  cost = out$performances$cost, 
  misclass = out$performances$error * nrow(data)
)
```

> c. Generate an appropriate test data set, and compute the test errors
>    corresponding to each of the values of `cost` considered. Which value of
>    `cost` leads to the fewest test errors, and how does this compare to the
>    values of `cost` that yield the fewest training errors and the fewest
>    cross-validation errors?

```{r}
set.seed(2)
test <- data.frame(
  x = rnorm(200),
  class = sample(c("red", "blue"), 200, replace = TRUE)
)
test$y <- (test$class == "red") * 5 + rnorm(200)
p + geom_point(data = test, pch = 21)

(errs <- sapply(costs, function(cost) {
    fit <- svm(as.factor(class) ~ ., data = data, kernel = "linear", cost = cost)
    pred <- predict(fit, test)
    sum(pred != test$class)
}))
(cost <- costs[which.min(errs)])

(fit <- svm(as.factor(class) ~ ., data = data, kernel = "linear", cost = cost))

test$prediction <- predict(fit, test)
p <- ggplot(test, aes(x = x, y = y, color = class, shape = prediction == class)) + 
  geom_point(size = 2) + 
  scale_colour_identity() 
p
```

> d. Discuss your results.

A large cost leads to overfitting as the model finds the perfect linear 
separation between red and blue in the training data. A lower cost then 
leads to improved prediction in the test data.

### Question 7

> In this problem, you will use support vector approaches in order to predict
> whether a given car gets high or low gas mileage based on the `Auto` data set.
> 
> a. Create a binary variable that takes on a 1 for cars with gas mileage above
>    the median, and a 0 for cars with gas mileage below the median.

```{r}
library(ISLR2)
data <- Auto
data$high_mpg <- as.factor(as.numeric(data$mpg > median(data$mpg)))
```

> b. Fit a support vector classifier to the data with various values of `cost`,
>    in order to predict whether a car gets high or low gas mileage. Report the
>    cross-validation errors associated with different values of this parameter.
>    Comment on your results. Note you will need to fit the classifier without
>    the gas mileage variable to produce sensible results.

```{r}
set.seed(42)
costs <- 10^seq(-4, 3, by = 0.5)
results <- list()
f <- high_mpg ~ displacement + horsepower + weight
results$linear <- tune(svm, f, data = data, kernel = "linear", 
  ranges = list(cost = costs))
summary(results$linear)
```

> c. Now repeat (b), this time using SVMs with radial and polynomial basis
>    kernels, with different values of `gamma` and `degree` and `cost`. Comment
>    on your results.

```{r}
results$polynomial <- tune(svm, f, data = data, kernel = "polynomial", 
  ranges = list(cost = costs, degree = 1:3))
summary(results$polynomial)

results$radial <- tune(svm, f, data = data, kernel = "radial", 
  ranges = list(cost = costs, gamma = 10^(-2:1)))
summary(results$radial)

sapply(results, function(x) x$best.performance)
sapply(results, function(x) x$best.parameters)
```

> d. Make some plots to back up your assertions in (b) and (c).
> 
>    _Hint: In the lab, we used the `plot()` function for `svm` objects only in 
>    cases with $p = 2$. When $p > 2$, you can use the `plot()` function to 
>    create plots displaying pairs of variables at a time. Essentially, instead 
>    of typing_
>    
>    ```r
>    > plot(svmfit, dat)
>    ```
>    
>    _where `svmfit` contains your fitted model and dat is a data frame 
>    containing your data, you can type_
>    
>    ```r
>    > plot(svmfit, dat, x1 ∼ x4)
>    ``` 
>    
>    _in order to plot just the first and fourth variables. However, you must 
>    replace `x1` and `x4` with the correct variable names. To find out more, 
>    type `?plot.svm`._

```{r}
table(predict(results$radial$best.model, data), data$high_mpg)

plot(results$radial$best.model, data, horsepower~displacement)
plot(results$radial$best.model, data, horsepower~weight)
plot(results$radial$best.model, data, displacement~weight)
```

### Question 8

> This problem involves the `OJ` data set which is part of the `ISLR2` package.
> 
> a. Create a training set containing a random sample of 800 observations, and a
>    test set containing the remaining observations.

```{r}
set.seed(42)
train <- sample(seq_len(nrow(OJ)), 800)
test <- setdiff(seq_len(nrow(OJ)), train)
```

> b. Fit a support vector classifier to the training data using `cost = 0.01`,
>    with `Purchase` as the response and the other variables as predictors. Use
>    the `summary()` function to produce summary statistics, and describe the
>    results obtained.

```{r}
fit <- svm(Purchase ~ ., data = OJ[train, ], kernel = "linear", cost = 0.01)
summary(fit)
```

> c. What are the training and test error rates?

```{r}
err <- function(model, data) {
  t <- table(predict(model, data), data[["Purchase"]])
  1 - sum(diag(t)) / sum(t)
}
errs <- function(model) {
  c(train = err(model, OJ[train, ]), test = err(model, OJ[test, ]))
}
errs(fit)
```

> d. Use the `tune()` function to select an optimal cost. Consider values in the
>    range 0.01 to 10.

```{r}
tuned <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "linear", 
  ranges = list(cost = 10^seq(-2, 1, length.out = 10)))
tuned$best.parameters
summary(tuned)
```

> e. Compute the training and test error rates using this new value for `cost`.

```{r}
errs(tuned$best.model)
```

> f. Repeat parts (b) through (e) using a support vector machine with a radial
>    kernel. Use the default value for `gamma`.

```{r}
tuned2 <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "radial", 
  ranges = list(cost = 10^seq(-2, 1, length.out = 10)))
tuned2$best.parameters
errs(tuned2$best.model)
```

> g. Repeat parts (b) through (e) using a support vector machine with a
>    polynomial kernel. Set `degree = 2`.

```{r}
tuned3 <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "polynomial", 
  ranges = list(cost = 10^seq(-2, 1, length.out = 10)), degree = 2)
tuned3$best.parameters
errs(tuned3$best.model)
```

> h. Overall, which approach seems to give the best results on this data?

Overall the "radial" kernel appears to perform best in this case.


# Deep Learning

## Conceptual

### Question 1

> Consider a neural network with two hidden layers: $p = 4$ input units, 2 units
> in the first hidden layer, 3 units in the second hidden layer, and a single
> output.
>
> a. Draw a picture of the network, similar to Figures 10.1 or 10.4.

```{r, echo=FALSE, out.width="80%"}
knitr::include_graphics("images/nn.png")
```

> b. Write out an expression for $f(X)$, assuming ReLU activation functions. Be
> as explicit as you can!

The three layers (from our final output layer back to the start of our network)
can be described as:

\begin{align*}
f(X) &= g(w_{0}^{(3)} + \sum^{K_2}_{l=1} w_{l}^{(3)} A_l^{(2)}) \\
A_l^{(2)} &= h_l^{(2)}(X) = g(w_{l0}^{(2)} + \sum_{k=1}^{K_1} w_{lk}^{(2)} A_k^{(1)})\\
A_k^{(1)} &= h_k^{(1)}(X) = g(w_{k0}^{(1)} + \sum_{j=1}^p w_{kj}^{(1)} X_j) \\
\end{align*}

for $l = 1, ..., K_2 = 3$ and $k = 1, ..., K_1 = 2$ and $p = 4$, where,

$$
g(z) = (z)_+ = \begin{cases}
  0, & \text{if } z < 0 \\
  z, & \text{otherwise}
\end{cases}
$$

> c. Now plug in some values for the coefficients and write out the value of
> $f(X)$.

We can perhaps achieve this most easily by fitting a real model. Note,
in the plot shown here, we also include the "bias" or intercept terms.

```{r}
library(ISLR2)
library(neuralnet)
library(sigmoid)
set.seed(5)
train <- sample(seq_len(nrow(ISLR2::Boston)), nrow(ISLR2::Boston) * 2/3)

net <- neuralnet(crim ~ lstat + medv + ptratio + rm,
    data = ISLR2::Boston[train, ],
    act.fct = relu,
    hidden = c(2, 3)
)
plot(net)
```

We can make a prediction for a given observation using this object.

Firstly, let's find an "ambiguous" test sample

```{r}
p <- predict(net, ISLR2::Boston[-train, ])
x <- ISLR2::Boston[-train, ][which.min(abs(p - mean(c(max(p), min(p))))), ]
x <- x[, c("lstat", "medv", "ptratio", "rm")]
predict(net, x)
```

Or, repeating by "hand":

```{r}
g <- function(x) ifelse(x > 0, x, 0) # relu activation function
w <- net$weights[[1]] # the estimated weights for each layer
v <- as.numeric(x) # our input predictors

# to calculate our prediction we can take the dot product of our predictors
# (with 1 at the start for the bias term) and our layer weights, lw)
for (lw in w) v <- g(c(1, v) %*% lw)
v
```

> d. How many parameters are there?

```{r} 
length(unlist(net$weights))
```

There are $4*2+2 + 2*3+3 + 3*1+1 = 23$ parameters.

### Question 2

> Consider the _softmax_ function in (10.13) (see also (4.13) on page 141)
> for modeling multinomial probabilities.
>
> a. In (10.13), show that if we add a constant $c$ to each of the $z_l$, then
> the probability is unchanged.

If we add a constant $c$ to each $Z_l$ in equation 10.13 we get:

\begin{align*}
Pr(Y=m|X) 
 &= \frac{e^{Z_m+c}}{\sum_{l=0}^9e^{Z_l+c}} \\
 &= \frac{e^{Z_m}e^c}{\sum_{l=0}^9e^{Z_l}e^c} \\
 &= \frac{e^{Z_m}e^c}{e^c\sum_{l=0}^9e^{Z_l}} \\
 &= \frac{e^{Z_m}}{\sum_{l=0}^9e^{Z_l}} \\
\end{align*}

which is just equation 10.13.

> b. In (4.13), show that if we add constants $c_j$, $j = 0,1,...,p$, to each of
> the corresponding coefficients for each of the classes, then the predictions
> at any new point $x$ are unchanged.

4.13 is 

$$
Pr(Y=k|X=x) = \frac
{e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
{\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}}
$$

adding constants $c_j$ to each class gives:

\begin{align*}
Pr(Y=k|X=x) 
&= \frac
  {e^{\beta_{K0} + \beta_{K1}x_1 + c_1 + ... + \beta_{Kp}x_p + c_p}}
  {\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + c_1 + ... + \beta_{lp}x_p + c_p}} \\
&= \frac
  {e^{c1 + ... + c_p}e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {\sum_{l=1}^K e^{c1 + ... + c_p}e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
&= \frac
  {e^{c1 + ... + c_p}e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {e^{c1 + ... + c_p}\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
&= \frac
  {e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
\end{align*}

which collapses to 4.13 (with the same argument as above).

> This shows that the softmax function is _over-parametrized_. However,
> regularization and SGD typically constrain the solutions so that this is not a
> problem.

### Question 3

> Show that the negative multinomial log-likelihood (10.14) is equivalent to
> the negative log of the likelihood expression (4.5) when there are $M = 2$
> classes.

Equation 10.14 is 

$$
-\sum_{i=1}^n \sum_{m=0}^9 y_{im}\log(f_m(x_i))
$$

Equation 4.5 is:

$$
\ell(\beta_0, \beta_1) = \prod_{i:y_i=1}p(x_i) \prod_{i':y_i'=0}(1-p(x_i'))
$$

So, $\log(\ell)$ is:

\begin{align*}
\log(\ell) 
 &= \log \left( \prod_{i:y_i=1}p(x_i) \prod_{i':y_i'=0}(1-p(x_i')) \right ) \\
 &= \sum_{i:y_1=1}\log(p(x_i)) + \sum_{i':y_i'=0}\log(1-p(x_i')) \\
\end{align*}

If we set $y_i$ to be an indicator variable such that $y_{i1}$ and $y_{i0}$ are
1 and 0 (or 0 and 1) when our $i$th observation is 1 (or 0) respectively, then
we can write:

$$
\log(\ell) = \sum_{i}y_{i1}\log(p(x_i)) + \sum_{i}y_{i0}\log(1-p(x_i'))
$$

If we also let $f_1(x) = p(x)$ and $f_0(x) = 1 - p(x)$ then:

\begin{align*}
\log(\ell) 
 &= \sum_i y_{i1}\log(f_1(x_i)) + \sum_{i}y_{i0}\log(f_0(x_i')) \\
 &= \sum_i \sum_{m=0}^1 y_{im}\log(f_m(x_i)) \\
\end{align*}

When we take the negative of this, it is equivalent to 10.14 for two classes 
($m = 0,1$).

### Question 4

> Consider a CNN that takes in $32 \times 32$ grayscale images and has a single
> convolution layer with three $5 \times 5$ convolution filters (without
> boundary padding).
>
> a. Draw a sketch of the input and first hidden layer similar to Figure 10.8.

```{r, echo=FALSE, out.width="50%"}
knitr::include_graphics("images/nn2.png")
```

> b. How many parameters are in this model?

There are 5 convolution matrices each with 5x5 weights (plus 5 bias terms) to
estimate, therefore 130 parameters 

> c. Explain how this model can be thought of as an ordinary feed-forward
> neural network with the individual pixels as inputs, and with constraints on
> the weights in the hidden units. What are the constraints?

We can think of a convolution layer as a regularized fully connected layer.
The regularization in this case is due to not all inputs being connected to
all outputs, and weights being shared between connections.

Each output node in the convolved image can be thought of as taking inputs from
a limited number of input pixels (the neighboring pixels), with a set of
weights specified by the convolution layer which are then shared by the
connections to all other output nodes.

> d. If there were no constraints, then how many weights would there be in the
> ordinary feed-forward neural network in (c)?

With no constraints, we would connect each output pixel in our 5x32x32
convolution layer to each node in the 32x32 original image (plus 5 bias terms),
giving a total of 5,242,885 weights to estimate.

### Question 5

> In Table 10.2 on page 433, we see that the ordering of the three methods with
> respect to mean absolute error is different from the ordering with respect to
> test set $R^2$. How can this be?

Mean absolute error considers _absolute_ differences between predictions and 
observed values, whereas $R^2$ considers the (normalized) sum of _squared_
differences, thus larger errors contribute relatively ore to $R^2$ than mean
absolute error.

## Applied

### Question 6

> Consider the simple function $R(\beta) = sin(\beta) + \beta/10$.
>
> a. Draw a graph of this function over the range $\beta \in [−6, 6]$.

```{r}
r <- function(x) sin(x) + x/10
x <- seq(-6, 6, 0.1)
plot(x, r(x), type = "l")
```

> b. What is the derivative of this function?

$$
cos(x) + 1/10
$$

> c. Given $\beta^0 = 2.3$, run gradient descent to find a local minimum of
> $R(\beta)$ using a learning rate of $\rho = 0.1$. Show each of 
> $\beta^0, \beta^1, ...$ in your plot, as well as the final answer.

The derivative of our function, i.e. $cos(x) + 1/10$ gives us the gradient for
a given $x$. For gradient descent, we move $x$ a little in the _opposite_
direction, for some learning rate $\rho = 0.1$:

$$
x^{m+1} = x^m - \rho (cos(x^m) + 1/10)
$$

```{r}
iter <- function(x, rho) x - rho*(cos(x) + 1/10)
gd <- function(start, rho = 0.1) {
  b <- start
  v <- b
  while(abs(b - iter(b, 0.1)) > 1e-8) {
    b <- iter(b, 0.1)
    v <- c(v, b)
  }
  v
}

res <- gd(2.3)
res[length(res)]
```

```{r}
plot(x, r(x), type = "l")
points(res, r(res), col = "red", pch = 19)
```


> d. Repeat with $\beta^0 = 1.4$.

```{r}
res <- gd(1.4)
res[length(res)]
```

```{r}
plot(x, r(x), type = "l")
points(res, r(res), col = "red", pch = 19)
```

### Question 7

> Fit a neural network to the `Default` data. Use a single hidden layer with 10
> units, and dropout regularization. Have a look at Labs 10.9.1–-10.9.2 for
> guidance. Compare the classification performance of your model with that of
> linear logistic regression.

```{r, cache = TRUE}
library(keras)

dat <- ISLR2::Boston
x <- scale(model.matrix(crim ~ . - 1, data = dat))
n <- nrow(dat)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)
y <- dat$crim

# logistic regression
lfit <- lm(crim ~ ., data = dat[-testid, ])
lpred <- predict(lfit, dat[testid, ])
with(dat[testid, ], mean(abs(lpred - crim)))

# keras
nn <- keras_model_sequential() |>
  layer_dense(units = 10, activation = "relu", input_shape = ncol(x)) |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1)

compile(nn, loss = "mse", 
  optimizer = optimizer_rmsprop(), 
  metrics = list("mean_absolute_error") 
)

history <- fit(nn,
  x[-testid, ], y[-testid], 
  epochs = 100, 
  batch_size = 26, 
  validation_data = list(x[testid, ], y[testid]),
  verbose = 0
)
plot(history, smooth = FALSE)
npred <- predict(nn, x[testid, ])
mean(abs(y[testid] - npred))
```

In this case, the neural network outperforms logistic regression having a lower
absolute error rate on the test data.

### Question 8

> From your collection of personal photographs, pick 10 images of animals (such
> as dogs, cats, birds, farm animals, etc.). If the subject does not occupy a
> reasonable part of the image, then crop the image. Now use a pretrained image
> classification CNN as in Lab 10.9.4 to predict the class of each of your
> images, and report the probabilities for the top five predicted classes for
> each image.

```{r, echo=FALSE}
knitr::include_graphics(c(
  "images/animals/bird.jpg",
  "images/animals/bird2.jpg",
  "images/animals/bird3.jpg",
  "images/animals/bug.jpg",
  "images/animals/butterfly.jpg",
  "images/animals/butterfly2.jpg",
  "images/animals/elba.jpg",
  "images/animals/hamish.jpg",
  "images/animals/poodle.jpg",
  "images/animals/tortoise.jpg"
))
```

```{r}
library(keras)
images <- list.files("images/animals")
x <- array(dim = c(length(images), 224, 224, 3))
for (i in seq_len(length(images))) {
  img <- image_load(paste0("images/animals/", images[i]), target_size = c(224, 224))
  x[i,,,] <- image_to_array(img)
}

model <- application_resnet50(weights = "imagenet")

pred <- model |>
  predict(x) |>
  imagenet_decode_predictions(top = 5)
  
names(pred) <- images
print(pred)
```

### Question 9

> Fit a lag-5 autoregressive model to the `NYSE` data, as described in the text
> and Lab 10.9.6. Refit the model with a 12-level factor representing the
> month. Does this factor improve the performance of the model?

Fitting the model as described in the text.

```{r}
library(tidyverse)
library(ISLR2)
xdata <- data.matrix(NYSE[, c("DJ_return", "log_volume","log_volatility")])
istrain <- NYSE[, "train"]
xdata <- scale(xdata)

lagm <- function(x, k = 1) {
  n <- nrow(x)
  pad <- matrix(NA, k, ncol(x))
  rbind(pad, x[1:(n - k), ])
}

arframe <- data.frame(
  log_volume = xdata[, "log_volume"], 
  L1 = lagm(xdata, 1), 
  L2 = lagm(xdata, 2),
  L3 = lagm(xdata, 3),
  L4 = lagm(xdata, 4),
  L5 = lagm(xdata, 5)
)

arframe <- arframe[-(1:5), ]
istrain <- istrain[-(1:5)]

arfit <- lm(log_volume ~ ., data = arframe[istrain, ])
arpred <- predict(arfit, arframe[!istrain, ])
V0 <- var(arframe[!istrain, "log_volume"])
1 - mean((arpred - arframe[!istrain, "log_volume"])^2) / V0
```

Now we add month (and work with tidyverse).

```{r}
arframe$month = as.factor(str_match(NYSE$date, "-(\\d+)-")[,2])[-(1:5)]
arfit2 <- lm(log_volume ~ ., data = arframe[istrain, ])
arpred2 <- predict(arfit2, arframe[!istrain, ])
V0 <- var(arframe[!istrain, "log_volume"])
1 - mean((arpred2 - arframe[!istrain, "log_volume"])^2) / V0
```

Adding month as a factor marginally improves the $R^2$ of our model (from 
0.413223 to 0.4170418). This is a significant improvement in fit and model
2 has a lower AIC.

```{r}
anova(arfit, arfit2)
AIC(arfit, arfit2)
```

### Question 10

> In Section 10.9.6, we showed how to fit a linear AR model to the `NYSE` data
> using the `lm()` function. However, we also mentioned that we can "flatten"
> the short sequences produced for the RNN model in order to fit a linear AR
> model. Use this latter approach to fit a linear AR model to the NYSE data.
> Compare the test $R^2$ of this linear AR model to that of the linear AR model
> that we fit in the lab. What are the advantages/disadvantages of each
> approach?

The `lm` model is the same as that fit above:

```{r}
arfit <- lm(log_volume ~ ., data = arframe[istrain, ])
arpred <- predict(arfit, arframe[!istrain, ])
V0 <- var(arframe[!istrain, "log_volume"])
1 - mean((arpred - arframe[!istrain, "log_volume"])^2) / V0
```

Now we reshape the data for the RNN

```{r}
n <- nrow(arframe)
xrnn <- data.matrix(arframe[, -1])
xrnn <- array(xrnn, c(n, 3, 5))
xrnn <- xrnn[, , 5:1]
xrnn <- aperm(xrnn, c(1, 3, 2))
```

We can add a "flatten" layer to turn the reshaped data into a long vector of
predictors resulting in a linear AR model.

```{r}
model <- keras_model_sequential() |>
  layer_flatten(input_shape = c(5, 3)) |>
  layer_dense(units = 1)
```

Now let's fit this model.

```{r}
model |>
  compile(optimizer = optimizer_rmsprop(), loss = "mse")

history <- model |>
  fit(
    xrnn[istrain,, ],
    arframe[istrain, "log_volume"],
    batch_size = 64,
    epochs = 200,
    validation_data = list(xrnn[!istrain,, ], arframe[!istrain, "log_volume"]),
    verbose = 0
  )

plot(history, smooth = FALSE)
kpred <- predict(model, xrnn[!istrain,, ])
1 - mean((kpred - arframe[!istrain, "log_volume"])^2) / V0
```

Both models estimate the same number of coefficients/weights (16):

```{r}
coef(arfit)
model$get_weights()
```

The flattened RNN has a lower $R^2$ on the test data than our `lm` model
above. The `lm` model is quicker to fit and conceptually simpler also 
giving us the ability to inspect the coefficients for different variables.

The flattened RNN is regularized to some extent as data are processed in
batches.

### Question 11

> Repeat the previous exercise, but now fit a nonlinear AR model by "flattening"
> the short sequences produced for the RNN model.

From the book:

> To fit a nonlinear AR model, we could add in a hidden layer.

```{r, c10q11}
xfun::cache_rds({

  model <- keras_model_sequential() |> 
    layer_flatten(input_shape = c(5, 3)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dropout(rate = 0.4) |> 
    layer_dense(units = 1)

  model |> compile(
    loss = "mse", 
    optimizer = optimizer_rmsprop(), 
    metrics = "mse"
  )

  history <- model |>
    fit(
      xrnn[istrain,, ],
      arframe[istrain, "log_volume"],
      batch_size = 64,
      epochs = 200,
      validation_data = list(xrnn[!istrain,, ], arframe[!istrain, "log_volume"]),
      verbose = 0
    )

  plot(history, smooth = FALSE, metrics = "mse")
  kpred <- predict(model, xrnn[!istrain,, ])
  1 - mean((kpred - arframe[!istrain, "log_volume"])^2) / V0

})
```

This approach improves our $R^2$ over the linear model above.

### Question 12

> Consider the RNN fit to the `NYSE` data in Section 10.9.6. Modify the code to
> allow inclusion of the variable `day_of_week`, and fit the RNN. Compute the
> test $R^2$.

To accomplish this, I'll include day of the week as one of the lagged variables
in the RNN. Thus, our input for each observation will be 4 x 5 (rather than
3 x 5).

```{r, c10q12}
xfun::cache_rds({
  xdata <- data.matrix(
    NYSE[, c("day_of_week", "DJ_return", "log_volume","log_volatility")] 
  )
  istrain <- NYSE[, "train"]
  xdata <- scale(xdata)

  arframe <- data.frame(
    log_volume = xdata[, "log_volume"], 
    L1 = lagm(xdata, 1),
    L2 = lagm(xdata, 2),
    L3 = lagm(xdata, 3), 
    L4 = lagm(xdata, 4),
    L5 = lagm(xdata, 5)
  )
  arframe <- arframe[-(1:5), ]
  istrain <- istrain[-(1:5)]

  n <- nrow(arframe)
  xrnn <- data.matrix(arframe[, -1])
  xrnn <- array(xrnn, c(n, 4, 5))
  xrnn <- xrnn[,, 5:1]
  xrnn <- aperm(xrnn, c(1, 3, 2))
  dim(xrnn)

  model <- keras_model_sequential() |>
      layer_simple_rnn(units = 12,
      input_shape = list(5, 4),
      dropout = 0.1, 
      recurrent_dropout = 0.1
    ) |>
    layer_dense(units = 1)

  model |> compile(optimizer = optimizer_rmsprop(), loss = "mse")

  history <- model |> 
    fit(
      xrnn[istrain,, ],
      arframe[istrain, "log_volume"],
      batch_size = 64,
      epochs = 200,
      validation_data = list(xrnn[!istrain,, ], arframe[!istrain, "log_volume"]),
      verbose = 0
  )

  kpred <- predict(model, xrnn[!istrain,, ])
  1 - mean((kpred - arframe[!istrain, "log_volume"])^2) / V0

})
```

### Question 13

> Repeat the analysis of Lab 10.9.5 on the `IMDb` data using a similarly
> structured neural network. There we used a dictionary of size 10,000. Consider
> the effects of varying the dictionary size. Try the values 1000, 3000, 5000,
> and 10,000, and compare the results.

```{r, c10q13}
xfun::cache_rds({
  library(knitr)
  accuracy <- c()
  for(max_features in c(1000, 3000, 5000, 10000)) {
    imdb <- dataset_imdb(num_words = max_features)
    c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

    maxlen <- 500
    x_train <- pad_sequences(x_train, maxlen = maxlen)
    x_test <- pad_sequences(x_test, maxlen = maxlen)

    model <- keras_model_sequential() |>
      layer_embedding(input_dim = max_features, output_dim = 32) |>
      layer_lstm(units = 32) |>
      layer_dense(units = 1, activation = "sigmoid")

    model |> compile(
      optimizer = "rmsprop",
      loss = "binary_crossentropy", 
      metrics = "acc"
    )

    history <- fit(model, x_train, y_train, 
      epochs = 10, 
      batch_size = 128, 
      validation_data = list(x_test, y_test),
      verbose = 0
    )

    predy <- predict(model, x_test) > 0.5
    accuracy <- c(accuracy, mean(abs(y_test == as.numeric(predy))))
  }

  tibble(
    "Max Features" = c(1000, 3000, 5000, 10000),
    "Accuracy" = accuracy
  ) |>
    kable()

})
```

Varying the dictionary size does not make a substantial impact on our estimates
of accuracy. However, the models do take a substantial amount of time to fit and
it is not clear we are finding the best fitting models in each case. For
example, the model using a dictionary size of 10,000 obtained an accuracy of
0.8721 in the text which is as different from the estimate obtained here as
are the differences between the models with different dictionary sizes.


# Survival Analysis and Censored Data

## Conceptual

### Question 1

> For each example, state whether or not the censoring mechanism is independent.
> Justify your answer.
>
> a. In a study of disease relapse, due to a careless research scientist, all
> patients whose phone numbers begin with the number "2" are lost to follow up.

Independent. There's no reason to think disease relapse should be related to 
the first digit of a phone number.

> b. In a study of longevity, a formatting error causes all patient ages that
> exceed 99 years to be lost (i.e. we know that those patients are more than 99
> years old, but we do not know their exact ages).

Not independent. Older patients are more likely to see an event that younger.

> c. Hospital A conducts a study of longevity. However, very sick patients tend
> to be transferred to Hospital B, and are lost to follow up.

Not independent. Sick patients are more likely to see an event that healthy.

> d. In a study of unemployment duration, the people who find work earlier are
> less motivated to stay in touch with study investigators, and therefore are
> more likely to be lost to follow up.

Not independent. More employable individuals are more likely to see an event.

> e. In a study of pregnancy duration, women who deliver their babies pre-term
> are more likely to do so away from their usual hospital, and thus are more
> likely to be censored, relative to women who deliver full-term babies.

Not independent. Delivery away from hospital will be associated with pregnancy
duration.

> f. A researcher wishes to model the number of years of education of the
> residents of a small town. Residents who enroll in college out of town are
> more likely to be lost to follow up, and are also more likely to attend
> graduate school, relative to those who attend college in town.

Not independent. Years of education will be associated with enrolling in out
of town colleges.

> g. Researchers conduct a study of disease-free survival (i.e. time until
> disease relapse following treatment). Patients who have not relapsed within
> five years are considered to be cured, and thus their survival time is
> censored at five years.

In other words we assume all events happen within five years, so
censoring after this time is equivalent to not censoring at all so
the censoring is independent.

> h. We wish to model the failure time for some electrical component. This
> component can be manufactured in Iowa or in Pittsburgh, with no difference in
> quality. The Iowa factory opened five years ago, and so components
> manufactured in Iowa are censored at five years. The Pittsburgh factory opened
> two years ago, so those components are censored at two years.

If there is no difference in quality then location and therefore censoring is
independent of failure time.

> i. We wish to model the failure time of an electrical component made in two
> different factories, one of which opened before the other. We have reason to
> believe that the components manufactured in the factory that opened earlier
> are of higher quality.

In this case, the difference in opening times of the two locations will mean
that any difference in quality between locations will be associated with
censoring, so censoring is not independent.

### Question 2

> We conduct a study with $n = 4$ participants who have just purchased cell
> phones, in order to model the time until phone replacement. The first
> participant replaces her phone after 1.2 years. The second participant still
> has not replaced her phone at the end of the two-year study period. The third
> participant changes her phone number and is lost to follow up (but has not yet
> replaced her phone) 1.5 years into the study. The fourth participant replaces
> her phone after 0.2 years.
>
> For each of the four participants ($i = 1,..., 4$), answer the following
> questions using the notation introduced in Section 11.1:
>
> a. Is the participant's cell phone replacement time censored?

No, Yes, Yes and No. Censoring occurs when we do not know if or when the phone
was replaced.

> b. Is the value of $c_i$ known, and if so, then what is it?

$c_i$ is censoring time. For the four participants these are: NA. 2. 1.5 and NA.

> c. Is the value of $t_i$ known, and if so, then what is it?

$t_i$ is time to event. For the four participants these are: 1.2, NA, NA and
0.2.

> d. Is the value of $y_i$ known, and if so, then what is it?

$y_i$ is the observed time. For the four participants these are: 1.2, 2, 1.5 and
0.2.

> e. Is the value of $\delta_i$ known, and if so, then what is it?

$\delta_i$ is an indicator for censoring. The nomenclature introduced here
defines this to be 1 if we observe the true "survival" time and 0 if we observe
the censored time. Therefore, for these participants, the values are: 1, 0, 0
and 1.

### Question 3

> For the example in Exercise 2, report the values of $K$, $d_1,...,d_K$,
> $r_1,...,r_K$, and $q_1,...,q_K$, where this notation was defined in Section
> 11.3.

* $K$ is the number of unique deaths, which is 2.
* $d_k$ represents the unique death times, which are: 0.2, 1.2.
* $r_k$ denotes the number of patients alive and in the study just before $d_k$.
  Note the first event is for patient 4, then patient 1, then patient 3 is 
  censored and finally the study ends with patient 2 still involved. Therefore
  $r_k$ takes values are: 4, 3.
* $q_k$ denotes the number of patients who died at time $d_k$, therefore this
  takes values: 1, 1.
  
We can check by using the `survival` package.

```{r}
library(survival)
x <- Surv(c(1.2, 2, 1.5, 0.2), event = c(1, 0, 0, 1))
summary(survfit(x ~ 1))
```

### Question 4

> This problem makes use of the Kaplan-Meier survival curve displayed in Figure
> 11.9. The raw data that went into plotting this survival curve is given in
> Table 11.4. The covariate column of that table is not needed for this problem.
>
> a. What is the estimated probability of survival past 50 days?

There are 2 events that happen before 50 days. The number at
risk $r_k$ are 5 and 4 (one was censored early on), thus survival probability is
$4/5 * 3/4 = 0.6$.

Equivalently, we can use the survival package.

```{r, message = FALSE, warning = FALSE}
library(tidyverse)
```

```{r}
table_data <- tribble(
  ~Y, ~D, ~X,
  26.5, 1, 0.1,
  37.2, 1, 11,
  57.3, 1, -0.3,
  90.8, 0, 2.8,
  20.2, 0, 1.8,
  89.8, 0, 0.4
)
x <- Surv(table_data$Y, table_data$D)
summary(survfit(x ~ 1))
```

> b. Write out an analytical expression for the estimated survival function. For
> instance, your answer might be something along the lines of 
>
> $$
> \hat{S}(t) = \begin{cases}
> 0.8  & \text{if } t < 31\\
> 0.5  & \text{if } 31 \le t < 77\\
> 0.22 & \text{if } 77 \le t
> \end{cases}
> $$
>
> (The previous equation is for illustration only: it is not the correct
> answer!)

$$
\hat{S}(t) = \begin{cases}
1   & \text{if } t < 26.5 \\
0.8 & \text{if } 26.5 \le t < 37.2 \\
0.6 & \text{if } 37.2 \le t < 57.3 \\
0.4 & \text{if } 57.3 \le t
\end{cases}
$$


### Question 5

> Sketch the survival function given by the equation
>
> $$
> \hat{S}(t) = \begin{cases}
> 0.8, & \text{if } t < 31\\
> 0.5, & \text{if } 31 \le t < 77\\
> 0.22 & \text{if } 77 \le t
> \end{cases}
> $$
>
> Your answer should look something like Figure 11.9.

We can draw this plot, or even engineer data that will generate the required
plot...

```{r}
plot(NULL,
  xlim = c(0, 100),
  ylim = c(0, 1),
  ylab = "Estimated Probability of Survival",
  xlab = "Time in Days"
)
lines(
  c(0, 31, 31, 77, 77, 100),
  c(0.8, 0.8, 0.5, 0.5, 0.22, 0.22)
)
```

### Question 6

> This problem makes use of the data displayed in Figure 11.1. In completing
> this problem, you can refer to the observation times as $y_1,...,y_4$. The
> ordering of these observation times can be seen from Figure 11.1; their exact
> values are not required.
>
> a. Report the values of $\delta_1,...,\delta_4$, $K$, $d_1,...,d_K$,
> $r_1,...,r_K$, and $q_1,...,q_K$. The relevant notation is defined in Sections
> 11.1 and 11.3.

* $\delta$ values are: 1, 0, 1, 0.
* $K$ is 2
* $d$ values are $y_3$ and $y_1$.
* $r$ values are 4 and 2.
* $q$ values are 1 and 1.

> b. Sketch the Kaplan-Meier survival curve corresponding to this data set. (You
> do not need to use any software to do this---you can sketch it by hand using
> the results obtained in (a).)

```{r}
plot(NULL,
  xlim = c(0, 350),
  ylim = c(0, 1),
  ylab = "Estimated Probability of Survival",
  xlab = "Time in Days"
)
lines(
  c(0, 150, 150, 300, 300, 350),
  c(1, 1, 0.75, 0.75, 0.375, 0.375)
)
```

x <- Surv(c(300, 350, 150, 250), c(1, 0, 1, 0))

> c. Based on the survival curve estimated in (b), what is the probability that
> the event occurs within 200 days? What is the probability that the event does
> not occur within 310 days?

0.25 and 0.375.

> d. Write out an expression for the estimated survival curve from (b).

$$
\hat{S}(t) = \begin{cases}
1     & \text{if } t < y_3 \\
0.75  & \text{if } y_3 \le t < y_1 \\
0.375 & \text{if } y_1 \le t
\end{cases}
$$

### Question 7

> In this problem, we will derive (11.5) and (11.6), which are needed for the
> construction of the log-rank test statistic (11.8). Recall the notation in
> Table 11.1.
>
> a. Assume that there is no difference between the survival functions of the
> two groups. Then we can think of $q_{1k}$ as the number of failures if we draw
> $r_{1k} observations, without replacement, from a risk set of $r_k$
> observations that contains a total of $q_k$ failures. Argue that $q_{1k}$
> follows a hypergeometric distribution. Write the parameters of this
> distribution in terms of $r_{1k}$, $r_k$, and $q_k$.

A hypergeometric distributions models sampling without replacement from a finite
pool where each sample is a success or failure. This fits the situation here,
where with have a finite number of samples in the risk set.

The hypergeometric distribution is parameterized as $k$ successes in $n$ draws, without replacement, from a population of size $N$ with $K$ objects with that feature.

Mapping to our situation, $q_{1k}$ is $k$, $r_{1k}$ is $n$, $r_k$ is $N$ and $q_k$ is $K$.

> b. Given your previous answer, and the properties of the hypergeometric
> distribution, what are the mean and variance of $q_{1k}$? Compare your answer
> to (11.5) and (11.6).

With the above parameterization, the mean ($n K/N$) is $r_{1k} q_k/r_K$.
The variance $n K/N (N-K)/N (N-n)/(N-1)$ is 

$$
r_{1k} \frac{q_k}{r_k} \frac{r_k-q_k}{r_k} \frac{r_k - r_{1k}}{r_k - 1}
$$

These are equivalent to 11.5 and 11.6.

### Question 8

> Recall that the survival function $S(t)$, the hazard function $h(t)$, and the
> density function $f(t)$ are defined in (11.2), (11.9), and (11.11),
> respectively. Furthermore, define $F(t) = 1 − S(t)$. Show that the following
> relationships hold:
>
> $$
> f(t) = dF(t)/dt \\
> S(t) = \exp\left(-\int_0^t h(u)du\right)
> $$

If $F(t) = 1 - S(t)$, then $F(t)$ is the *cumulative density function* (cdf)
for $t$.

For a continuous distribution, a cdf, e.g. $F(t)$ can be expressed as an
integral (up to some value $x$) of the *probability density function* (pdf),
i.e. $F(t) = \int_{-\infty}^x f(x) dt$. Equivalently, the derivative of the cdf
is its pdf: $f(t) = \frac{d F(t)}{dt}$.

Then,
$h(t) = \frac{f(t)}{S(t)} = \frac{dF(t)/dt}{S(t)} = \frac{-dS(t)/dt}{S(t)}$. 
From basic calculus, this can be rewritten as $h(t) = -\frac{d}{dt}\log{S(t)}$. 
Integrating and then exponentiating we get the second identity.

### Question 9

> In this exercise, we will explore the consequences of assuming that the
> survival times follow an exponential distribution.
>
> a. Suppose that a survival time follows an $Exp(\lambda)$ distribution, so
> that its density function is $f(t) = \lambda\exp(−\lambda t)$. Using the
> relationships provided in Exercise 8, show that $S(t) = \exp(-\lambda t)$.

The cdf of an exponential distribution is $1 - \exp(-\lambda x)$ and 
$S(t)$ is $1 - F(t)$ where $F(t)$ is the cdf.

Hence, $S(t) = \exp(-\lambda t)$.

> b. Now suppose that each of $n$ independent survival times follows an
> $\exp(\lambda)$ distribution. Write out an expression for the likelihood
> function (11.13).

The reference to (11.13) gives us the following formula:

$$
L = \prod_{i=1}^{n} h(y_i)^{\delta_i} S(y_i)
$$

(11.10) also gives us

$$
h(t) = \frac{f(t)}{S(t)}
$$

Plugging in the expressions from part (a), we get

\begin{align*}
h(t) &= \frac{\lambda \exp(- \lambda t)}{\exp(- \lambda t)} \\
     &= \lambda
\end{align*}

Using (11.13), we get the following loss expression:

$$
\ell = \prod_i \lambda^{\delta_i} e^{- \lambda y_i}
$$

> c. Show that the maximum likelihood estimator for $\lambda$ is
> $$ 
> \hat\lambda = \sum_{i=1}^n \delta_i / \sum_{i=1}^n y_i.
> $$

Take the log likelihood.

\begin{align*}
\log \ell &= \sum_i \log \left( \lambda^{\delta_i} e^{- \lambda y_i} \right) \\
    &= \sum_i{\delta_i\log\lambda - \lambda y_i \log e} \\
    &= \sum_i{\delta_i\log\lambda - \lambda y_i} \\
    &= \log\lambda\sum_i{\delta_i} - \lambda\sum_i{y_i}
\end{align*}

Differentiating this expression with respect to $\lambda$ we get:

$$
\frac{d \log \ell}{d \lambda} = \frac{\sum_i{\delta_i}}{\lambda} - \sum_i{y_i}
$$

This function maximises when its gradient is 0. Solving for this gives a MLE of
$\hat\lambda = \sum_{i=1}^n \delta_i / \sum_{i=1}^n y_i$.

> d. Use your answer to (c) to derive an estimator of the mean survival time.
>
> _Hint: For (d), recall that the mean of an $Exp(\lambda)$ random variable is
> $1/\lambda$._

Estimated mean survival would be $1/\lambda$ which given the above would be
$\sum_{i=1}^n y_i / \sum_{i=1}^n \delta_i$, which can be thought of as 
the total observation time over the total number of deaths.

## Applied

### Question 10

> This exercise focuses on the brain tumor data, which is included in the
> `ISLR2` `R` library.
>
> a. Plot the Kaplan-Meier survival curve with ±1 standard error bands, using
> the `survfit()` function in the `survival` package.

```{r}
library(ISLR2)
x <- Surv(BrainCancer$time, BrainCancer$status)
plot(survfit(x ~ 1),
  xlab = "Months",
  ylab = "Estimated Probability of Survival",
  col = "steelblue",
  conf.int = 0.67
)
```

> b. Draw a bootstrap sample of size $n = 88$ from the pairs ($y_i$,
> $\delta_i$), and compute the resulting Kaplan-Meier survival curve. Repeat
> this process $B = 200$ times. Use the results to obtain an estimate of the
> standard error of the Kaplan-Meier survival curve at each timepoint. Compare
> this to the standard errors obtained in (a).

```{r}
plot(survfit(x ~ 1),
  xlab = "Months",
  ylab = "Estimated Probability of Survival",
  col = "steelblue",
  conf.int = 0.67
)
fit <- survfit(x ~ 1)
dat <- tibble(time = c(0, fit$time))
for (i in 1:200) {
  y <- survfit(sample(x, 88, replace = TRUE) ~ 1)
  y <- tibble(time = c(0, y$time), "s{i}" := c(1, y$surv))
  dat <- left_join(dat, y, by = "time")
}
res <- fill(dat, starts_with("s")) |>
  rowwise() |>
  transmute(sd = sd(c_across(starts_with("s"))))
se <- res$sd[2:nrow(res)]
lines(fit$time, fit$surv - se, lty = 2, col = "red")
lines(fit$time, fit$surv + se, lty = 2, col = "red")
```

> c. Fit a Cox proportional hazards model that uses all of the predictors to
> predict survival. Summarize the main findings.

```{r}
fit <- coxph(Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo, data = BrainCancer)
fit
```

`diagnosisHG` and `ki` are highly significant.

> d. Stratify the data by the value of `ki`. (Since only one observation has
> `ki=40`, you can group that observation together with the observations that
> have `ki=60`.) Plot Kaplan-Meier survival curves for each of the five strata,
> adjusted for the other predictors.

To adjust for other predictors, we fit a model that includes those predictors
and use this model to predict new, artificial, data where we allow `ki` to 
take each possible value, but set the other predictors to be the mode or mean
of the other predictors.

```{r}
library(ggfortify)

modaldata <- data.frame(
  sex = rep("Female", 5),
  diagnosis = rep("Meningioma", 5),
  loc = rep("Supratentorial", 5),
  ki = c(60, 70, 80, 90, 100),
  gtv = rep(mean(BrainCancer$gtv), 5),
  stereo = rep("SRT", 5)
)
survplots <- survfit(fit, newdata = modaldata)
plot(survplots, xlab = "Months", ylab = "Survival Probability", col = 2:6)
legend("bottomleft", c("60", "70", "80", "90", "100"), col = 2:6, lty = 1)
```

### Question 11

> This example makes use of the data in Table 11.4.
>
> a. Create two groups of observations. In Group 1, $X < 2$, whereas in Group 2,
> $X \ge 2$. Plot the Kaplan-Meier survival curves corresponding to the two
> groups. Be sure to label the curves so that it is clear which curve
> corresponds to which group. By eye, does there appear to be a difference
> between the two groups' survival curves?

```{r}
x <- split(Surv(table_data$Y, table_data$D), table_data$X < 2)
plot(NULL, xlim = c(0, 100), ylim = c(0, 1), ylab = "Survival Probability")
lines(survfit(x[[1]] ~ 1), conf.int = FALSE, col = 2)
lines(survfit(x[[2]] ~ 1), conf.int = FALSE, col = 3)
legend("bottomleft", c(">= 2", "<2"), col = 2:3, lty = 1)
```

There does not appear to be any difference between the curves.

> b. Fit Cox's proportional hazards model, using the group indicator as a
> covariate. What is the estimated coefficient? Write a sentence providing the
> interpretation of this coefficient, in terms of the hazard or the
> instantaneous probability of the event. Is there evidence that the true
> coefficient value is non-zero?

```{r}
fit <- coxph(Surv(Y, D) ~ X < 2, data = table_data)
fit
```

The coefficient is $0.3401$. This implies a slightly increased hazard when
$X < 2$ but it is not significantly different to zero (P = 0.8).

> c. Recall from Section 11.5.2 that in the case of a single binary covariate,
> the log-rank test statistic should be identical to the score statistic for the
> Cox model. Conduct a log-rank test to determine whether there is a difference
> between the survival curves for the two groups. How does the p-value for the
> log-rank test statistic compare to the $p$-value for the score statistic for
> the Cox model from (b)?

```{r}
summary(fit)$sctest
survdiff(Surv(Y, D) ~ X < 2, data = table_data)$chisq
```

They are identical.


# Unsupervised Learning

## Conceptual

### Question 1

> This problem involves the $K$-means clustering algorithm.
>
> a. Prove (12.18).

12.18 is:

$$
\frac{1}{|C_k|}\sum_{i,i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2 =
2 \sum_{i \in C_k} \sum_{j=1}^p  (x_{ij} - \bar{x}_{kj})^2
$$

where $$\bar{x}_{kj} = \frac{1}{|C_k|}\sum_{i \in C_k} x_{ij}$$

On the left hand side we compute the difference between each observation
(indexed by $i$ and $i'$). In the second we compute the difference between
each observation and the mean. Intuitively this identity is clear (the factor
of 2 is present because we calculate the difference between each pair twice).
However, to prove.

Note first that,
\begin{align}
(x_{ij} - x_{i'j})^2
  = & ((x_{ij} - \bar{x}_{kj}) - (x_{i'j} - \bar{x}_{kj}))^2 \\
  = & (x_{ij} - \bar{x}_{kj})^2 -
      2(x_{ij} - \bar{x}_{kj})(x_{i'j} - \bar{x}_{kj}) +
      (x_{i'j} - \bar{x}_{kj})^2
\end{align}

Note that the first term is independent of $i'$ and the last is independent of
$i$.

Therefore, 10.12 can be written as:

\begin{align}
\frac{1}{|C_k|}\sum_{i,i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2
= & \frac{1}{|C_k|}\sum_{i,i' \in C_k}\sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2 -
    \frac{1}{|C_k|}\sum_{i,i' \in C_k}\sum_{j=1}^p 2(x_{ij} - \bar{x}_{kj})(x_{i'j} - \bar{x}_{kj}) +
    \frac{1}{|C_k|}\sum_{i,i' \in C_k}\sum_{j=1}^p (x_{i'j} - \bar{x}_{kj})^2 \\
= & \frac{|C_k|}{|C_k|}\sum_{i \in C_k}\sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2 -
    \frac{2}{|C_k|}\sum_{i,i' \in C_k}\sum_{j=1}^p (x_{ij} - \bar{x}_{kj})(x_{i'j} - \bar{x}_{kj}) +
    \frac{|C_k|}{|C_k|}\sum_{i \in C_k}\sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2 \\
= & 2 \sum_{i \in C_k}\sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2
\end{align}

Note that we can drop the term containing
$(x_{ij} - \bar{x}_{kj})(x_{i'j} - \bar{x}_{kj})$ since this is 0 when summed
over combinations of $i$ and $i'$ for a given $j$.

> b. On the basis of this identity, argue that the $K$-means clustering
>    algorithm (Algorithm 12.2) decreases the objective (12.17) at each
>    iteration.

Equation 10.12 demonstrates that the euclidean distance between each possible
pair of samples can be related to the difference from each sample to the mean
of the cluster. The K-means algorithm works by minimizing the euclidean distance
to each centroid, thus also minimizes the within-cluster variance.

### Question 2

> Suppose that we have four observations, for which we compute a dissimilarity
> matrix, given by
> 
> \begin{bmatrix}
>      & 0.3  & 0.4  & 0.7  \\
> 0.3  &      & 0.5  & 0.8  \\
> 0.4  & 0.5  &      & 0.45 \\
> 0.7  & 0.8  & 0.45 &      \\
> \end{bmatrix}
> 
> For instance, the dissimilarity between the first and second observations is
> 0.3, and the dissimilarity between the second and fourth observations is 0.8.
> 
> a. On the basis of this dissimilarity matrix, sketch the dendrogram that
>    results from hierarchically clustering these four observations using
>    complete linkage. Be sure to indicate on the plot the height at which each
>    fusion occurs, as well as the observations corresponding to each leaf in
>    the dendrogram.

```{r}
m <- matrix(c(0, 0.3, 0.4, 0.7, 0.3, 0, 0.5, 0.8, 0.4, 0.5, 0., 0.45, 0.7, 0.8, 0.45, 0), ncol = 4)
c1 <- hclust(as.dist(m), method = "complete")
plot(c1)
```

> b. Repeat (a), this time using single linkage clustering.

```{r}
c2 <- hclust(as.dist(m), method = "single")
plot(c2)
```

> c. Suppose that we cut the dendrogram obtained in (a) such that two clusters
>    result. Which observations are in each cluster?

```{r}
table(1:4, cutree(c1, 2))
```

> d. Suppose that we cut the dendrogram obtained in (b) such that two clusters
>    result. Which observations are in each cluster?

```{r}
table(1:4, cutree(c2, 2))
```

> e. It is mentioned in the chapter that at each fusion in the dendrogram, the
>    position of the two clusters being fused can be swapped without changing
>    the meaning of the dendrogram. Draw a dendrogram that is equivalent to the
>    dendrogram in (a), for which two or more of the leaves are repositioned,
>    but for which the meaning of the dendrogram is the same.

```{r}
plot(c1, labels = c(2, 1, 3, 4))
```

### Question 3

> In this problem, you will perform $K$-means clustering manually, with $K = 2$,
> on a small example with $n = 6$ observations and $p = 2$ features. The
> observations are as follows.
> 
> | Obs. | $X_1$ | $X_2$ |
> |------|-------|-------|
> | 1    | 1     | 4     |
> | 2    | 1     | 3     |
> | 3    | 0     | 4     |
> | 4    | 5     | 1     |
> | 5    | 6     | 2     |
> | 6    | 4     | 0     |
> 
> a. Plot the observations.

```{r}
library(ggplot2)
d <- data.frame(
  x1 = c(1, 1, 0, 5, 6, 4),
  x2 = c(4, 3, 4, 1, 2, 0)
)
ggplot(d, aes(x = x1, y = x2)) + geom_point()
```

> b. Randomly assign a cluster label to each observation. You can use the
>    `sample()` command in `R` to do this. Report the cluster labels for each
>    observation.

```{r}
set.seed(42)
d$cluster <- sample(c(1, 2), size = nrow(d), replace = TRUE)
```

> c. Compute the centroid for each cluster.

```{r}
centroids <- sapply(c(1,2), function(i) colMeans(d[d$cluster == i, 1:2]))
```

> d. Assign each observation to the centroid to which it is closest, in terms of
>    Euclidean distance. Report the cluster labels for each observation.

```{r}
dist <- sapply(1:2, function(i) {
    sqrt((d$x1 - centroids[1, i])^2 + (d$x2 - centroids[2, i])^2)
})
d$cluster <- apply(dist, 1, which.min)
```

> e. Repeat (c) and (d) until the answers obtained stop changing.

```{r}
centroids <- sapply(c(1,2), function(i) colMeans(d[d$cluster == i, 1:2]))
dist <- sapply(1:2, function(i) {
    sqrt((d$x1 - centroids[1, i])^2 + (d$x2 - centroids[2, i])^2)
})
d$cluster <- apply(dist, 1, which.min)
```

In this case, we get stable labels after the first iteration.

> f. In your plot from (a), color the observations according to the cluster
>    labels obtained.

```{r}
ggplot(d, aes(x = x1, y = x2, color = factor(cluster))) + geom_point()
```

### Question 4

> Suppose that for a particular data set, we perform hierarchical clustering
> using single linkage and using complete linkage. We obtain two dendrograms.
>
> a. At a certain point on the single linkage dendrogram, the clusters {1, 2, 3}
>    and {4, 5} fuse. On the complete linkage dendrogram, the clusters {1, 2, 3}
>    and {4, 5} also fuse at a certain point. Which fusion will occur higher on
>    the tree, or will they fuse at the same height, or is there not enough
>    information to tell?

The complete linkage fusion will likely be higher in the tree since single
linkage is defined as being the minimum distance between two clusters. However,
there is a chance that they could be at the same height (so technically there
is not enough information to tell).

> b. At a certain point on the single linkage dendrogram, the clusters {5} and
>    {6} fuse. On the complete linkage dendrogram, the clusters {5} and {6} also
>    fuse at a certain point. Which fusion will occur higher on the tree, or
>    will they fuse at the same height, or is there not enough information to
>    tell?

They will fuse at the same height (the algorithm for calculating distance is
the same when the clusters are of size 1).

### Question 5

> In words, describe the results that you would expect if you performed
> $K$-means clustering of the eight shoppers in Figure 12.16, on the basis of
> their sock and computer purchases, with $K = 2$. Give three answers, one for
> each of the variable scalings displayed. Explain.

In cases where variables are scaled we would expect clusters to correspond
to whether or not the retainer sold a computer. In the first case (raw numbers
of items sold), we would expect clusters to represent low vs high numbers of 
sock purchases.

To test, we can run the analysis in R:

```{r}
set.seed(42)
dat <- data.frame(
  socks = c(8, 11, 7, 6, 5, 6, 7, 8),
  computers = c(0, 0, 0, 0, 1, 1, 1, 1)
)
kmeans(dat, 2)$cluster
kmeans(scale(dat), 2)$cluster
dat$computers <- dat$computers * 2000
kmeans(dat, 2)$cluster
```

### Question 6

> We saw in Section 12.2.2 that the principal component loading and score
> vectors provide an approximation to a matrix, in the sense of (12.5).
> Specifically, the principal component score and loading vectors solve the
> optimization problem given in (12.6).
>
> Now, suppose that the M principal component score vectors zim, $m = 1,...,M$,
> are known. Using (12.6), explain that the first $M$ principal component
> loading vectors $\phi_{jm}$, $m = 1,...,M$, can be obtaining by performing $M$
> separate least squares linear regressions. In each regression, the principal
> component score vectors are the predictors, and one of the features of the
> data matrix is the response.

## Applied

### Question 7

> In the chapter, we mentioned the use of correlation-based distance and
> Euclidean distance as dissimilarity measures for hierarchical clustering.
> It turns out that these two measures are almost equivalent: if each
> observation has been centered to have mean zero and standard deviation one,
> and if we let $r_{ij}$ denote the correlation between the $i$th and $j$th
> observations, then the quantity $1 − r_{ij}$ is proportional to the squared
> Euclidean distance between the ith and jth observations.
>
> On the `USArrests` data, show that this proportionality holds.
>
> _Hint: The Euclidean distance can be calculated using the `dist()` function,_
> _and correlations can be calculated using the `cor()` function._

```{r}
dat <- t(scale(t(USArrests)))
d1 <- dist(dat)^2
d2 <- as.dist(1 - cor(t(dat)))
plot(d1, d2)
```

### Question 8

> In Section 12.2.3, a formula for calculating PVE was given in Equation
> 12.10. We also saw that the PVE can be obtained using the `sdev` output of the
> `prcomp()` function.
>
> On the `USArrests` data, calculate PVE in two ways:
>
> a. Using the `sdev` output of the `prcomp()` function, as was done in Section
>    12.2.3.

```{r}
pr <- prcomp(USArrests, scale = TRUE)
pr$sdev^2 / sum(pr$sdev^2)
```

> b. By applying Equation 12.10 directly. That is, use the `prcomp()` function to
>    compute the principal component loadings. Then, use those loadings in
>    Equation 12.10 to obtain the PVE. 
>
> These two approaches should give the same results.

```{r}
colSums(pr$x^2) / sum(colSums(scale(USArrests)^2))
```

> _Hint: You will only obtain the same results in (a) and (b) if the same_
> _data is used in both cases. For instance, if in (a) you performed_
> _`prcomp()` using centered and scaled variables, then you must center and_
> _scale the variables before applying Equation 12.10 in (b)._

### Question 9

> Consider the `USArrests` data. We will now perform hierarchical clustering on
> the states.
>
> a. Using hierarchical clustering with complete linkage and Euclidean distance,
>    cluster the states.

```{r}
set.seed(42)
hc <- hclust(dist(USArrests), method = "complete")
```

> b. Cut the dendrogram at a height that results in three distinct clusters.
>    Which states belong to which clusters?

```{r}
ct <- cutree(hc, 3)
sapply(1:3, function(i) names(ct)[ct == i])
```

> c. Hierarchically cluster the states using complete linkage and Euclidean
>    distance, _after scaling the variables to have standard deviation one_.

```{r}
hc2 <- hclust(dist(scale(USArrests)), method = "complete")
```

> d. What effect does scaling the variables have on the hierarchical clustering
>    obtained? In your opinion, should the variables be scaled before the
>    inter-observation dissimilarities are computed? Provide a justification for
>    your answer.

```{r}
ct <- cutree(hc, 3)
sapply(1:3, function(i) names(ct)[ct == i])
```

Scaling results in different clusters and the choice of whether to scale or 
not depends on the data in question. In this case, the variables are:

  - Murder    numeric  Murder arrests (per 100,000)  
  - Assault   numeric  Assault arrests (per 100,000) 
  - UrbanPop  numeric  Percent urban population      
  - Rape      numeric  Rape arrests (per 100,000)    

These variables are not naturally on the same unit and the units involved are
somewhat arbitrary (so for example, Murder could be measured per 1 million 
rather than per 100,000) so in this case I would argue the data should be 
scaled.

### Question 10

> In this problem, you will generate simulated data, and then perform PCA and
> $K$-means clustering on the data.
>
> a. Generate a simulated data set with 20 observations in each of three classes
>    (i.e. 60 observations total), and 50 variables. 
>    
>    _Hint: There are a number of functions in `R` that you can use to generate_
>    _data. One example is the `rnorm()` function; `runif()` is another option._
>    _Be sure to add a mean shift to the observations in each class so that_
>    _there are three distinct classes._

```{r}
set.seed(42)
data <- matrix(rnorm(60 * 50), ncol = 50)
classes <- rep(c("A", "B", "C"), each = 20)
dimnames(data) <- list(classes, paste0("v", 1:50))
data[classes == "B", 1:10] <- data[classes == "B", 1:10] + 1.2
data[classes == "C", 5:30] <- data[classes == "C", 5:30] + 1
```

> b. Perform PCA on the 60 observations and plot the first two principal
>    component score vectors. Use a different color to indicate the
>    observations in each of the three classes. If the three classes appear
>    separated in this plot, then continue on to part (c). If not, then return
>    to part (a) and modify the simulation so that there is greater separation
>    between the three classes. Do not continue to part (c) until the three
>    classes show at least some separation in the first two principal component
>    score vectors.

```{r}
pca <- prcomp(data)
ggplot(data.frame(Class = classes, PC1 = pca$x[, 1], PC2 = pca$x[, 2]),
    aes(x = PC1, y = PC2, col = Class)) + 
  geom_point()
```

> c. Perform $K$-means clustering of the observations with $K = 3$. How well do
>    the clusters that you obtained in $K$-means clustering compare to the true
>    class labels?
>    
>    _Hint: You can use the `table()` function in `R` to compare the true class_
>    _labels to the class labels obtained by clustering. Be careful how you_
>    _interpret the results: $K$-means clustering will arbitrarily number the_
>    _clusters, so you cannot simply check whether the true class labels and_
>    _clustering labels are the same._

```{r}
km <- kmeans(data, 3)$cluster
table(km, names(km))
```

$K$-means separates out the clusters nearly perfectly.

> d.  Perform $K$-means clustering with $K = 2$. Describe your results.

```{r}
km <- kmeans(data, 2)$cluster
table(km, names(km))
```

$K$-means effectively defines cluster 2 to be class B, but cluster 1 is a mix
of classes A and B. 

> e.  Now perform $K$-means clustering with $K = 4$, and describe your results.

```{r}
km <- kmeans(data, 4)$cluster
table(km, names(km))
```

$K$-means effectively defines cluster 1 to be class B, cluster 2 to be class A
but clusters 3 and 4 are split over class C.

> f.  Now perform $K$-means clustering with $K = 3$ on the first two principal
>     component score vectors, rather than on the raw data. That is, perform
>     $K$-means clustering on the $60 \times 2$ matrix of which the first column
>     is the first principal component score vector, and the second column is
>     the second principal component score vector. Comment on the results.

```{r}
km <- kmeans(pca$x[, 1:2], 3)$cluster
table(km, names(km))
```

$K$-means again separates out the clusters nearly perfectly.

> g.  Using the `scale()` function, perform $K$-means clustering with $K = 3$ on
>     the data _after scaling each variable to have standard deviation one_. How
>     do these results compare to those obtained in (b)? Explain.

```{r}
km <- kmeans(scale(data), 3)$cluster
table(km, names(km))
```

$K$-means appears to perform less well on the scaled data in this case.

### Question 11

> Write an `R` function to perform matrix completion as in Algorithm 12.1, and
> as outlined in Section 12.5.2. In each iteration, the function should keep
> track of the relative error, as well as the iteration count. Iterations should
> continue until the relative error is small enough or until some maximum number
> of iterations is reached (set a default value for this maximum number).
> Furthermore, there should be an option to print out the progress in each
> iteration.
> 
> Test your function on the `Boston` data. First, standardize the features to
> have mean zero and standard deviation one using the `scale()` function. Run an
> experiment where you randomly leave out an increasing (and nested) number of
> observations from 5% to 30%, in steps of 5%. Apply Algorithm 12.1 with $M =
> 1,2,...,8$. Display the approximation error as a function of the fraction of
> observations that are missing, and the value of $M$, averaged over 10
> repetitions of the experiment.

### Question 12

> In Section 12.5.2, Algorithm 12.1 was implemented using the `svd()` function.
> However, given the connection between the `svd()` function and the `prcomp()`
> function highlighted in the lab, we could have instead implemented the
> algorithm using `prcomp()`.
>
> Write a function to implement Algorithm 12.1 that makes use of `prcomp()`
> rather than `svd()`.

### Question 13

> On the book website, `www.StatLearning.com`, there is a gene expression data
> set (`Ch12Ex13.csv`) that consists of 40 tissue samples with measurements on
> 1,000 genes. The first 20 samples are from healthy patients, while the
> second 20 are from a diseased group.
>
> a. Load in the data using `read.csv()`. You will need to select `header = F`.

```{r}
data <- read.csv("data/Ch12Ex13.csv", header = FALSE)
colnames(data) <- c(paste0("H", 1:20), paste0("D", 1:20))
```

> b. Apply hierarchical clustering to the samples using correlation-based
>    distance, and plot the dendrogram. Do the genes separate the samples into
>    the two groups? Do your results depend on the type of linkage used?

```{r}
hc.complete <- hclust(as.dist(1 - cor(data)), method = "complete")
plot(hc.complete)

hc.complete <- hclust(as.dist(1 - cor(data)), method = "average")
plot(hc.complete)

hc.complete <- hclust(as.dist(1 - cor(data)), method = "single")
plot(hc.complete)
```

Yes the samples clearly separate into the two groups, although the results 
depend somewhat on the linkage method used. In the case of average clustering,
the disease samples all fall within a subset of the healthy samples.

> c. Your collaborator wants to know which genes differ the most across the two
>    groups. Suggest a way to answer this question, and apply it here.

This is probably best achieved with a supervised approach. A simple method
would be to determine which genes show the most significant differences between
the groups by applying a t-test to each group. We can then select those with a
FDR adjusted p-value less than some given threshold (e.g. 0.05).

```{r}
class <- factor(rep(c("Healthy", "Diseased"), each = 20))
pvals <- p.adjust(apply(data, 1, function(v) t.test(v ~ class)$p.value))
which(pvals < 0.05)
```


# Multiple Testing

## Conceptual

### Question 1

> Suppose we test $m$ null hypotheses, all of which are true. We control the
> Type I error for each null hypothesis at level $\alpha$. For each sub-problem,
> justify your answer.
>
> a. In total, how many Type I errors do we expect to make?

We expect $m\alpha$.

> b. Suppose that the m tests that we perform are independent. What is the
>    family-wise error rate associated with these m tests?
>    
>    _Hint: If two events A and B are independent, then Pr(A ∩ B) = Pr(A) Pr(B)._

The family-wise error rate (FWER) is defined as the probability of making at 
least one Type I error. We can think of this as 1 minus the probability of
no type I errors, which is:

$1 - (1 - \alpha)^m$

Alternatively, for two tests this is: Pr(A ∪ B) = Pr(A) + Pr(B) - Pr(A ∩ B).
For independent tests this is $\alpha + \alpha - \alpha^2$



> c. Suppose that $m = 2$, and that the p-values for the two tests are
>    positively correlated, so that if one is small then the other will tend to
>    be small as well, and if one is large then the other will tend to be large.
>    How does the family-wise error rate associated with these $m = 2$ tests
>    qualitatively compare to the answer in (b) with $m = 2$?
>    
>    _Hint: First, suppose that the two p-values are perfectly correlated._

If they were perfectly correlated, we would effectively be performing a single
test (thus FWER would be $alpha$). In the case when they are positively
correlated therefore, we can expect the FWER to be less than in b.

Alternatively, as above, FWEW = Pr(A ∪ B) = Pr(A) + Pr(B) - Pr(A ∩ B).
For perfectly positively correlated tests Pr(A ∩ B) = $\alpha$, so the 
FWEW is $\alpha$ which is smaller than b.

> d. Suppose again that $m = 2$, but that now the p-values for the two tests are
>    negatively correlated, so that if one is large then the other will tend to
>    be small. How does the family-wise error rate associated with these $m = 2$
>    tests qualitatively compare to the answer in (b) with $m = 2$?
>
>    _Hint: First, suppose that whenever one p-value is less than $\alpha$,_
>    _then the other will be greater than $\alpha$. In other words, we can_
>    _never reject both null hypotheses._

Taking the equation above, for two tests,
FWEW = Pr(A ∪ B) = Pr(A) + Pr(B) - Pr(A ∩ B). In the case considered in the
hint Pr(A ∩ B) = 0, so Pr(A ∪ B) = $2\alpha$, which is larger than b.

### Question 2

> Suppose that we test $m$ hypotheses, and control the Type I error for each
> hypothesis at level $\alpha$. Assume that all $m$ p-values are independent,
> and that all null hypotheses are true.
>
> a. Let the random variable $A_j$ equal 1 if the $j$th null hypothesis is
> rejected, and 0 otherwise. What is the distribution of $A_j$?

$A_j$ follows a Bernoulli distribution: $A_j \sim \text{Bernoulli}(p)$

> b. What is the distribution of $\sum_{j=1}^m A_j$?

Follows a binomial distribution $\sum_{j=1}^m A_j \sim Bi(m, \alpha)$.

> c. What is the standard deviation of the number of Type I errors that we will
> make?

The variance of a Binomial is $npq$, so for this situation the standard 
deviation would be $\sqrt{m \alpha (1-\alpha)}$.

### Question 3

> Suppose we test $m$ null hypotheses, and control the Type I error for the
> $j$th null hypothesis at level $\alpha_j$, for $j=1,...,m$. Argue that the
> family-wise error rate is no greater than $\sum_{j=1}^m \alpha_j$.

$p(A \cup B) = p(A) + p(B)$ if $A$ and $B$ are independent or 
$p(A) + p(B) - p(A \cap B)$ when they are not. Since $p(A \cap B)$ must be 
positive, $p(A \cup B) < p(A) + p(B)$ (whether independent or not).

Therefore, the probability of a type I error in _any_ of $m$ hypotheses can
be no larger than the sum of the probabilities for each individual hypothesis
(which is $\alpha_j$ for the $j$th).

### Question 4

> Suppose we test $m = 10$ hypotheses, and obtain the p-values shown in Table
> 13.4.

```{r}
pvals <- c(0.0011, 0.031, 0.017, 0.32, 0.11, 0.90, 0.07, 0.006, 0.004, 0.0009)
names(pvals) <- paste0("H", sprintf("%02d", 1:10))
```

> a. Suppose that we wish to control the Type I error for each null hypothesis
> at level $\alpha = 0.05$. Which null hypotheses will we reject?

```{r}
names(which(pvals < 0.05))
```

We reject all NULL hypotheses where $p < 0.05$.

> b. Now suppose that we wish to control the FWER at level $\alpha = 0.05$.
> Which null hypotheses will we reject? Justify your answer.

```{r}
names(which(pvals < 0.05 / 10))
```

We reject all NULL hypotheses where $p < 0.005$.

> c. Now suppose that we wish to control the FDR at level $q = 0.05$. Which null
> hypotheses will we reject? Justify your answer.

```{r}
names(which(p.adjust(pvals, "fdr") < 0.05))
```

We reject all NULL hypotheses where $q < 0.05$.

> d. Now suppose that we wish to control the FDR at level $q = 0.2$. Which null
> hypotheses will we reject? Justify your answer.

```{r}
names(which(p.adjust(pvals, "fdr") < 0.2))
```

We reject all NULL hypotheses where $q < 0.2$.

> e. Of the null hypotheses rejected at FDR level $q = 0.2$, approximately how
> many are false positives? Justify your answer.

We expect 20% (in this case 2 out of the 8) rejections to be false (false
positives).

### Question 5

> For this problem, you will make up p-values that lead to a certain number of
> rejections using the Bonferroni and Holm procedures.
>
> a. Give an example of five p-values (i.e. five numbers between 0 and 1 which,
> for the purpose of this problem, we will interpret as p-values) for which
> both Bonferroni’s method and Holm’s method reject exactly one null hypothesis
> when controlling the FWER at level 0.1.

In this case, for Bonferroni, we need one p-value to be less than $0.1 / 5 =
0.02$. and the others to be above. For Holm's method, we need the most
significant p-value to be below $0.1/(5 + 1 - 1) = 0.02$ also.

An example would be: 1, 1, 1, 1, 0.001.

```{r}
pvals <- c(1, 1, 1, 1, 0.001)
sum(p.adjust(pvals, method = "bonferroni") < 0.1)
sum(p.adjust(pvals, method = "holm") < 0.1)
```

> b. Now give an example of five p-values for which Bonferroni rejects one
> null hypothesis and Holm rejects more than one null hypothesis at level 0.1.

An example would be: 1, 1, 1, 0.02, 0.001. For Holm's method we reject two
because $0.02 < 0.1/(5 + 1 - 2)$.

```{r}
pvals <- c(1, 1, 1, 0.02, 0.001)
sum(p.adjust(pvals, method = "bonferroni") < 0.1)
sum(p.adjust(pvals, method = "holm") < 0.1)
```

### Question 6

> For each of the three panels in Figure 13.3, answer the following questions:

* There are always: 8 positives (red) and 2 negatives (black).
* False / true positives are black / red points _below_ the line respectively.
* False / true negatives are red / black points _above_ the line respectively.
* Type I / II errors are the same as false positives and false negatives
  respectively.

> a. How many false positives, false negatives, true positives, true negatives,
> Type I errors, and Type II errors result from applying the Bonferroni
> procedure to control the FWER at level $\alpha = 0.05$?

| Panel | FP | FN | TP | TN | Type I | Type II |
|------ |--- |--- |--- |--- |------- |-------- |
| 1     | 0  | 1  | 7  | 2  | 0      | 1       |
| 2     | 0  | 1  | 7  | 2  | 0      | 1       |
| 3     | 0  | 5  | 3  | 2  | 0      | 5       |

> b. How many false positives, false negatives, true positives, true negatives,
> Type I errors, and Type II errors result from applying the Holm procedure to
> control the FWER at level $\alpha = 0.05$?

| Panel | FP | FN | TP | TN | Type I | Type II |
|------ |--- |--- |--- |--- |------- |-------- |
| 1     | 0  | 1  | 7  | 2  | 0      | 1       |
| 2     | 0  | 0  | 8  | 2  | 0      | 0       |
| 3     | 0  | 0  | 8  | 2  | 0      | 0       |

> c. What is the false discovery rate associated with using the Bonferroni
> procedure to control the FWER at level $\alpha = 0.05$?

False discovery rate is the expected ratio of false positives to total
positives. There are never any false positives (black points below the line).
There are always the same number of total positives (8).

For panels 1, 2, 3 this would be 0/8, 0/8 and 0/8 respectively.

> d. What is the false discovery rate associated with using the Holm procedure
> to control the FWER at level $\alpha = 0.05$?

For panels 1, 2, 3 this would be 0/8, 0/8 and 0/8 respectively.

> e. How would the answers to (a) and (c) change if we instead used the
> Bonferroni procedure to control the FWER at level $\alpha = 0.001$?

This would equate to a more stringent threshold. We would not call any more
false positives, so the results would not change.

## Applied

### Question 7

> This problem makes use of the `Carseats` dataset in the `ISLR2` package.
>
> a. For each quantitative variable in the dataset besides `Sales`, fit a linear
> model to predict `Sales` using that quantitative variable. Report the p-values
> associated with the coefficients for the variables. That is, for each model of
> the form $Y = \beta_0 + \beta_1X + \epsilon$, report the p-value associated
> with the coefficient $\beta_1$. Here, $Y$ represents `Sales` and $X$
> represents one of the other quantitative variables.

```{r}
library(ISLR2)

nm <- c("CompPrice", "Income", "Advertising", "Population", "Price", "Age")
pvals <- sapply(nm, function(n) {
  summary(lm(Carseats[["Sales"]] ~ Carseats[[n]]))$coef[2, 4]
})
```

> b. Suppose we control the Type I error at level $\alpha = 0.05$ for the
> p-values obtained in (a). Which null hypotheses do we reject?

```{r}
names(which(pvals < 0.05))
```

> c. Now suppose we control the FWER at level 0.05 for the p-values. Which null
> hypotheses do we reject?

```{r}
names(which(pvals < 0.05 / length(nm)))
```

> d. Finally, suppose we control the FDR at level 0.2 for the p-values. Which
> null hypotheses do we reject?

```{r}
names(which(p.adjust(pvals, "fdr") < 0.2))
```

### Question 8

> In this problem, we will simulate data from $m = 100$ fund managers.
>
> ```r
> set.seed(1)
> n <- 20
> m <- 100
> X <- matrix(rnorm(n * m), ncol = m)
> ```

```{r}
set.seed(1)
n <- 20
m <- 100
X <- matrix(rnorm(n * m), ncol = m)
```

> These data represent each fund manager’s percentage returns for each of $n =
> 20$ months. We wish to test the null hypothesis that each fund manager’s
> percentage returns have population mean equal to zero. Notice that we
> simulated the data in such a way that each fund manager’s percentage returns
> do have population mean zero; in other words, all $m$ null hypotheses are true.
>
> a. Conduct a one-sample $t$-test for each fund manager, and plot a histogram
> of the $p$-values obtained.

```{r}
pvals <- apply(X, 2, function(p) t.test(p)$p.value)
hist(pvals, main = NULL)
```

> b. If we control Type I error for each null hypothesis at level $\alpha =
> 0.05$, then how many null hypotheses do we reject?

```{r}
sum(pvals < 0.05)
```
> c. If we control the FWER at level 0.05, then how many null hypotheses do we
> reject?

```{r}
sum(pvals < 0.05 / length(pvals))
```

> d. If we control the FDR at level 0.05, then how many null hypotheses do we
> reject?

```{r}
sum(p.adjust(pvals, "fdr") < 0.05)
```

> e. Now suppose we “cherry-pick” the 10 fund managers who perform the best in
> our data. If we control the FWER for just these 10 fund managers at level
> 0.05, then how many null hypotheses do we reject? If we control the FDR for
> just these 10 fund managers at level 0.05, then how many null hypotheses do we
> reject?

```{r}
best <- order(apply(X, 2, sum), decreasing = TRUE)[1:10]
sum(pvals[best] < 0.05 / 10)
sum(p.adjust(pvals[best], "fdr") < 0.05)
```

> f. Explain why the analysis in (e) is misleading.
>
> _Hint The standard approaches for controlling the FWER and FDR assume that all
> tested null hypotheses are adjusted for multiplicity, and that no
> “cherry-picking” of the smallest p-values has occurred. What goes wrong if we
> cherry-pick?_

This is misleading because we are not correctly accounting for all tests
performed. Cherry picking the similar to repeating a test until by chance we
find a significant result.
