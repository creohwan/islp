# islp

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
