seed <- 8675309
set.seed(seed)

ntimes <- 250
nseries <- 20
nfactors <- 6
residual.sd <- 1.2
state.innovation.sd <- .75

##---------------------------------------------------------------------------
## simulate latent state for fake data.
##---------------------------------------------------------------------------
state <- matrix(rnorm(ntimes * nfactors, 0, state.innovation.sd), nrow = ntimes)
for (i in 1:ncol(state)) {
  state[, i] <- cumsum(state[, i])
}

##---------------------------------------------------------------------------
## Simulate "observed" data from state.
##---------------------------------------------------------------------------
observation.coefficients <- matrix(rnorm(nseries * nfactors), nrow = nseries)
print("observation.coefficients =\n", observation.coefficients)
diag(observation.coefficients) <- 1.0
observation.coefficients[upper.tri(observation.coefficients)] <- 0

errors <- matrix(rnorm(nseries * ntimes, 0, residual.sd), ncol = ntimes)
y <- t(observation.coefficients %*% t(state) + errors)

## Not run: 
cat("simulated y (first 10 rows): \n")
print(y[1:10, ])

##---------------------------------------------------------------------------
## Plot the data.
##---------------------------------------------------------------------------
par(mfrow=c(1,2))
plot.ts(y, plot.type="single", col = rainbow(nseries), main = "observed data")
plot.ts(state, plot.type = "single", col = 1:nfactors, main = "latent state")

## End(Not run)

##---------------------------------------------------------------------------
## Fit the model
##---------------------------------------------------------------------------
ss <- AddSharedLocalLevel(list(), y, nfactors = nfactors)

opts <- list("fixed.state" = t(state),
  fixed.residual.sd = rep(residual.sd, nseries),
  fixed.regression.coefficients = matrix(rep(0, nseries), ncol = 1))

model <- mbsts(y, shared.state.specification = ss, niter = 100,
  data.format = "wide", seed = seed)

##---------------------------------------------------------------------------
## Plot the state
##---------------------------------------------------------------------------
par(mfrow=c(1, nfactors))
ylim <- range(model$shared.state, state)
for (j in 1:nfactors) {
  PlotDynamicDistribution(model$shared.state[, j, ], ylim=ylim)
  lines(state[, j], col = "blue")
}

##---------------------------------------------------------------------------
## Plot the factor loadings.
##---------------------------------------------------------------------------
## Not run: 
opar <- par(mfrow=c(nfactors,1), mar=c(0, 4, 0, 4), omi=rep(.25, 4))
burn <- 10
for(j in 1:nfactors) {
  BoxplotTrue(model$shared.local.level.coefficients[-(1:burn), j, ],
    t(observation.coefficients[, j]), axes=F, truth.color="blue")
  abline(h=0, lty=3)
  box()
  axis(2)
}
axis(1)
par(opar)

## End(Not run)

##---------------------------------------------------------------------------
## Plot the predicted values of the series.
##---------------------------------------------------------------------------
index <- 1:12
nr <- floor(sqrt(length(index)))
nc <- ceiling(length(index) / nr)
opar <- par(mfrow = c(nr, nc), mar = c(2, 4, 1, 2))
for (i in index) {
  PlotDynamicDistribution(
    model$shared.state.contributions[, 1, i, ]
    + model$regression.coefficients[, i, 1]
  , ylim=range(y))
  points(y[, i], col="blue", pch = ".", cex = .2)
}
par(opar)