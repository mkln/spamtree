
\name{spamtree}
\alias{spamtree}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Bayesian Spatial Multivariate Tree GP Regression
}
\description{
Bayesian linear multivariate spatial regression using SpamTrees.
}
\usage{
spamtree(y, x, coords,
        mv_id = rep(1, length(y)),
        cell_size = 25,
        K = rep(2, ncol(coords)),
        start_level = 0,
        tree_depth = Inf,
        last_not_reference = TRUE,
        limited_tree = FALSE,
        cherrypick_same_margin = TRUE,
        cherrypick_group_locations = TRUE,
        mvbias = 0,
        mcmc = list(keep = 1000, burn = 0, thin = 1),
        num_threads = 4,
        verbose = FALSE, 
        settings = list(adapting = TRUE, mcmcsd = 0.01, 
          debug = FALSE, printall = FALSE),
        prior = list(set_unif_bounds = NULL, 
          btmlim = NULL, toplim = NULL, vlim = NULL),
        starting = list(beta = NULL, tausq = NULL, theta = NULL, w = NULL),
        debug = list(sample_beta = TRUE, sample_tausq = TRUE, 
          sample_theta = TRUE, sample_w = TRUE, 
          sample_predicts = TRUE)
)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{
vector of outcomes of size \eqn{n}. Correspondingly, \code{y[i]} is the observation of outcome \code{mv_id[i]} at location \code{coords[i,]} and with covariates \code{x[i,]}. This means that if the number of outcomes is \eqn{q>1} then these are all stacked in a vector, and their integer ID is stored in \code{mv_id}. 
}
  \item{x}{
matrix of covariates with dimension \eqn{(n, p)}.
}
  \item{coords}{
matrix of coordinates with dimension \eqn{(n, 2)}.
}
  \item{mv_id}{
integer vector of outcome IDs of size \eqn{n} with values in \eqn{\{1, \dots, q\}}.
}
  \item{cell_size}{
integer number of knots for each node in the treed DAG. Defaults to 25. This is a target number and some nodes may include more or less locations. Here, knots can only be chosen among observed locations.
}
  \item{K}{
integer vector of dimension 2 indicating the number of intervals for axis-parallel recursive partitioning. Each tree level will thus have \code{prod(K)} times as many partitions as the previous. Defaults to \code{c(2,2)}, leading to a recursive quadtree.
}
  \item{start_level}{
integer indicating the root level. Example: \code{start_level=0} means there is 1 root node. \code{start_level=1} means there are \code{prod(K)} root nodes.
}
  \item{tree_depth}{
integer indicating the number of branching steps in the tree. Defaults to \code{Inf} meaning that observed locations will be placed on tree nodes as much as possible.
}
  \item{last_not_reference}{
bool indicating whether to treat the last level of the tree as a reference set. The default value \code{TRUE} is recommended when \code{tree_depth=Inf} or whenever only a very small number of observed locations remain at the last level.  
}
  \item{limited_tree}{
bool determining whether to use a recursive tree. If \code{TRUE}, each non-root node has 1 parent and \code{prod(K)} children. Otherwise, each node at level \eqn{L} (where root nodes have \eqn{L=0}) has \eqn{L} parents.  
}
  \item{cherrypick_same_margin}{
bool used only for multivariate outcomes. This determines how to assign parents to leaf nodes. In a SpamTree, outcome \eqn{j} at a new spatial location is assigned the same parent of its nearest neighbor. If \code{cherrypick_same_margin=TRUE} then the nearest neighbor is searched within the subset of locations for which outcome \eqn{j} was observed. Otherwise, the nearest neighbor is searched within all locations at which any outcome was observed. If outcomes are aligned (all observed at the same locations) and \code{cherrypick_group_locations=TRUE}, then this setting has minimal or no effect. 
}
  \item{cherrypick_group_locations}{
bool used in multivariate settings to determine whether the allocation of knots to tree nodes should treat the \eqn{q} outcomes at location \eqn{s \in D} as either (1) a \eqn{q} dimensional vector observed at 1 location, or (2) one observed outcome at each of \eqn{q} locations (i.e. same spatial location but different outcome index). 
}
  \item{mvbias}{
parameter used in settings of multivariate misalignment in which one or more outcomes are observed at a number of locations that is much smaller than others. \code{mvbias} can be used to disproportionately place the more sparsely observed outcomes near root nodes. This is justified by Prop. 1 in Peruzzi and Dunson (2021).
}
  \item{mcmc}{
list for setting up MCMC. \code{mcmc$keep} is the number of MCMC samples to be saved, \code{mcmc$burn} is the number of iterations for burn-in, \code{mcmc$thin} is the thinning level for the chain. The total number of iterations that will be run is \code{burn + thin*keep}.
}
  \item{num_threads}{
integer number of OpenMP threads to use within MCMC. Ineffective if source is compiled without OpenMP support.
}
\item{verbose}{
level of verbosity. All messages are suppressed if \code{verbose=FALSE} (default). It is useful to set \code{verbose=TRUE} for data of medium size or long MCMC chains.
}
  \item{settings}{
list with additional settings. \code{settings$adapting} determines whether to use Robust Adaptive Metropolis algorithm of Vihola (2012). \code{settings$mcmcsd} is the initial standard deviation for the MCMC proposals before adaptation. \code{settings$debug} prints some debug messages. \code{settings$printall} determines whether to print to console at each iteration.
}
  \item{prior}{
setup for prior on \eqn{\theta}, which currently only allows to specify the support of independent uniform distributions. See examples. (subject to change).
}
  \item{starting}{
list with starting values for all unknowns. Compatibility checks with \code{prior} are minimal and incompatible values may result in crashes.
}
  \item{debug}{
list with debug settings. Can be used to turn off parts of MCMC.  
}
}
\details{
This implements the following model (in stacked vector form):
    \deqn{ y = X \beta + w + \epsilon, }
where \eqn{y} is a \eqn{n}-dimensional vector of outcomes, \eqn{X} is a matrix of covariates of dimension \eqn{(n, p)} associated to coefficients \eqn{\beta}, \eqn{w} is a \eqn{n}-dimensional vector storing the realization of a spatial multivariate Gaussian tree \eqn{w(\cdot) \sim SpamTree_{G}(0, C_{\theta})} where \eqn{G} is a treed directed acyclic graph, and where \eqn{C_{\theta}(s, s')} is a matrix-valued non-separable cross-covariance function on latent dimensions (see Peruzzi and Dunson (2021), equation 18, and \code{\link{CrossCovarianceAG10}}) where \eqn{\theta} is a vector of unknown parameters.
SpamTrees Gaussian processes are a scalable alternative to a spatial multivariate GP. Conditional independence across domain locations is assumed to be determined by the treed graph \eqn{G}, whose sparsity enables more efficient computations for the Gibbs sampler computed with this function. The graph architecture can be customized using inputs of the \code{spamtree} function. The example below computes SpamTrees on univariate data. A vignette exists with bivariate misaligned spatial data.
}
\value{
\item{coords}{reordered spatial coordinates}
\item{coordsinfo}{reordered spatial coordinates plus partitioning information.}
\item{mv_id}{reordered outcome IDs.}
\item{w_mcmc}{posterior sample of the spatial random effect. This is a list of length \code{mcmc$thin} whose elements are \code{n}-dimensional vectors of multivariate spatial random effects whose \eqn{q} margins are listed in \code{mv_id} as output here.}
\item{yhat_mcmc}{posterior predictive sample. This is a list of length \code{mcmc$thin} whose elements are \code{n}-dimensional vectors of predictions whose \eqn{q} margins are listed in \code{mv_id} as output here.}
\item{beta_mcmc}{array of size \code{c(p, mcmc$keep, q)} with posterior samples of the regression coefficients on each outcome. Example: \code{beta_mcmc[2,,1]} is the posterior sample for the second regressor on the first outcome.}
\item{tausq_mcmc}{matrix with posterior samples of the \eqn{q} nuggets, one for each outcome.}
\item{theta_mcmc}{matrix with posterior samples of the cross-covariance parameters. These include the latent distance between outcomes which may be poorly identifiable.}
\item{mcmc_time}{elapsed clock time for MCMC.}
}

\references{
Peruzzi, M. and Dunson, D. B. (2021)
  Spatial Multivariate Trees for Big Data Bayesian Regression. \url{https://arxiv.org/abs/2012.00943}
  
Vihola, M. (2012) Robust adaptive Metropolis algorithm with coerced acceptance rate. \emph{Statistics and Computing}, 22:997-1008. \doi{10.1007/s11222-011-9269-5}
}
\author{
  Michele Peruzzi \email{michele.peruzzi@duke.edu}, \cr
  David B. Dunson \email{dunson@duke.edu}
}

\examples{

# toy example with tiny dataset and short MCMC
# on a univariate outcome

library(magrittr)
library(dplyr)
library(ggplot2)
library(spamtree)

set.seed(2021)

SS <- 15 
n <- SS^2 # total n. locations, including missing ones

coords <- data.frame(Var1=runif(n), Var2=runif(n)) \%>\%
  as.matrix()

# generate data
sigmasq <- 2.3
phi <- 6
tausq <- .1
B <- c(-1,.5,1)

CC <- sigmasq * exp(-phi * as.matrix(dist(coords)))
LC <- t(chol(CC))
w <- LC \%*\% rnorm(n)
p <- length(B)
X <- rnorm(n * p) \%>\% matrix(ncol=p)
y <- X \%*\% B + w + tausq^.5 * rnorm(n)

set_missing <- rbinom(n, 1, 0.1)

simdata <- data.frame(coords,
                      y_full = y,
                      w_latent = w) \%>\%
  mutate(y_observed = ifelse(set_missing==1, NA, y_full))

# MCMC setup
mcmc_keep <- 300
mcmc_burn <- 300
mcmc_thin <- 1

ybar <- mean(y, na.rm=TRUE)

# fit spamtree with defaults
spamtree_done <- spamtree(y - ybar, X, coords, 
          mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin), 
          num_threads = 1)

# predictions
y_out <- spamtree_done$yhat_mcmc \%>\% 
  abind::abind(along=3) \%>\% `[`(,1,) \%>\% 
  add(ybar) \%>\% apply(1, mean)
w_out <- spamtree_done$w_mcmc \%>\% 
  abind::abind(along=3) \%>\% `[`(,1,) \%>\% 
  apply(1, mean)

outdf <- spamtree_done$coordsinfo \%>\% 
  cbind(data.frame(w_spamtree = w_out, 
                   y_spamtree = y_out)) \%>\%
  left_join(simdata)

# plot predictions
pred_plot <- outdf \%>\% 
  ggplot(aes(Var1, Var2, color=y_spamtree)) +
  geom_point() +
  scale_color_viridis_c()

# plot latent process
latent_plot <- outdf \%>\% 
  ggplot(aes(Var1, Var2, color=w_spamtree)) +
  geom_point() + 
  scale_color_viridis_c()

# estimation of regression coefficients
plot(density(spamtree_done$beta_mcmc[1,,1]))
abline(v=B[1], col="red")


}
