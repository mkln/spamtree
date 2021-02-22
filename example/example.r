rm(list=ls())

library(magrittr)
library(tidyverse)
library(spamtree)

# example using gridded data so we can make raster images later.
# (ie just for convenience)

set.seed(1)

SS <- 50 # coordinate values for jth dimension 
n <- SS^2 # total n. locations, including missing ones

xlocs <- seq(0.0, 1, length.out=SS+2) %>% head(-1) %>% tail(-1)
coords <- expand.grid(xlocs, xlocs) %>% 
  as.data.frame() %>% 
  arrange(Var1, Var2)

c1 <- coords %>% mutate(mv_id=1)
c2 <- coords %>% mutate(mv_id=2)

coords <- bind_rows(c1, c2)

coords_q <- coords %>% dplyr::select(-mv_id)
cx <- coords_q %>% as.matrix()
ix <- 1:nrow(cx) - 1
mv_id <- coords$mv_id

q <- 2
sigma.sq <- 1
tau.sq <- c(.03, .1)
tausq_long <- rep(0, nrow(cx))
tausq_long[mv_id == 1] <- tau.sq[1]
tausq_long[mv_id == 2] <- tau.sq[2]

# some true values for the non-separable multivariate cross-covariance implemented here
phi_i <- c(1, 2)
phi   <- 5
alpha_v <- 5
beta_v <- .6

ai1 <- c(1, 1.5)
ai2 <- c(.1, .51)
thetamv <- c(alpha_v, beta_v, phi)

Dmat <- matrix(0, q, q)
Dmat[2,1] <- 1
Dmat[upper.tri(Dmat)] <- Dmat[lower.tri(Dmat)]

X <- cbind(rnorm(nrow(coords)), rnorm(nrow(coords)))
B <- c(-.9, .05)

# generate covariance matrix for full GP
system.time({
  CC <- spamtree::mvCovAG20107x(cx, mv_id-1, cx, mv_id-1, ai1, ai2, phi_i, thetamv, Dmat, T)
})
LC <- t(chol(CC))

# sample the outcomes at all locations
y_full <- X %*% B + LC %*% rnorm(nrow(cx)) + sqrt(tausq_long) * rnorm(nrow(cx))
rm(list=c("CC", "LC"))

# make some na: 0=na
# (this also creates some misalignment)
lna <- rep(1, nrow(coords)) 
lna[((coords_q$Var1 > .4) & (coords_q$Var1 < .9) & (coords_q$Var2 < .7) & (coords_q$Var2 > .4)) & (mv_id == 1)] <- NA
lna[((coords_q$Var1 > .2) & (coords_q$Var1 < .7) & (coords_q$Var2 < .7) & (coords_q$Var2 > .4)) & (mv_id == 2)] <- NA
y <- y_full * lna


simdata <- coords %>% cbind(y) %>% as.data.frame()


# prepare for spamtrees
mcmc_keep <- 1000
mcmc_burn <- 1000
mcmc_thin <- 2

# we sample w so we center and dont include intercept in X
ybar <- mean(y, na.rm=T)

# preprocessing before computing
stp <- spamtree::prebuild(y - ybar, # vector of outcomes
                       X, # matrix of covariates
                       coords[,1:2], # matrix of coordinates
                       mv_id,  # vector of outcome ids (with values in {1, ..., q})
                       
                       cell_size=c(5,5), # (approx) cell size 
                       K=c(2,2), # number of intervals along each axis for partitioning each cell recursively
                       start_level = 0, # starting level for recursion
                       tree_depth=Inf, # depth of the tree 
                       last_not_reference = F, # consider the top level of the tree as non-reference
                       limited_tree = F, # removes all recursions and makes a tree with delta=1 as in the manuscript
                       cherrypick_same_margin = T, # for non-reference: choose based on same margin (T) or based solely on distance (F)
                       cherrypick_group_locations = T, # for non-reference: T=[q-variate outcome at each location], F=[1-variate outcome at each (location, mv_id)]
                       use_alg = 'S', # not in use currently
                       mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin), # mcmc iterations. total = mcmc_burn + mcmc_thin*mcmc_keep
                       settings    = list(adapting=T, mcmcsd=.1, 
                                          verbose=F, debug=F, printall=F), # some additional mcmc settings
                       prior=list(btmlim=1e-3, toplim=15), # bounds for all priors of parameters being sampled via metropolis
                       debug       = list(sample_beta=T, sample_tausq=T, 
                                          sample_theta=T, 
                                          sample_w=T, sample_predicts=T), # things can be turned off. the "starting" parameter -not used here- can be used to fix some things
                       num_threads = 10 # number of OMP threads
)

# run mcmc -- which will also compute predictions at the missing locations
spamtree_time <- system.time({
  set.seed(1)
  spamtree_done <- spamtree::spamtree(model_data=stp)
})

# predictions
y_out <- spamtree_done$yhat_mcmc %>% meshgp::list_mean() %>% add(ybar)
w_out <- spamtree_done$w_mcmc %>% meshgp::list_mean()

outdf <- spamtree_done$model_data$coords_blocking %>% 
  rename(mv_id = sort_mv_id) %>%
  cbind(data.frame(w_mvmr = w_out, 
                   y_mvmr = y_out))

# plot predictions
outdf %>% 
  ggplot(aes(Var1, Var2, fill=y_mvmr)) +
  geom_raster() + 
  facet_grid(~mv_id) +
  scale_fill_viridis_c()

# plot latent process
outdf %>% 
  ggplot(aes(Var1, Var2, fill=w_mvmr)) +
  geom_raster() + 
  facet_grid(~mv_id) +
  scale_fill_viridis_c()

results_df <- simdata %>% full_join(outdf %>% dplyr::select(contains("Var"), mv_id, w_mvmr, y_mvmr))

results_df %>% filter(!complete.cases(.)) %>% with((y_mvmr - y_full)^2) %>% mean() %>% sqrt()

# observed data v full data v predictions
results_df %>% 
  dplyr::select(contains("Var"), y_full, y, mv_id, y_mvmr) %>%
  gather(variable, value, -Var1, -Var2, -mv_id) %>%
  ggplot(aes(Var1, Var2, fill=value)) +
  geom_raster() + 
  scale_fill_viridis_c() + 
  facet_grid(mv_id~variable) +
  theme(legend.position="none")


