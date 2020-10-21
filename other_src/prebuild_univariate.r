
prebuild <- function(y, X, Z, coords, 
                     mv_id = rep(1, length(y)),
                     cell_size=25, K=rep(2, ncol(coords)),
                     max_res = Inf,
                     last_not_reference = T,
                     
                     mcmc        = list(keep=1000, burn=0, thin=1),
                     family      = "gaussian",
                     num_threads = 4,
                     use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                     settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F),
                     prior       = list(set_unif_bounds=NULL, 
                                        toplim = 1e5,
                                        btmlim = 1e-5),
                     starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                     debug       = list(sample_beta=T, sample_tausq=T, 
                                        sample_sigmasq=T, sample_theta=T, 
                                        sample_w=T, sample_predicts=T,
                                        family="gaussian")
){
  
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  
  if(F){
    coords <- coords_q
    mv_id <- matrix(1, nrow=n) %x% 1:q
    Z <- X
    cell_size=16; K=rep(2, ncol(coords)); max_res <- Inf
    mcmc        = list(keep=1000, burn=0, thin=1);
    num_threads = 4;
    use_alg     = 'S'; #S: standard, P: using residual process ortho decomp, R: P with recursive functions
    settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F);
    prior       = list(set_unif_bounds=NULL);
    starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL);
    debug       = list(sample_beta=T, sample_tausq=T, 
                       sample_sigmasq=T, sample_theta=T, 
                       sample_w=T, sample_predicts=T)
    
  }
  
  # init
  cat(" Bayesian Spatial Univariate Tree \n
      
          :   :  :   :
           \\ /    \\ /
            o      o
              \\   /
                o\n\n
\n
Building...")
  
  if(1){
    mcmc_keep <- mcmc$keep
    mcmc_burn <- mcmc$burn
    mcmc_thin <- mcmc$thin
    
    mcmc_adaptive    <- settings$adapting
    mcmc_cache       <- settings$cache
    mcmc_cache_gibbs <- settings$cache_gibbs
    mcmc_verbose     <- settings$verbose
    mcmc_debug       <- settings$debug
    mcmc_printall    <- settings$printall
    
    sample_beta    <- debug$sample_beta
    sample_tausq   <- debug$sample_tausq
    sample_sigmasq <- debug$sample_sigmasq
    sample_theta   <- debug$sample_theta
    sample_w       <- debug$sample_w
    sample_predicts<- debug$sample_predicts
    
    dd             <- ncol(coords)
    p              <- ncol(X)
    q              <- ncol(Z)
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    #Mv = # ms levels
    
    if(is.null(starting$beta)){
      start_beta   <- rep(0, p)
    } else {
      start_beta   <- starting$beta
    }
    
    space_uni      <- (dd==2) & (q==1)
    space_biv      <- (dd==2) & (q==2) 
    space_mul      <- (dd==2) & (q >2)  
    stime_uni      <- (dd==3) & (q==1)
    stime_biv      <- (dd==3) & (q==2) 
    stime_mul      <- (dd==3) & (q >2)  
    
    if(is.null(starting$theta)){
      if(dd == 3){
        # spacetime
        if(q > 2){
          # multivariate
          start_theta <- c(10, 0.5, 10, 0.5, 10)
          start_theta <- c(start_theta, rep(1, k))
        } else {
          if(q == 2){
            # bivariate
            start_theta <- c(10, 0.5, 10, 1)
          } else {
            # univariate
            start_theta <- c(10, 0.5, 10)
          }
        }
      } else {
        # space
        if(q > 2){
          # multivariate
          start_theta <- c(10, 0.5, 10)
          start_theta <- c(start_theta, rep(1, k))
        } else {
          if(q == 2){
            # bivariate
            start_theta <- c(10, 1)
          } else {
            # univariate
            start_theta <- c(10)
          }
        }
      }
    } else {
      start_theta  <- starting$theta
    }
    
    
    if(is.null(prior$btmlim)){
      btmlim <- 1e-5
    } else {
      btmlim <- prior$btmlim
    }
    
    if(is.null(prior$toplim)){
      toplim <- 1e5
    } else {
      toplim <- prior$toplim
    }
    
    
    if(is.null(prior$set_unif_bounds)){
      if(dd == 3){
        # spacetime
        if(q > 2){
          # multivariate
          set_unif_bounds <- matrix(rbind(c(btmlim, toplim), # sigmasq
                                          c(btmlim, toplim), # alpha_1
                                          c(btmlim, 1-btmlim), # beta_1
                                          c(btmlim, toplim), # alpha_2
                                          c(btmlim, 1-btmlim), # beta_2
                                          c(btmlim, toplim)), # phi
                                    ncol=2)
        } else {
          # bivariate or univariate
          set_unif_bounds <- matrix(rbind(c(btmlim, toplim),
                                          c(btmlim, toplim),
                                          c(btmlim, 1-btmlim),
                                          c(btmlim, toplim)), ncol=2)
        }
      } else {
        # space
        if(q > 2){
          # multivariate
          set_unif_bounds <- matrix(rbind(c(btmlim, toplim),
                                          c(btmlim, toplim),
                                          c(btmlim, 1-btmlim),
                                          c(btmlim, toplim)), ncol=2)
          
        } else {
          # bivariate or univariate
          set_unif_bounds <- matrix(rbind(c(btmlim, toplim),
                                          c(btmlim, toplim)), ncol=2)
        }
      }
      
      if(q > 1){
        kk <- q * (q-1) / 2
        vbounds <- matrix(0, nrow=kk, ncol=2)
        
        if(q > 2){
          dlim <- sqrt(q+.0)
        } else {
          dlim <- toplim
        }
        vbounds[,1] <- btmlim;
        vbounds[,2] <- toplim - btmlim
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      }
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(2.01, 1)
    } else {
      sigmasq_ab <- prior$sigmasq
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)+1) * settings$mcmcsd
    } else {
      mcmc_mh_sd <- settings$mcmcsd
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- .1
    } else {
      start_tausq    <- starting$tausq
    }
    
    if(is.null(starting$sigmasq)){
      start_sigmasq <- 10
    } else {
      start_sigmasq  <- starting$sigmasq
    }
    
  }
  
  # data management
  if(is.null(colnames(X))){
    orig_X_colnames <- colnames(X) <- paste0('X_', 1:ncol(X))
  } else {
    orig_X_colnames <- colnames(X)
    colnames(X)     <- paste0('X_', 1:ncol(X))
  }
  
  if(is.null(colnames(Z))){
    orig_Z_colnames <- colnames(Z) <- paste0('Z_', 1:ncol(Z))
  } else {
    orig_Z_colnames <- colnames(Z)
    colnames(Z)     <- paste0('Z_', 1:ncol(Z))
  }
  
  #coords %<>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  colnames(coords)  <- paste0('Var', 1:dd)
  
  na_which <- ifelse(!is.na(y), 1, NA)
  simdata <- 1:nrow(coords) %>% cbind(coords, mv_id) %>% 
    cbind(y) %>% cbind(na_which) %>% 
    cbind(X) %>% cbind(Z) %>% as.data.frame()
  colnames(simdata)[1] <- "ix"
  
  simdata %<>% arrange(!!!syms(paste0("Var", 1:dd)))
  colnames(simdata)[dd + (2:4)] <- c("mv_id", "y", "na_which")
  
  coords <- simdata %>% dplyr::select(contains("Var"))
  simdata %<>% mutate(type="obs")
  sort_ix     <- simdata$ix
  
  if(!is.matrix(coords)){
    coords %<>% as.matrix()
  }
  
  # Domain partitioning and gibbs groups
  #system.time(coords_blocking <- coords %>% tessellation_axis_parallel(Mv, num_threads) %>% cbind(na_which))
  #cell_size <- 25
  if(length(cell_size) == 1){
    axis_size <- round(cell_size^(1/dd))
  } else {
    axis_size <- cell_size
  }
  
  ###### partitioning
  coords <- simdata %>% dplyr::select(contains("Var"), ix) %>% as.matrix()
  na_which <- simdata$na_which
  axis_cell_size <- rep(axis_size, dd)
  #save(file="debug.RData", list=c("coords", "na_which", "axis_cell_size"))
  #load("debug.RData")
  
  cat("Partitioning into resolution layers.\n")
  ptime <- system.time(
    mgp_tree <- spamtree:::make_tree(coords, na_which, mv_id, axis_cell_size, K, max_res, last_not_reference)
  )
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking
  thresholds <- mgp_tree$thresholds
  res_is_ref <- mgp_tree$res_is_ref
  
  # debug
  #na_blocks <- coords_blocking %>% arrange(Var1, Var2) %>% `[`(is.na(na_which),) %$% block %>% unique()
  #ixna <- simdata %>% filter(is.na(na_which)) %>% pull(ix)
  
  q <- ncol(Z)
  start_w <- rep(0, q*nrow(simdata)) %>% matrix(ncol=q)
  
  coordsnames <- paste0("Var", 1:dd)
  simdata %<>% arrange(!!!syms(coordsnames), ix)
  
  coords_blocking %<>% arrange(!!!syms(coordsnames), ix)
  
  mv_group_ix <- coords_blocking %>% 
    group_by(!!!syms(paste0("Var", 1:dd))) %>% 
    group_indices()
  
  coords_blocking %<>%  mutate(gix = mv_group_ix) %>%
    group_by(block) %>% 
    mutate(gix_block = as.numeric(factor(gix))) %>% as.data.frame()
  cx_all <- coords_blocking %>% dplyr::select(!!!syms(coordsnames)) %>% as.matrix()
  gix_block   <- coords_blocking$gix_block
  
  y           <- simdata$y %>% matrix(ncol=1)
  X           <- simdata %>% dplyr::select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  Z           <- simdata %>% dplyr::select(contains("Z_")) %>% as.matrix()
  colnames(Z) <- orig_Z_colnames
  na_which    <- simdata$na_which
  
  blocking <- coords_blocking$block
  
  block_info <- coords_blocking %>% mutate(color = res) %>%
    dplyr::select(block, color) %>% unique()
  
  
  ###### building graph
  block_ct_obs_df <- simdata %>% dplyr::select(ix, na_which) %>% 
    left_join(coords_blocking %>% dplyr::select(ix, block), by=c("ix"="ix")) %>%
    group_by(block) %>% summarise(perc_avail = sum(na_which, na.rm=T)/n(), .groups="drop")
  non_empty_blocks <- block_ct_obs_df[block_ct_obs_df$perc_avail>0, "block"] %>% pull(block)
  
  
  #save.image(file="temp.RData")
  
  cat("Building graph.\n")
  gtime <- system.time({
    parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  })
  #cat("Graph total time: ", as.numeric(gtime["elapsed"]), "\n")
  #npars <- parents_children$parents %>% lapply(length) %>% unlist()
  #hist(npars)
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  return(list(
    y=y, 
    X=X, 
    Z=Z, 
    mv_id=mv_id,
    cx_all=cx_all, 
    blocking=blocking,
    gix_block=gix_block,
    thresholds=thresholds,
    res_is_ref=res_is_ref,
    parents=parents, 
    children=children, 
    block_names=block_names, 
    block_groups=block_groups,
    indexing=indexing,
    set_unif_bounds=set_unif_bounds,
    start_w=start_w, 
    start_theta=start_theta,
    start_beta=start_beta,
    start_tausq=start_tausq,
    start_sigmasq=start_sigmasq,
    mcmc_mh_sd=mcmc_mh_sd,
    mcmc_keep=mcmc_keep, mcmc_burn=mcmc_burn, mcmc_thin=mcmc_thin,
    num_threads=num_threads,
    use_alg=use_alg,
    mcmc_adaptive=mcmc_adaptive, 
    mcmc_verbose=mcmc_verbose, 
    mcmc_debug=mcmc_debug,
    mcmc_printall=mcmc_printall,
    sample_beta=sample_beta, 
    sample_tausq=sample_tausq, 
    sample_sigmasq=sample_sigmasq, 
    sample_theta=sample_theta,
    sample_w=sample_w, 
    sample_predicts=sample_predicts,
    family=family,
    coords_blocking=coords_blocking,
    sort_ix=sort_ix))
  
}

