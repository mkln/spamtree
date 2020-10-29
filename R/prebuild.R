


#' @export
prebuild <- function(y, X, coords, 
                     mv_id = rep(1, length(y)),
                     cell_size=25, 
                     K=rep(2, ncol(coords)),
                     start_level = 0,
                     tree_depth = 5,
                     last_not_reference = T,
                     limited_tree = F,
                     cherrypick_same_margin = T,
                     cherrypick_group_locations = T,
                     mvbias=0,
                     mcmc        = list(keep=1000, burn=0, thin=1),
                     num_threads = 4,
                     use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                     settings    = list(adapting=T, mcmcsd=.2, verbose=F, debug=F, printall=F),
                     prior       = list(set_unif_bounds=NULL, btmlim=NULL, toplim=NULL, vlim=NULL),
                     starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                     debug       = list(sample_beta=T, sample_tausq=T, 
                                        sample_theta=T, 
                                        sample_w=T, sample_predicts=T,
                                        family="gaussian")
){
  
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  
  if(F){
    coords <- coords_q
    #mv_id <- matrix(1, nrow=n) %x% 1:q
    #Z <- X
    cell_size=c(4,4); 
    K=rep(2, ncol(coords)); 
    start_level <- 2
    tree_depth <- 3
    last_not_reference <- F
    limited_tree <- F
    cherrypick_same_margin <- T
    cherrypick_group_locations <- F
    mvbias <- 0
    mcmc        = list(keep=1000, burn=0, thin=1);
    num_threads = 4;
    use_alg     = 'S'; #S: standard, P: using residual process ortho decomp, R: P with recursive functions
    settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F);
    prior       = list(set_unif_bounds=NULL);
    starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL);
    debug       = list(sample_beta=T, sample_tausq=T, sample_theta=T, sample_w=T, sample_predicts=T)
  }
  
  # init
  cat(" Bayesian Spatial Multivariate Tree\n
      
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
    q              <- length(unique(mv_id))
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    if(dd == 3){
      elevation_3d <- T
    } else {
      elevation_3d <- F
    }
    
    Z <- matrix(0, nrow=nrow(coords), q)
    for(j in 1:q){
      Z[mv_id==j, j] <- 1
    }
    colnames(Z)     <- paste0('Z_', 1:ncol(Z))
    
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
    
    if(is.null(prior$btmlim)){
      btmlim <- 1e-3
    } else {
      btmlim <- prior$btmlim
    }
    
    if(is.null(prior$toplim)){
      toplim <- 1e3
    } else {
      toplim <- prior$toplim
    }
    
    if(is.null(prior$vlim)){
      vlim <- toplim
    } else {
      vlim <- prior$vlim
    }
    
    
    if((dd == 2) || (dd==3 & elevation_3d)){
      el <- elevation_3d*1
      
      n_cbase <- ifelse(q > 2, 3, 1)
      npars <- 3*q + n_cbase + el
      
      start_theta <- rep(2, npars) %>% c(rep(1, k))
      
      set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
      set_unif_bounds[,1] <- btmlim
      set_unif_bounds[,2] <- toplim
      
      if(q>1){
        set_unif_bounds[2:q, 1] <- -toplim
      }
      
      if(n_cbase == 3){
        start_theta[npars-1-el] <- .5
        set_unif_bounds[npars-1-el,] <- c(btmlim, 1-btmlim)
      }
      
      if(q > 1){
        vbounds <- matrix(0, nrow=k, ncol=2)
        if(q > 2){
          dlim <- vlim#sqrt(q+.0)
        } else {
          dlim <- vlim#toplim
        }
        vbounds[,1] <- btmlim
        vbounds[,2] <- dlim - btmlim
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      }
    } else {
      # multidim (no time)
      # first is sigmasq, then the variables
      if(is.null(starting$theta)){
        start_theta <- rep(1, 1+ncol(coords))
      } else {
        start_theta  <- starting$theta
      }
      
      if(is.null(prior$set_unif_bounds)){
        set_unif_bounds <- matrix(0, ncol=2, nrow=1+ncol(coords))
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
      } else {
        set_unif_bounds <- prior$set_unif_bounds
      }
    }
    
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
  
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)) * settings$mcmcsd
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
  sort_mv_id <- simdata$mv_id
  
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
  #save(file="debug.RData", list=c("coords", "sort_mv_id", "na_which", "axis_cell_size", "K", "max_res", "last_not_reference",
  #                                "cherrypick_same_margin", "cherrypick_group_locations"))
  #load("debug.RData")
  max_res <- start_level + tree_depth
  
  cat("Building reference set.\n")
  ptime <- system.time(
    mgp_tree <- spamtree:::make_tree(coords, na_which, sort_mv_id, 
                                     axis_cell_size, K, start_level, tree_depth, 
                                     last_not_reference,
                                     cherrypick_same_margin,
                                     cherrypick_group_locations,
                                     mvbias)
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
  if(limited_tree){
    parents_children <- spamtree:::make_edges_limited(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  } else {
    parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  }
    
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
    limited_tree=limited_tree,
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
    
    sample_theta=sample_theta,
    sample_w=sample_w, 
    sample_predicts=sample_predicts,
    family=family,
    coords_blocking=coords_blocking,
    data=simdata,
    sort_mv_id=sort_mv_id,
    sort_ix=sort_ix))
  
}



#' @export
prebuild_dev_preds <- function(y, X, coords, 
                        mv_id = rep(1, length(y)),
                        cell_size=25, 
                        start_level = 0,
                        tree_depth = 5,
                        mcmc        = list(keep=1000, burn=0, thin=1),
                        num_threads = 4,
                        last_not_reference = NULL,
                        limited_tree = NULL,
                        K = NULL,
                        use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                        settings    = list(adapting=T, mcmcsd=.2, verbose=F, debug=F, printall=F),
                        prior       = list(set_unif_bounds=NULL, btmlim=NULL, toplim=NULL, vlim=NULL),
                        starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                        debug       = list(sample_beta=T, sample_tausq=T, 
                                           sample_theta=T, 
                                           sample_w=T, sample_predicts=T,
                                           family="gaussian")
){
  
  # cell_size = (approximate) number of location for each block
  if(!is.null(K)){
    warning("Setting K disabled, as K=c(2,2) in this quadtree")
  }
  K=c(2,2)
  
  if(F){
    cell_size <- c(4,4)
    K <- c(2,2)
    
    start_level = 2
    tree_depth = 3
    #coords <- coords_q
    mcmc        = list(keep=1000, burn=0, thin=1);
    num_threads = 4;
    use_alg     = 'S'; #S: standard, P: using residual process ortho decomp, R: P with recursive functions
    settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F);
    prior       = list(set_unif_bounds=NULL);
    starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL);
    debug       = list(sample_beta=T, sample_tausq=T, sample_theta=T, sample_w=T, sample_predicts=T)
  }
  
  # init
  cat(" Bayesian Spatial Multivariate Tree ~ development\n
      
          :   :  :   :
           \\ /    \\ /
            o      o
              \\   /
                o\n\n
\n
Building...")
  
  # processing part 1
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
    q              <- length(unique(mv_id))
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    Z <- matrix(0, nrow=nrow(coords), q)
    for(j in 1:q){
      Z[mv_id==j, j] <- 1
    }
    colnames(Z)     <- paste0('Z_', 1:ncol(Z))
    
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
    
    if(is.null(prior$btmlim)){
      btmlim <- 1e-5
    } else {
      btmlim <- prior$btmlim
    }
    
    if(is.null(prior$toplim)){
      toplim <- 1e2
    } else {
      toplim <- prior$toplim
    }
    
    if(is.null(prior$vlim)){
      vlim <- toplim
    } else {
      vlim <- prior$vlim
    }
    
    if(dd == 2){
      n_cbase <- ifelse(q > 2, 3, 1)
      npars <- 3*q + n_cbase
      
      start_theta <- rep(1, npars) %>% c(rep(1, k))
      
      set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
      set_unif_bounds[,1] <- btmlim
      set_unif_bounds[,2] <- toplim
      
      if(q>1){
        set_unif_bounds[2:q, 1] <- -toplim
      }
      
      if(n_cbase == 3){
        start_theta[npars-1] <- .5
        set_unif_bounds[npars-1,] <- c(btmlim, 1-btmlim)
      }
      
      if(q > 1){
        vbounds <- matrix(0, nrow=k, ncol=2)
        if(q > 2){
          dlim <- vlim#sqrt(q+.0)
        } else {
          dlim <- vlim#toplim
        }
        vbounds[,1] <- btmlim
        vbounds[,2] <- dlim - btmlim
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      }
    }
    
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)) * settings$mcmcsd
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
  
  # processing part 2
  if(1){
    if(is.null(colnames(X))){
      orig_X_colnames <- colnames(X) <- paste0('X_', 1:ncol(X))
    } else {
      orig_X_colnames <- colnames(X)
      colnames(X)     <- paste0('X_', 1:ncol(X))
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
    sort_mv_id <- simdata$mv_id
    
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
  }
  
  ###### partitioning
  simdata_all <- simdata
  simdata %<>% filter(complete.cases(y))
  simdata_missing <- simdata_all %>% filter(!complete.cases(y))
  
  coords <- simdata %>% dplyr::select(contains("Var"), ix) %>% as.matrix()
  na_which <- simdata$na_which
  sort_mv_id <- simdata$mv_id
  
  min_res <- start_level
  max_res <- start_level + tree_depth
  
  axis_cell_size <- axis_size #rep(axis_size, dd)
  #save(file="debug.RData", list=c("coords", "sort_mv_id", "na_which", "axis_cell_size", "K", "max_res", "last_not_reference",
  #                                "cherrypick_same_margin", "cherrypick_group_locations"))
  #load("debug.RData")
  
  cat("Building reference set.\n")
  ptime <- system.time(
    mgp_tree <- spamtree:::make_tree_devel(coords, na_which, 
                                            sort_mv_id, 
                                     axis_cell_size, K, max_res, min_res)
  )
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking 
  thresholds <- mgp_tree$thresholds
  res_is_ref <- mgp_tree$res_is_ref
  
  if(nrow(simdata_missing) > 0){
    # manage missing data
    simdata_managed <- spamtree:::manage_for_predictions(simdata_missing, coords_blocking, parchi_map) 
    
    simdata_missing <- simdata_managed$simdata_missing
    parchi_missing <- simdata_managed$pred_parchis
    
    coords_blocking %<>% bind_rows(simdata_missing[,colnames(coords_blocking)])
    
    parchi_knots <- parchi_map[,-ncol(parchi_map)]
    parchi_obs <- bind_rows(parchi_map[,(ncol(parchi_map)-1):ncol(parchi_map),drop=F], 
                            parchi_missing)
    parchi_map <- parchi_knots %>% left_join(parchi_obs)
    
    res_is_ref <- c(res_is_ref, 0)
  }
  
  # debug
  #na_blocks <- coords_blocking %>% arrange(Var1, Var2) %>% `[`(is.na(na_which),) %$% block %>% unique()
  #ixna <- simdata %>% filter(is.na(na_which)) %>% pull(ix)
  simdata_all %<>% right_join(coords_blocking %>% rename(mv_id = sort_mv_id) %>% dplyr::select(-ix))
  
  q <- ncol(Z)
  start_w <- rep(0, q*nrow(simdata)) %>% matrix(ncol=q)
  
  coordsnames <- paste0("Var", 1:dd)
  
  simdata_all %<>% arrange(!!!syms(coordsnames), ix)
  coords_blocking %<>% arrange(!!!syms(coordsnames), ix)
  
  #mv_group_ix <- coords_blocking %>% 
  #  group_by(!!!syms(paste0("Var", 1:dd))) %>% 
  #  group_indices()
  
  #coords_blocking %<>%  mutate(gix = mv_group_ix) %>%
  #  group_by(block) %>% 
  #  mutate(gix_block = as.numeric(factor(gix))) %>% as.data.frame()
  
  cx_all <- simdata_all %>% dplyr::select(!!!syms(coordsnames)) %>% as.matrix()
  gix_block   <- rep(1, nrow(simdata_all)) #$gix_block
  
  y           <- simdata_all$y %>% matrix(ncol=1)
  X           <- simdata_all %>% dplyr::select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  Z           <- simdata_all %>% dplyr::select(contains("Z_")) %>% as.matrix()
  
  na_which    <- simdata_all$na_which
  sort_mv_id <- simdata_all$mv_id
  
  blocking <- simdata_all$block
  
  can_be_child_reference <- simdata_all %>% filter(res < max(res)) %$% block %>% unique()
  can_be_child_observed <- simdata_all %>% group_by(block) %>%
    summarise(avail_perc = sum(!is.na(na_which))/n()) %>%
    filter(avail_perc > 0) %$% block %>% unique()
  can_be_child <- c(can_be_child_reference, can_be_child_observed)
  
  block_info <- simdata_all %>% mutate(color = res) %>%
    dplyr::select(block, color) %>% unique()

  ###### building graph
  cat("Building graph.\n")
  parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), can_be_child, res_is_ref)
  
  
  #cat("Graph total time: ", as.numeric(gtime["elapsed"]), "\n")
  #npars <- parents_children$parents %>% lapply(length) %>% unlist()
  #hist(npars)
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  
  predictable_blocks <- simdata_all %>% 
    dplyr::select(Var1, Var2, mv_id, na_which) %>% 
    rename(sort_mv_id=mv_id) %>%
    left_join(coords_blocking) %>% mutate(theknots = ifelse(res==max(res), 0, 1)) %>%
    group_by(block) %>% summarise(theknots=mean(theknots, na.rm=T), perc_availab=mean(!is.na(na_which))) %>%
    mutate(predict_here = ifelse(theknots==0, perc_availab<1, 0)) %>% arrange(block) %$% predict_here
  
  maxres <- max(coords_blocking$res)
  obs_res <- c(maxres, ifelse(nrow(simdata_missing) > 0, maxres-1, NULL))
  
  indexing_knots_ids <- coords_blocking %>% 
    mutate(theknots = ifelse(res %in% obs_res, 0, 1)) %$% theknots %>% split(blocking)
  indexing_knots <- list()
  indexing_obs <- list()
  for(i in 1:length(indexing)){
    indexing_knots[[i]] <- indexing[[i]][which(indexing_knots_ids[[i]] == 1)]
    indexing_obs[[i]] <- indexing[[i]][which(indexing_knots_ids[[i]] == 0)]
  }
  
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
    simdata=simdata,
    parents=parents, 
    children=children, 
    block_names=block_names, 
    block_groups=block_groups,
    indexing=indexing,
    indexing_knots=indexing_knots,
    indexing_obs=indexing_obs,
    set_unif_bounds=set_unif_bounds,
    start_w=start_w, 
    start_theta=start_theta,
    start_beta=start_beta,
    start_tausq=start_tausq,
    
    parchi_map=parchi_map,
    
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
    
    sample_theta=sample_theta,
    sample_w=sample_w, 
    sample_predicts=sample_predicts,
    family=family,
    coords_blocking=coords_blocking,
    sort_mv_id=sort_mv_id,
    sort_ix=sort_ix))
  
}



#' @export
prebuild_dev <- function(y, X, coords, 
                         mv_id = rep(1, length(y)),
                         cell_size=25, 
                         start_level = 0,
                         tree_depth = 5,
                         mcmc        = list(keep=1000, burn=0, thin=1),
                         num_threads = 4,
                         use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                         settings    = list(adapting=T, mcmcsd=.2, verbose=F, debug=F, printall=F),
                         prior       = list(set_unif_bounds=NULL, btmlim=NULL, toplim=NULL, vlim=NULL),
                         starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                         debug       = list(sample_beta=T, sample_tausq=T, 
                                            sample_theta=T, 
                                            sample_w=T, sample_predicts=T,
                                            family="gaussian"),
                         ...
){
  # cell_size = (approximate) number of location for each block
  if(!is.null(K)){
    warning("Setting K disabled, as K=c(2,2) in this quadtree")
  }
  K=c(2,2)
  
  if(F){
    cell_size <- c(2,2)
    K <- c(2,2)
    
    start_level = 0
    tree_depth = 5
    #coords <- coords_q
    mcmc        = list(keep=1000, burn=0, thin=1);
    num_threads = 4;
    use_alg     = 'S'; #S: standard, P: using residual process ortho decomp, R: P with recursive functions
    settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F);
    prior       = list(set_unif_bounds=NULL);
    starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL);
    debug       = list(sample_beta=T, sample_tausq=T, sample_theta=T, sample_w=T, sample_predicts=T)
  }
  
  # init
  cat(" Bayesian Spatial Multivariate Tree ~ development\n
      
          :   :  :   :
           \\ /    \\ /
            o      o
              \\   /
                o\n\n
\n
Building...")
  
  # processing part 1
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
    q              <- length(unique(mv_id))
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    Z <- matrix(0, nrow=nrow(coords), q)
    for(j in 1:q){
      Z[mv_id==j, j] <- 1
    }
    colnames(Z)     <- paste0('Z_', 1:ncol(Z))
    
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
    
    if(is.null(prior$vlim)){
      vlim <- toplim
    } else {
      vlim <- prior$vlim
    }
    
    if(dd == 2){
      n_cbase <- ifelse(q > 2, 3, 1)
      npars <- 3*q + n_cbase
      
      start_theta <- rep(1, npars) %>% c(rep(1, k))
      
      set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
      set_unif_bounds[,1] <- btmlim
      set_unif_bounds[,2] <- toplim
      
      if(q>1){
        set_unif_bounds[2:q, 1] <- -toplim
      }
      
      if(n_cbase == 3){
        start_theta[npars-1] <- .5
        set_unif_bounds[npars-1,] <- c(btmlim, 1-btmlim)
      }
      
      if(q > 1){
        vbounds <- matrix(0, nrow=k, ncol=2)
        if(q > 2){
          dlim <- vlim#sqrt(q+.0)
        } else {
          dlim <- vlim#toplim
        }
        vbounds[,1] <- btmlim
        vbounds[,2] <- dlim - btmlim
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      }
    }
    
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)) * settings$mcmcsd
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
  
  # processing part 2
  if(1){
    if(is.null(colnames(X))){
      orig_X_colnames <- colnames(X) <- paste0('X_', 1:ncol(X))
    } else {
      orig_X_colnames <- colnames(X)
      colnames(X)     <- paste0('X_', 1:ncol(X))
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
    sort_mv_id <- simdata$mv_id
    
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
  }
  
  ###### partitioning
  simdata_all <- simdata
  simdata %<>% filter(complete.cases(y))
  simdata_missing <- simdata_all %>% filter(!complete.cases(y))
  
  coords <- simdata %>% dplyr::select(contains("Var"), ix) %>% as.matrix()
  na_which <- simdata$na_which
  sort_mv_id <- simdata$mv_id
  
  min_res <- start_level
  max_res <- start_level + tree_depth
  
  axis_cell_size <- axis_size #rep(axis_size, dd)
  #save(file="debug.RData", list=c("coords", "sort_mv_id", "na_which", "axis_cell_size", "K", "max_res", "last_not_reference",
  #                                "cherrypick_same_margin", "cherrypick_group_locations"))
  #load("debug.RData")
  
  cat("Building reference set.\n")
  ptime <- system.time(
    mgp_tree <- spamtree:::make_tree_devel(coords, na_which, 
                                           sort_mv_id, 
                                           axis_cell_size, K, max_res, min_res)
  )
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking 
  thresholds <- mgp_tree$thresholds
  res_is_ref <- mgp_tree$res_is_ref
  
  # manage missing data
  simdata_missing %<>% spamtree:::manage_for_predictions(coords_blocking, parchi_map) %>%
    dplyr::rename(sort_mv_id = mv_id)
  
  coords_blocking %<>% bind_rows(simdata_missing[,colnames(coords_blocking)])
  
  # debug
  #na_blocks <- coords_blocking %>% arrange(Var1, Var2) %>% `[`(is.na(na_which),) %$% block %>% unique()
  #ixna <- simdata %>% filter(is.na(na_which)) %>% pull(ix)
  simdata_all %<>% right_join(coords_blocking %>% rename(mv_id = sort_mv_id) %>% dplyr::select(-ix))
  
  q <- ncol(Z)
  start_w <- rep(0, q*nrow(simdata)) %>% matrix(ncol=q)
  
  coordsnames <- paste0("Var", 1:dd)
  
  simdata_all %<>% arrange(!!!syms(coordsnames), ix)
  coords_blocking %<>% arrange(!!!syms(coordsnames), ix)
  
  #mv_group_ix <- coords_blocking %>% 
  #  group_by(!!!syms(paste0("Var", 1:dd))) %>% 
  #  group_indices()
  
  #coords_blocking %<>%  mutate(gix = mv_group_ix) %>%
  #  group_by(block) %>% 
  #  mutate(gix_block = as.numeric(factor(gix))) %>% as.data.frame()
  
  cx_all <- simdata_all %>% dplyr::select(!!!syms(coordsnames)) %>% as.matrix()
  gix_block   <- rep(1, nrow(simdata_all)) #$gix_block
  
  y           <- simdata_all$y %>% matrix(ncol=1)
  X           <- simdata_all %>% dplyr::select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  Z           <- simdata_all %>% dplyr::select(contains("Z_")) %>% as.matrix()
  
  na_which    <- simdata_all$na_which
  sort_mv_id <- simdata_all$mv_id
  
  blocking <- simdata_all$block
  
  block_info <- simdata_all %>% mutate(color = res) %>%
    dplyr::select(block, color) %>% unique()
  
  
  ###### building graph
  #block_ct_obs_df <- simdata %>% dplyr::select(ix, na_which) %>% 
  #  left_join(coords_blocking %>% dplyr::select(ix, block), by=c("ix"="ix")) %>%
  #  group_by(block) %>% summarise(perc_avail = sum(na_which, na.rm=T)/n(), .groups="drop")
  #non_empty_blocks <- block_ct_obs_df[block_ct_obs_df$perc_avail>0, "block"] %>% pull(block)
  non_empty_blocks <- block_info$block
  
  #save.image(file="temp.RData")
  
  cat("Building graph.\n")
  parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  
  
  #cat("Graph total time: ", as.numeric(gtime["elapsed"]), "\n")
  #npars <- parents_children$parents %>% lapply(length) %>% unlist()
  #hist(npars)
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  
  predictable_blocks <- simdata_all %>% 
    dplyr::select(Var1, Var2, mv_id, na_which) %>% 
    rename(sort_mv_id=mv_id) %>%
    left_join(coords_blocking) %>% mutate(theknots = ifelse(res==max(res), 0, 1)) %>%
    group_by(block) %>% summarise(theknots=mean(theknots, na.rm=T), perc_availab=mean(!is.na(na_which))) %>%
    mutate(predict_here = ifelse(theknots==0, perc_availab<1, 0)) %>% arrange(block) %$% predict_here
  
  
  indexing_knots_ids <- coords_blocking %>% 
    mutate(theknots = ifelse(res==max(res), 0, 1)) %$% theknots %>% split(blocking)
  indexing_knots <- list()
  indexing_obs <- list()
  for(i in 1:length(indexing)){
    indexing_knots[[i]] <- indexing[[i]][which(indexing_knots_ids[[i]] == 1)]
    indexing_obs[[i]] <- indexing[[i]][which(indexing_knots_ids[[i]] == 0)]
  }
  
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
    simdata=simdata,
    parents=parents, 
    children=children, 
    block_names=block_names, 
    block_groups=block_groups,
    indexing=indexing,
    indexing_knots=indexing_knots,
    indexing_obs=indexing_obs,
    set_unif_bounds=set_unif_bounds,
    start_w=start_w, 
    start_theta=start_theta,
    start_beta=start_beta,
    start_tausq=start_tausq,
    
    parchi_map=parchi_map,
    
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
    
    sample_theta=sample_theta,
    sample_w=sample_w, 
    sample_predicts=sample_predicts,
    family=family,
    coords_blocking=coords_blocking,
    sort_mv_id=sort_mv_id,
    sort_ix=sort_ix))
  
}