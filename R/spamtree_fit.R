spamtree <- function(y, x, coords, 
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
                     mcmc        = list(keep=1000, burn=0, thin=1),
                     num_threads = 4,
                     verbose = FALSE,
                     settings    = list(adapting=TRUE, mcmcsd=.01, debug=FALSE, printall=FALSE),
                     prior       = list(set_unif_bounds=NULL, btmlim=NULL, toplim=NULL, vlim=NULL),
                     starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                     debug       = list(sample_beta=TRUE, sample_tausq=TRUE, 
                                        sample_theta=TRUE, 
                                        sample_w=TRUE, sample_predicts=TRUE)
){
  
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  use_alg <- 'S'
  
  if(1){
    mcmc_keep <- mcmc$keep
    mcmc_burn <- mcmc$burn
    mcmc_thin <- mcmc$thin
    
    mcmc_adaptive    <- settings$adapting
    mcmc_cache       <- settings$cache
    mcmc_cache_gibbs <- settings$cache_gibbs
    main_verbose     <- verbose 
    mcmc_verbose     <- verbose > 1
    mcmc_debug       <- settings$debug
    mcmc_printall    <- settings$printall
    
    if(main_verbose & mcmc_printall){
      cat(" Bayesian Spatial Multivariate Tree regression\n")
    }
    
    
    sample_beta    <- debug$sample_beta
    sample_tausq   <- debug$sample_tausq
    sample_sigmasq <- debug$sample_sigmasq
    sample_theta   <- debug$sample_theta
    sample_w       <- debug$sample_w
    sample_predicts<- debug$sample_predicts
    
    dd             <- ncol(coords)
    p              <- ncol(x)
    q              <- length(unique(mv_id))
    k              <- q * (q-1)/2
    nr             <- nrow(x)
    
    if(dd > 2){
      stop("Not implemented in domains of dimension d>2.")
    }

    elevation_3d <- FALSE
    
    
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
      
      
      
      set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
      set_unif_bounds[,1] <- btmlim
      set_unif_bounds[,2] <- toplim
      
      if(q>1){
        set_unif_bounds[2:q, 1] <- -toplim
      }
      
      if(n_cbase == 3){
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
      
      #start_theta[npars-1-el] <- .5
      #start_theta <- rep(2, npars) %>% c(rep(1, k))
      start_theta <- set_unif_bounds %>% apply(1, mean)
      
      
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
      beta_Vi <- diag(ncol(x)) * 1/100
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
  if(is.null(colnames(x))){
    orig_X_colnames <- colnames(x) <- paste0('X_', 1:ncol(x))
  } else {
    orig_X_colnames <- colnames(x)
    colnames(x)     <- paste0('X_', 1:ncol(x))
  }
  
  #coords %<>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  colnames(coords)  <- paste0('Var', 1:dd)
  
  na_which <- ifelse(!is.na(y), 1, NA)
  simdata <- 1:nrow(coords) %>% 
    cbind(coords, mv_id) %>% 
    cbind(y) %>% 
    cbind(na_which) %>% 
    cbind(x) %>% 
    cbind(Z) %>% 
    as.data.frame()
  colnames(simdata)[1] <- "ix"
  
  simdata %<>% dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
  colnames(simdata)[dd + (2:4)] <- c("mv_id", "y", "na_which")
  
  coords <- simdata %>% dplyr::select(dplyr::contains("Var"))
  simdata %<>% dplyr::mutate(type="obs")
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
  coords <- simdata %>% dplyr::select(dplyr::contains("Var"), .data$ix) %>% as.matrix()
  na_which <- simdata$na_which
  axis_cell_size <- rep(axis_size, dd)

  
  if(main_verbose) {
    cat("Building reference set.\n")
  }
  mgp_tree <- make_tree(coords, na_which, sort_mv_id, 
                        axis_cell_size, K, start_level, tree_depth, 
                        last_not_reference,
                        cherrypick_same_margin,
                        cherrypick_group_locations,
                        mvbias, main_verbose)

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
  simdata %<>% dplyr::arrange(!!!rlang::syms(coordsnames), .data$ix)
  
  coords_blocking %<>% dplyr::arrange(!!!rlang::syms(coordsnames), .data$ix)
  
  mv_group_ix <- coords_blocking %>% 
    dplyr::group_by(!!!rlang::syms(paste0("Var", 1:dd))) %>% 
    dplyr::group_indices()
  
  coords_blocking %<>% dplyr::mutate(gix = mv_group_ix) %>%
    dplyr::group_by(.data$block) %>% 
    dplyr::mutate(gix_block = as.numeric(factor(.data$gix))) %>% as.data.frame()
  cx_all <- coords_blocking %>% dplyr::select(!!!rlang::syms(coordsnames)) %>% as.matrix()
  gix_block   <- coords_blocking$gix_block
  
  y           <- simdata$y %>% matrix(ncol=1)
  x           <- simdata %>% dplyr::select(dplyr::contains("X_")) %>% as.matrix()
  colnames(x) <- orig_X_colnames
  Z           <- simdata %>% dplyr::select(dplyr::contains("Z_")) %>% as.matrix()
  
  na_which    <- simdata$na_which
  
  blocking <- coords_blocking$block
  
  block_info <- coords_blocking %>% 
    dplyr::mutate(color = .data$res) %>%
    dplyr::select(.data$block, .data$color) %>% unique()
  
  
  ###### building graph
  block_ct_obs_df <- simdata %>% 
    dplyr::select(.data$ix, .data$na_which) %>% 
    dplyr::left_join(coords_blocking %>%
                       dplyr::select(.data$ix, .data$block), by=c("ix"="ix")) %>%
    dplyr::group_by(.data$block) %>% 
    dplyr::summarise(perc_avail = sum(na_which, na.rm=TRUE)/dplyr::n(), .groups="drop")
  non_empty_blocks <- block_ct_obs_df[block_ct_obs_df$perc_avail>0, "block"] %>% 
    dplyr::pull(.data$block)
  
  
  if(main_verbose){
    cat("Building graph.\n")
  }
  
  if(limited_tree){
    parents_children <- make_edges_limited(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  } else {
    parents_children <- make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  }
  
  #cat("Graph total time: ", as.numeric(gtime["elapsed"]), "\n")
  #npars <- parents_children$parents %>% lapply(length) %>% unlist()
  #hist(npars)
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
    comp_time <- system.time({
      results <- spamtree_mv_mcmc(
        y, x, Z, 
        cx_all, 
        sort_mv_id,
        blocking, 
        gix_block,
        res_is_ref,
        
        parents, children, 
        limited_tree,
        
        block_names, block_groups,
        indexing,
        
        set_unif_bounds,
        
        start_w, 
        start_theta,
        start_beta,
        start_tausq,
        
        mcmc_mh_sd,
        
        mcmc_keep, mcmc_burn, mcmc_thin,
        
        num_threads,
        
        use_alg,
        
        mcmc_adaptive, 
        main_verbose,
        mcmc_verbose, mcmc_debug, 
        mcmc_printall,
        
        sample_beta, sample_tausq, sample_theta,
        sample_w, sample_predicts) 
    })
    
    returning <- list(coords    = cx_all,
                      coordsinfo = coords_blocking,
                      mv_id = mv_id) %>% 
      c(results)
    
    return(returning)
}
