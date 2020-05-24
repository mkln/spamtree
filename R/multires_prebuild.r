spamtree_prebuild <- function(y, X, Z, coords, 
                              cell_size=25, K=rep(2, ncol(coords)),
                              mcmc        = list(keep=1000, burn=0, thin=1),
                              num_threads = 4,
                              use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                              settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F),
                              prior       = list(set_unif_bounds=NULL),
                              starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                              debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T, sample_predicts=T)
                              ){
                                
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.

  # init
  cat(" Bayesian Spatial Multiresolution Tree (with NN predictions)\n
      
          :   :  :   :
           \\ /    \\ /
            o      o
              \\   /
                o\n\n
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
    
    if(is.null(settings$saving)){
      saving         <- F
    } else {
      saving         <- settings$saving
    }
    
    rfc_dependence <- settings$reference_full_coverage
    
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
    
    space_uni      <- (q==1) & (dd==2)
    space_mul      <- (q >1) & (dd==2)
    stime_uni      <- (q==1) & (dd==3)
    stime_biv      <- (q==2) & (dd==3) 
    stime_mul      <- (q >2) & (dd==3) 
    
    if(is.null(starting$theta)){
      if(space_uni){
        start_theta <- 10
      } 
      if(space_mul || stime_mul){
        start_theta <- c(10, 0.5, 10, 0.5, 10)
        start_theta <- c(start_theta, rep(1, k))
      }
      if(stime_uni){
        start_theta <- c(10, 0.5, 10)
      }
      if(stime_biv){
        start_theta <- c(10, 0.5, 10, 1)
      }
    } else {
      start_theta  <- starting$theta
    }
    
    
    toplim <- 1e5
    btmlim <- 1e-5
    if(is.null(prior$set_unif_bounds)){
      if(space_uni){
        set_unif_bounds <- matrix(rbind(
          c(btmlim, toplim), 
          c(btmlim, toplim)), ncol=2)
      } else {
        if(stime_uni){
          set_unif_bounds <- matrix(rbind(c(btmlim, toplim),
                                          c(btmlim, 1e5), 
                                          c(btmlim, 1-btmlim), 
                                          c(btmlim, 1e5)), ncol=2)
        }
      }
    } else {
      set_unif_bounds <- prior$set_unif_bounds
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
  
  coords %<>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  colnames(coords)  <- paste0('Var', 1:dd)
  
  na_which <- ifelse(!is.na(y), 1, NA)
  simdata <- 1:nrow(coords) %>% cbind(coords) %>% 
    cbind(y) %>% cbind(na_which) %>% 
    cbind(X) %>% cbind(Z) %>% as.data.frame()
  colnames(simdata)[1] <- "ix"
  if(dd == 2){
    simdata %<>% arrange(Var1, Var2)
    coords <- simdata %>% select(Var1, Var2)
    colnames(simdata)[4:5] <- c("y", "na_which")
  } else {
    simdata %<>% arrange(Var1, Var2, Var3)
    coords <- simdata %>% select(Var1, Var2, Var3)
    colnames(simdata)[5:6] <- c("y", "na_which")
  }
  
  simdata %<>% mutate(type="obs")
  sort_ix     <- simdata$ix
  
  if(!is.matrix(coords)){
    coords %<>% as.matrix()
  }
  
  # Domain partitioning and gibbs groups
  #system.time(coords_blocking <- coords %>% tessellation_axis_parallel(Mv, num_threads) %>% cbind(na_which))
  #cell_size <- 25
  axis_size <- round(cell_size^(1/dd))
  
  ###### partitioning
  coords <- simdata %>% dplyr::select(contains("Var"), ix) %>% as.matrix()
  na_which <- simdata$na_which
  axis_cell_size <- rep(axis_size, dd)
  #save(file="debug.RData", list=c("coords", "na_which", "axis_cell_size"))
  
  #load("debug.RData")
  cat("Partitioning into resolution layers.\n")
  ptime <- system.time(
    mgp_tree <- make_tree(coords, na_which, axis_cell_size, K)
  )
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking
  thresholds <- mgp_tree$thresholds
  
  # debug
  #na_blocks <- coords_blocking %>% arrange(Var1, Var2) %>% `[`(is.na(na_which),) %$% block %>% unique()
  #ixna <- simdata %>% filter(is.na(na_which)) %>% pull(ix)
  
  
  q <- ncol(Z)
  start_w <- rep(0, q*nrow(simdata)) %>% matrix(ncol=q)
  
  if(dd == 2){
    simdata %<>% arrange(Var1, Var2)
    coords_blocking %<>% arrange(Var1, Var2)
    cx_all <- coords_blocking %>% select(Var1, Var2) %>% as.matrix()
  } else {
    simdata %<>% arrange(Var1, Var2, Var3)
    coords_blocking %<>% arrange(Var1, Var2, Var3)
    cx_all <- coords_blocking %>% select(Var1, Var2, Var3) %>% as.matrix()
  }
  
  y           <- simdata$y %>% matrix(ncol=1)
  X           <- simdata %>% select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  Z           <- simdata %>% select(contains("Z_")) %>% as.matrix()
  colnames(Z) <- orig_Z_colnames
  na_which    <- simdata$na_which
  
  blocking <- coords_blocking$block
  
  block_info <- coords_blocking %>% mutate(color = res) %>%
    select(block, color) %>% unique()
  
  
  ###### building graph
  block_ct_obs_df <- simdata %>% dplyr::select(ix, na_which) %>% 
    left_join(coords_blocking %>% dplyr::select(ix, block), by=c("ix"="ix")) %>%
    group_by(block) %>% summarise(perc_avail = sum(na_which, na.rm=T)/n()) 
  non_empty_blocks <- block_ct_obs_df[block_ct_obs_df$perc_avail>0, "block"] %>% pull(block)
  
  cat("Building graph.\n")
  gtime <- system.time({
    parents_children <- multires_graph(parchi_map %>% as.matrix(), non_empty_blocks)
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
    cx_all=cx_all, 
    blocking=blocking,
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
    sort_ix=sort_ix))
  
}