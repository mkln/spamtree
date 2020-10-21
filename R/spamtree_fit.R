
#' @export
spamtree <- function(y, X, coords, mv_id = rep(1, length(y)),
                     cell_size=25, K=rep(2, ncol(coords)),
                     max_depth = Inf, use_leaves = T,
                     tree_recursive = T,
                     
                     iter        = list(keep=1000, burn=0, thin=1),
                     num_threads = 4,
                     use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                     settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F),
                     prior       = list(set_unif_bounds=NULL),
                     starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                     debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T, sample_predicts=T),
                     model_data  = NULL
){
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  if(is.null(model_data)){
    model_data <- prebuild(y, X, coords, mv_id, cell_size, K, 
                              max_depth, use_leaves, !tree_recursive,
                              iter, 
                           num_threads, use_alg, settings, prior, starting, debug)
  }
  
  list2env(model_data, environment())

    comp_time <- system.time({
      results <- spamtree:::spamtree_mv_mcmc(
        y, X, Z, 
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
        mcmc_verbose, mcmc_debug, 
        mcmc_printall,
        
        sample_beta, sample_tausq, sample_theta,
        sample_w, sample_predicts) 
    })
    
    list2env(results, environment())
    
    return(list(coords    = cx_all,
                mv_id = mv_id,
                beta_mcmc    = beta_mcmc,
                tausq_mcmc   = tausq_mcmc,
                theta_mcmc   = theta_mcmc,
                
                w_mcmc    = w_mcmc,
                yhat_mcmc = yhat_mcmc,
                
                indexing = indexing,
                parents_indexing = parents_indexing,
                
                runtime_all   = comp_time,
                runtime_mcmc  = mcmc_time,
                
                paramsd = paramsd,
                #block_names = block_names,
                #block_is_reference = block_is_reference,
                #u_by_block_groups = u_by_block_groups,
                #n_actual_groups = n_actual_groups,
                #Ciblocks = Ciblocks,
                #Hblocks = Hblocks,
                #ImHinv= ImHinv,
                #Riblocks = Riblocks,
                
                model_data = model_data
                ))
}

Cinv_ <- function(y, X, coords, mv_id = rep(1, length(y)),
                     cell_size=25, K=rep(2, ncol(coords)),
                     max_depth = Inf, use_leaves = T,
                     tree_recursive = T,
                     
                     iter        = list(keep=1000, burn=0, thin=1),
                     num_threads = 4,
                     use_alg     = 'S', #S: standard, P: using residual process ortho decomp, R: P with recursive functions
                     settings    = list(adapting=T, mcmcsd=.3, verbose=F, debug=F, printall=F),
                     prior       = list(set_unif_bounds=NULL),
                     starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                     debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T, sample_predicts=T),
                     model_data  = NULL
){
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  if(is.null(model_data)){
    model_data <- prebuild(y, X, coords, mv_id, cell_size, K, 
                           max_depth, use_leaves, !tree_recursive,
                           iter, 
                           num_threads, use_alg, settings, prior, starting, debug)
  }
  
  list2env(model_data, environment())
  
  comp_time <- system.time({
    results <- spamtree:::Cinv_builtin(
      y, X, Z, 
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
      mcmc_verbose, mcmc_debug, 
      mcmc_printall,
      
      sample_beta, sample_tausq, sample_theta,
      sample_w, sample_predicts) 
  })
  
  return(results)
}


#' @export
spamtree_dev <- function(model_data){
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.

  list2env(model_data, environment())
  
  comp_time <- system.time({
    results <- spamtree:::spamtree_mv_mcmc_devel(
      y, X, Z, 
      cx_all, 
      sort_mv_id,
      
      blocking, 
      gix_block,
      res_is_ref,
      
      parents, children, 
      F,
      
      block_names, block_groups,
      indexing_knots, indexing_obs,
      
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
      mcmc_verbose, mcmc_debug, 
      mcmc_printall,
      
      sample_beta, sample_tausq, sample_theta,
      sample_w, sample_predicts) 
  })
  
  list2env(results, environment())
  
  return(list(coords    = cx_all,
              mv_id = mv_id,
              beta_mcmc    = beta_mcmc,
              tausq_mcmc   = tausq_mcmc,
              theta_mcmc   = theta_mcmc,
              
              w_mcmc    = w_mcmc,
              yhat_mcmc = yhat_mcmc,
              
              indexing = indexing_knots,
              indexing_obs = indexing_obs,
              parents_indexing = parents_indexing,
              
              runtime_all   = comp_time,
              runtime_mcmc  = mcmc_time,
              
              Kxx_inv = Kxx_inv,
              w_cond_mean_K = w_cond_mean_K,
              Kcc = Kcc,
              Kxc = Kxc,
              Rcc_invchol = Rcc_invchol,
              
              #block_names = block_names,
              #block_is_reference = block_is_reference,
              #u_by_block_groups = u_by_block_groups,
              #n_actual_groups = n_actual_groups,
              #Ciblocks = Ciblocks,
              #Hblocks = Hblocks,
              #ImHinv= ImHinv,
              #Riblocks = Riblocks,
              
              model_data = model_data
  ))
}

