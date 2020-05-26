#' @export
spamtree <- function(y, X, Z, coords, 
                       cell_size=25, K=rep(2, ncol(coords)),
                   mcmc        = list(keep=1000, burn=0, thin=1),
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
    model_data <- prebuild(y, X, Z, coords, cell_size, K, mcmc,
                                num_threads, use_alg, settings, prior, starting, debug)
  }

  list2env(model_data, environment())
  
  comp_time <- system.time({
    results <- spamtree:::mesh_multires_mcmc(y, X, Z, cx_all, blocking,
                                
                                parents, children, 
                                block_names, block_groups,
                                indexing,
                                
                                set_unif_bounds,
                                
                                start_w, 
                                start_theta,
                                start_beta,
                                start_tausq,
                                start_sigmasq,
                                
                                mcmc_mh_sd,
                                
                                mcmc_keep, mcmc_burn, mcmc_thin,
                                
                                num_threads,
                                
                                use_alg,
                                
                                mcmc_adaptive, 
                                mcmc_verbose, mcmc_debug, 
                                mcmc_printall,
                                
                                sample_beta, sample_tausq, sample_sigmasq, sample_theta,
                            sample_w, sample_predicts) 
  })
  
  list2env(results, environment())
  return(list(coords    = cx_all,
              
              beta_mcmc    = beta_mcmc,
              tausq_mcmc   = tausq_mcmc,
              theta_mcmc   = theta_mcmc,
              
              w_mcmc    = w_mcmc,
              yhat_mcmc = yhat_mcmc,
              
              eta_rpx = eta_rpx,
              
              runtime_all   = comp_time,
              runtime_mcmc  = mcmc_time,
              
              model_data = model_data))
  
}