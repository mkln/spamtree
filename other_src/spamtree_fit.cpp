#include "spamtree_model.h"
#include "multires_utils.h"
#include "interrupt_handler.h"

//[[Rcpp::export]]
Rcpp::List spamtree_mcmc(
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    const arma::uvec& res_is_ref,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::mat& set_unif_bounds_in,
    
    const arma::mat& start_w,
    const arma::vec& theta,
    const arma::vec& beta,
    const double& tausq,
    const double& sigmasq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int num_threads = 1,
    
    char use_alg='S',
    
    bool adapting=false,
    bool verbose=false,
    bool debug=false,
    bool printall=false,
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_sigmasq=true,
    bool sample_theta=true,
    bool sample_w=true,
    bool sample_predicts=true){
  
  omp_set_num_threads(num_threads);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  
  bool verbose_mcmc = printall;
  
  double tempr=1;
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  int q = Z.n_cols;
  
  int k;
  int npars;
  double dlim=0;
  Rcpp::Rcout << "d=" << d << " q=" << q << ".\n";
  Rcpp::Rcout << "Lower and upper bounds for priors:\n";
  arma::mat set_unif_bounds = set_unif_bounds_in;
  
  int sigmasq_mh = 1;
  arma::vec theta_start = theta;
  arma::mat mcmcsd_start = mcmcsd;
  
  if(d == 2){
    if(q > 2){
      npars = sigmasq_mh + 3;
    } else {
      if(q==1){
        sigmasq_mh = 1;
        npars = sigmasq_mh + 1;
        
      } else {
        npars = sigmasq_mh + 1;
      }
      
    }
  } else {
    if(q > 2){
      npars = sigmasq_mh+5;
    } else {
      npars = sigmasq_mh+3; // sigmasq + alpha + beta + phi
    }
  }
  
  if(sigmasq_mh == 0){
    theta_start = theta.subvec(0, npars-1);
    mcmcsd_start = mcmcsd.submat(0, 0, npars-1, npars-1);
  }
  
  k = q * (q-1)/2;
  Rcpp::Rcout << theta << endl;
  Rcpp::Rcout << "Number of pars: " << npars << " plus " << k << " for multivariate\n"; 
  npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)
  
  Rcpp::Rcout << set_unif_bounds << endl;
  
  SpamTree mtree = SpamTree();
  
  mtree = SpamTree("gaussian", y, X, Z, coords, blocking, res_is_ref,
                  
                  parents, children, layer_names, layer_gibbs_group,
                  indexing,
                  
                  start_w, beta, theta_start, 1.0/tausq, sigmasq,
                  
                  use_alg, num_threads,
                  verbose, debug);
  
  
  arma::mat b_mcmc = arma::zeros(X.n_cols, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(1, mcmc_keep);
  
  arma::mat theta_mcmc = arma::zeros(npars, mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_keep);
  
  arma::vec grps = arma::unique(layer_gibbs_group);
  int n_res = grps.n_elem;
  arma::field<arma::cube> eta_rpx_mcmc(mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
//*** omp parallel for
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mtree.w.n_rows, q);
    yhat_mcmc(i) = arma::zeros(mtree.y.n_rows, 1);
    eta_rpx_mcmc(i) = arma::zeros(coords.n_rows, q, n_res);
  }
  
  mtree.get_loglik_comps_w( mtree.param_data );
  mtree.get_loglik_comps_w( mtree.alter_data );
  
  arma::vec param = mtree.param_data.theta;
  arma::vec predict_param = param;
  double current_loglik = tempr*mtree.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  }
  
  double logaccept;
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  int msaved = 0;
  bool interrupted = false;
  
  
  MHAdapter adaptivemc(param.n_elem, mcmc, mcmcsd_start);
  
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  
  double ll_upd_msg;
  
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  try { 
    for(m=0; m<mcmc; m++){
      
      mtree.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          mtree.predicting = true;
        } 
      }
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      ll_upd_msg = current_loglik;
      
      start = std::chrono::steady_clock::now();
      if(sample_w){
        mtree.deal_with_w();
        mtree.get_loglik_w(mtree.param_data);/*
        Rcpp::Rcout << mesh.param_data.logdetCi << " " 
                    << arma::accu(mesh.param_data.loglik_w_comps) 
                    << endl;
        mesh.get_loglik_comps_w_std(mesh.param_data);
        Rcpp::Rcout << mesh.param_data.logdetCi << " " 
                    << arma::accu(mesh.param_data.loglik_w_comps) 
                    << endl;*/
        current_loglik = tempr*mtree.param_data.loglik_w;
      }
      
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_w & verbose){
        Rcpp::Rcout << "[w] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
        //mesh.get_loglik_w(mesh.param_data);
        //Rcpp::Rcout << " >>>> CHECK : " << mesh.param_data.loglik_w << endl;
      }
    
      ll_upd_msg = current_loglik;
      start = std::chrono::steady_clock::now();
      bool accepted = true;
      if(sample_theta){
        //propos_count++;
        //propos_count_local++;
        adaptivemc.count_proposal();
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * arma::randn(npars), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        //Rcpp::Rcout << "new phi: " << new_param << endl;
        mtree.theta_update(mtree.alter_data, new_param); 
        mtree.get_loglik_comps_w( mtree.alter_data );
        
        bool accepted = !out_unif_bounds;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        new_loglik = tempr*mtree.alter_data.loglik_w;
        current_loglik = tempr*mtree.param_data.loglik_w;
        
        if(isnan(current_loglik)){
          Rcpp::Rcout << "At nan loglik: error. \n";
          throw 1;
        }
        
        //prior_logratio = calc_prior_logratio(k, new_param, param, npars, dlim);
        jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
        logaccept = new_loglik - current_loglik + //prior_logratio + 
          invgamma_logdens(new_param(0), 2, 2) -
          invgamma_logdens(mtree.param_data.theta(0), 2, 2) +
          jacobian;
        
        if(isnan(logaccept)){
          Rcpp::Rcout << new_param.t() << endl;
          Rcpp::Rcout << param.t() << endl;
          Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
          throw 1;
        }
        
        accepted = do_I_accept(logaccept);

        if(accepted){
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          adaptivemc.count_accepted();
          //accept_count++;
          //accept_count_local++;
          
          current_loglik = new_loglik;
          mtree.accept_make_change();
          param = new_param;
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us.\n"; 
          } 
        } else {
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")" << endl;
          }
        }
        //accept_ratio = accept_count/propos_count;
        //accept_ratio_local = accept_count_local/propos_count_local;
        adaptivemc.update_ratios();
        
        if(adapting){
          adaptivemc.adapt(par_huvtransf_fwd(param, set_unif_bounds), m); // **
        }
        

      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_theta & verbose){
        Rcpp::Rcout << "[theta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. ";
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      if((mtree.predicting == true) & sample_predicts){
        // tell predict() if theta has changed because if not, we can avoid recalculating
        bool need_update = arma::accu(abs(param - predict_param) > 1e-05);
        mtree.predict(need_update);
        predict_param = param;
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_beta){
        mtree.deal_with_beta();
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_beta & verbose){
        Rcpp::Rcout << "[beta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_sigmasq & (sigmasq_mh==0)){
        mtree.gibbs_sample_sigmasq();
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_beta & verbose){
        Rcpp::Rcout << "[sigmasq] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      
      
      start = std::chrono::steady_clock::now();
      if(sample_tausq){
        mtree.gibbs_sample_tausq();
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_tausq & verbose){
        Rcpp::Rcout << "[tausq] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                    << endl; 
      }
      
      if(printall){
        //Rcpp::checkUserInterrupt();
        interrupted = checkInterrupt();
        if(interrupted){
          throw 1;
        }
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick_mcmc ).count();
        adaptivemc.print(itertime, m);
        
        for(int pp=0; pp<npars; pp++){
          printf("theta%1d=%.4f ", pp, mtree.param_data.theta(pp));
        }
        printf("\n");
        
        tick_mcmc = std::chrono::steady_clock::now();
      } else {
        if((m>0) & (mcmc > 200)){
          if(!(m % (mcmc / 10))){
            //Rcpp::Rcout << paramsd << endl;
            //accept_count_local = 0;
            //propos_count_local = 0;
            
            interrupted = checkInterrupt();
            if(interrupted){
              throw 1;
            }
            end_mcmc = std::chrono::steady_clock::now();
            if(true){
              int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
              int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
              adaptivemc.print_summary(time_tick, time_mcmc, m, mcmc);
              
              tick_mcmc = std::chrono::steady_clock::now();
            }
          } 
        }
      }
      
      //save
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          tausq_mcmc.col(msaved) = 1.0 / mtree.tausq_inv;
          //sigmasq_mcmc.col(msaved) = mtree.sigmasq;
          b_mcmc.col(msaved) = mtree.Bcoeff;
          theta_mcmc.col(msaved) = mtree.param_data.theta;
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mtree.w;
          yhat_mcmc(msaved) = mtree.X * mtree.Bcoeff + mtree.Zw + pow(mtree.tausq_inv, -.5) * arma::randn(mtree.X.n_rows);
          eta_rpx_mcmc(msaved) = mtree.eta_rpx;
          
          msaved++;
        }
      }
      
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" 
                << mcmc_time
                <<  "ms]" << endl;

    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("eta_rpx") = eta_rpx_mcmc,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("block_ct_obs") = mtree.block_ct_obs,
      
      Rcpp::Named("indexing") = mtree.indexing,
      Rcpp::Named("parents_indexing") = mtree.parents_indexing,
      
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
    
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted." << endl;
    
    return Rcpp::List::create(
      Rcpp::Named("None") = arma::zeros(0)
    );
  }
  
}


//[[Rcpp::export]]
Rcpp::List spamtree_bfirls(
    const std::string& family,
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    const arma::uvec& res_is_ref,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::mat& start_w,
    const arma::vec& theta,
    const arma::vec& beta,
    const double& tausq,
    const double& sigmasq,
    
    int maxiter = 100,
    
    int num_threads = 1,
    
    
    bool verbose=false,
    bool debug=false,
    bool printall=false,
    
    bool sample_beta=true,
    bool sample_w=true,
    bool sample_predicts=true){
  
  omp_set_num_threads(num_threads);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();

  std::chrono::steady_clock::time_point tick = std::chrono::steady_clock::now();
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  int q = Z.n_cols;
  
  int k;
  int npars;
  double dlim=0;
  Rcpp::Rcout << "d = " << d << ", q = " << q << ".\n";
  
  
  SpamTree mtree = SpamTree();
  
  mtree = SpamTree(family, y, X, Z, coords, blocking, res_is_ref,
                  
                  parents, children, layer_names, layer_gibbs_group,
                  indexing,
                  
                  start_w, beta, theta, 1.0/tausq, sigmasq,
                  
                  'I', num_threads,
                  verbose, debug);
  
  arma::vec delta_iter = arma::zeros(maxiter);
  //arma::mat w_iter = arma::zeros(coords.n_rows, maxiter);
  //arma::mat b_iter = arma::zeros(X.n_cols, maxiter);
  arma::vec Bcoeff_previous = arma::zeros(X.n_cols);
  arma::vec w_previous = arma::ones(mtree.n * Z.n_cols);
  
  
  mtree.get_loglik_comps_w( mtree.param_data );
      
  bool interrupted = false;
  
  Rcpp::Rcout << "Running BF-IRLS for a max " << maxiter << " iterations." << endl;
  double check_start = 0;
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  try { 
    for(m=0; m<maxiter; m++){
      
      if(printall){
        tick = std::chrono::steady_clock::now();
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_w){
        mtree.deal_with_w();
      }
      
      end = std::chrono::steady_clock::now();
      if(printall & sample_w & verbose){
        Rcpp::Rcout << "[w] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
        //mesh.get_loglik_w(mesh.param_data);
        //Rcpp::Rcout << " >>>> CHECK : " << mesh.param_data.loglik_w << endl;
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_beta){
        mtree.deal_with_beta();
      }
      end = std::chrono::steady_clock::now();
      if(printall & sample_beta & verbose){
        Rcpp::Rcout << "[beta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      
      if(printall){
        //Rcpp::checkUserInterrupt();
        interrupted = checkInterrupt();
        if(interrupted){
          throw 1;
        }
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick ).count();
        
        printf("%5d-th iteration [ %dms ] \n", m+1, itertime);
        
        tick = std::chrono::steady_clock::now();
      } 
      
      if(mtree.w.has_nan()){
        Rcpp::Rcout << "NaN in w" << endl;
        throw 1;
      }
      
      arma::uvec bix = arma::find(mtree.block_ct_obs > 0) + 1;
      arma::uvec zerou = arma::zeros<arma::uvec>(1);
      arma::uvec beta_w_blocks_ix = arma::join_vert(zerou, bix);
      //arma::uvec converged = mtree.converged.elem(beta_w_blocks_ix);
      
      double beta_converged = pow(arma::max(abs(mtree.Bcoeff - Bcoeff_previous)), 2);
      arma::vec w_current = arma::vectorise(arma::trans(mtree.w.rows(mtree.na_ix_all)));
      double w_converged = pow(arma::max(abs(w_current - w_previous)), 2);
      
      w_previous = w_current;
      Bcoeff_previous = mtree.Bcoeff;

      if((beta_converged < 1e-4) & (w_converged < 1e-4)){
        printf("BF-IRLS converged.\n");
        
        break;
      }
      
      delta_iter(m) = w_converged;
      //w_iter.col(m) = mtree.w;
      //b_iter.col(m) = mtree.Bcoeff;
    }
    
    end_all = std::chrono::steady_clock::now();
    double alg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    
    if(m == maxiter){
      //Rcpp::Rcout << "Did not converge. Returning last values.\n" << endl;
      printf("Did not converge within the specified number of iterations. Returning last values.\n");
    }
    
    mtree.predicting = true;
    if(sample_predicts){
      mtree.predict(true);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("w") = mtree.w,
      //Rcpp::Named("w_iter") = w_iter,
      //Rcpp::Named("b_iter") = b_iter,
      Rcpp::Named("delta_iter") = delta_iter,
      
      Rcpp::Named("linear_predictor") = mtree.X * mtree.Bcoeff + mtree.Zw,
      Rcpp::Named("beta") = mtree.Bcoeff,
      Rcpp::Named("tausq") = 1.0/mtree.tausq_inv,
      Rcpp::Named("theta") = mtree.param_data.theta,
      Rcpp::Named("eta_rpx") = mtree.eta_rpx,
      
      Rcpp::Named("iter") = m,
      Rcpp::Named("alg_time") = alg_time/1000.0,
      
      Rcpp::Named("block_ct_obs") = mtree.block_ct_obs
    );
    
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double alg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    
    printf("BF-IRLS has been interrupted.\n");
    return Rcpp::List::create(
      Rcpp::Named("w") = mtree.w,
      Rcpp::Named("yhat") = mtree.X * mtree.Bcoeff + mtree.Zw,
      Rcpp::Named("beta") = mtree.Bcoeff,
      Rcpp::Named("tausq") = 1.0/mtree.tausq_inv,
      Rcpp::Named("theta") = mtree.param_data.theta,
      Rcpp::Named("eta_rpx") = mtree.eta_rpx,
      
      Rcpp::Named("iter") = m,
      Rcpp::Named("alg_time") = alg_time/1000.0,
      
      Rcpp::Named("block_ct_obs") = mtree.block_ct_obs
    );
  }
  
}


