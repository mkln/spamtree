#include "spamtree_mv_model_devel.h"
#include "multires_utils.h"
#include "interrupt_handler.h"

#include <iostream>

//[[Rcpp::export]]
Rcpp::List spamtree_mv_mcmc_devel(
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& mv_id,
    
    const arma::uvec& blocking,
    const arma::uvec& gix_block,
    
    const arma::uvec& res_is_ref,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    bool limited_tree,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::mat& set_unif_bounds_in,
    
    const arma::mat& start_w,
    const arma::vec& theta,
    const arma::vec& beta,
    const double& tausq,
    
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

  /*
  if(d == 2){
    if(q > 2){
      npars = 1 + 3;
    } else {
      npars = 1 + 1;
    }
  } else {
    if(q > 2){
      npars = 1+5;
    } else {
      npars = 1+3; // sigmasq + alpha + beta + phi
    }
  }*/
  
  if(d == 2){
    int n_cbase = q > 2? 3: 1;
    npars = 3*q + n_cbase;
  }
  
  k = q * (q-1)/2;
  
  Rcpp::Rcout << "Number of pars: " << npars << " plus " << k << " for multivariate\n"; 
  npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)
  
  // metropolis search limits
  //arma::mat tsqi_unif_bounds = arma::zeros(q, 2);
  //tsqi_unif_bounds.col(0).fill(1e-5);
  //tsqi_unif_bounds.col(1).fill(1000-1e-5);
  
  arma::mat set_unif_bounds = set_unif_bounds_in;//arma::join_vert(set_unif_bounds_in, 
                                              //tsqi_unif_bounds);
  
  arma::mat metropolis_sd = mcmcsd;//arma::zeros(set_unif_bounds.n_rows, set_unif_bounds.n_rows);
  //metropolis_sd.submat(0, 0, npars-1, npars-1) = mcmcsd;
  //metropolis_sd.submat(npars, npars, npars+q-1, npars+q-1) = .1 * arma::eye(q, q);
  
  Rcpp::Rcout << set_unif_bounds << endl;
  
  SpamTreeMVdevel mtree = SpamTreeMVdevel();
  
  arma::vec start_w_vec = arma::zeros(y.n_elem);
  
  mtree = SpamTreeMVdevel("gaussian", y, X, Z, coords, mv_id, 
                     blocking, gix_block, res_is_ref,
                     
                     parents, children, limited_tree, 
                     layer_names, layer_gibbs_group,
                     indexing,
                     
                     start_w_vec, beta, theta, 1.0/tausq, 
                     
                     use_alg, num_threads,
                     verbose, debug);

  
  mtree.get_loglik_comps_w( mtree.param_data );
  mtree.get_loglik_comps_w( mtree.alter_data );
  //mtree.deal_with_w(true);
  //mtree.get_loglik_w(mtree.param_data);
   
  arma::vec param = mtree.param_data.theta;//arma::join_vert( , mtree.tausq_inv );
  arma::vec predict_param = param;
  double current_loglik = tempr*mtree.param_data.loglik_w;
  //if(verbose & debug){
  //  Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  //}
  
  arma::cube beta_mcmc = arma::zeros(X.n_cols, mcmc_keep, q);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(npars, mcmc_keep);
  
  arma::vec grps = arma::unique(layer_gibbs_group);
  int n_res = grps.n_elem;
  //arma::field<arma::cube> eta_rpx_mcmc(mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
  //*** omp parallel for
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mtree.w.n_rows, q);
    yhat_mcmc(i) = arma::zeros(mtree.y.n_rows, 1);
    //eta_rpx_mcmc(i) = arma::zeros(coords.n_rows, q, n_res);
  }
  
  
  
  double logaccept;
  
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  int msaved = 0;
  bool interrupted = false;
  
  MHAdapter adaptivemc(param.n_elem, mcmc, metropolis_sd);
  
  
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  Rcpp::Rcout << "Starting from "<< mtree.param_data.loglik_w << endl;
  Rcpp::Rcout << "check with "<< mtree.alter_data.loglik_w << endl;
  double ll_upd_msg;
  
  bool need_update = true;
    
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  //try { 
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
      //Rcpp::Rcout << "before updating w "<< mtree.param_data.loglik_w << endl;
      if(sample_w){
        mtree.deal_with_w(true);
        mtree.get_loglik_w(mtree.param_data);
        current_loglik = tempr*mtree.param_data.loglik_w;
      }
      
      //Rcpp::Rcout << "after updating w "<< mtree.param_data.loglik_w << endl;
      
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
      
      //Rcpp::Rcout << "before updating theta "<< mtree.param_data.loglik_w << endl;
      bool accepted = true;
      if(sample_theta){
        //propos_count++;
        //propos_count_local++;
        adaptivemc.count_proposal();
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * arma::randn(param.n_elem), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        //Rcpp::Rcout << "new phi: " << new_param << endl;
        
        arma::vec theta_proposal = new_param.subvec(0, npars-1);
        //arma::vec tausqi_proposal = new_param.subvec(npars, param.n_elem-1);
        //arma::vec tausqi_original = param.subvec(npars, param.n_elem-1);
        
        mtree.theta_update(mtree.alter_data, theta_proposal);
        bool acceptable = mtree.get_loglik_comps_w( mtree.alter_data );
        //double tsqi_ll_ratio = mtree.mh_tausq_loglik(tausqi_proposal, tausqi_original);
        
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
        
        prior_logratio = calc_prior_logratio(new_param, param);
        jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
        logaccept = new_loglik - current_loglik + 
          //prior_logratio + 
          //tsqi_ll_ratio + 
          //invgamma_logdens(new_param(0), 2, 2) -
          //invgamma_logdens(mtree.param_data.theta(0), 2, 2) +
          jacobian;
        
        
        if(isnan(logaccept)){
          Rcpp::Rcout << new_param.t() << endl;
          Rcpp::Rcout << param.t() << endl;
          Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
          throw 1;
        }
        
        accepted = do_I_accept(logaccept) & acceptable;
        
        if(accepted){
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          adaptivemc.count_accepted();
          //accept_count++;
          //accept_count_local++;
          
          current_loglik = new_loglik;
          mtree.accept_make_change();
          //mtree.tausq_inv = tausqi_proposal;
          
          param = new_param;
          //need_update = true;
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us.\n"; 
          } 
        } else {
          //need_update = false;
          //mtree.tausq_inv = tausqi_original;
          
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
      //Rcpp::Rcout << "after updating theta "<< mtree.param_data.loglik_w << endl;
      
      //need_update = true;
      need_update = arma::accu(abs(param - predict_param) > 1e-05);
      
      if((mtree.predicting == true) & sample_predicts){
        // tell predict() if theta has changed because if not, we can avoid recalculating
        //***mtree.predict(need_update);
        predict_param = param;
      }
      
      
      //Rcpp::Rcout << "before updating tausq/beta "<< mtree.param_data.loglik_w << endl;
      if(sample_tausq){
        start = std::chrono::steady_clock::now();
        mtree.gibbs_sample_tausq();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & sample_tausq & verbose){
          Rcpp::Rcout << "[tausq] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                      << endl; 
        }
      }
      
      if(sample_beta){
        start = std::chrono::steady_clock::now();
        mtree.deal_with_beta();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & sample_beta & verbose){
          Rcpp::Rcout << "[beta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
          if(verbose || debug){
            Rcpp::Rcout << endl;
          }
        }
      }
      
      if(sample_tausq | sample_beta){
        mtree.update_ll_after_beta_tausq();
      }
      //Rcpp::Rcout << "after updating tausq/beta "<< mtree.param_data.loglik_w << endl;
      //need_update = true;
      
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
        for(int tt=0; tt<q; tt++){
          printf("tausq%1d=%.4f ", tt, 1.0/mtree.tausq_inv(tt));
        }
        printf("\n");
        
        
      } 
    
      if((m>0) & (mcmc > 100)){
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
      } else {
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      
      //save
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          tausq_mcmc.col(msaved) = 1.0 / mtree.tausq_inv;
          //sigmasq_mcmc.col(msaved) = mtree.sigmasq;
          beta_mcmc.col(msaved) = mtree.Bcoeff;
          theta_mcmc.col(msaved) = mtree.param_data.theta;
          
          w_mcmc(msaved) = mtree.w;
          yhat_mcmc(msaved) = mtree.XB + mtree.w + pow(mtree.tausq_inv_long, -.5) % arma::randn(mtree.X.n_rows);
          //eta_rpx_mcmc(msaved) = mtree.eta_rpx;
          
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
      Rcpp::Named("beta_mcmc") = beta_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("block_ct_obs") = mtree.block_ct_obs,
      
      Rcpp::Named("indexing") = mtree.indexing,
      Rcpp::Named("parents_indexing") = mtree.parents_indexing,
      
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("Kxx_inv") = mtree.param_data.Kxx_inv,
      Rcpp::Named("w_cond_mean_K") = mtree.param_data.w_cond_mean_K,
      Rcpp::Named("Kcc") = mtree.param_data.Kcc
    );
    
  /*} catch (const std::exception &exc) {
    end_all = std::chrono::steady_clock::now();
    Rcpp::Rcout << exc.what() << endl;
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted." << endl;
    
    return Rcpp::List::create(
      Rcpp::Named("None") = arma::zeros(0)
    );
  }*/
  
}

