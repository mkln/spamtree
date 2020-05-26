#include "mesh_multires.h"
#include "multires_utils.h"
#include "interrupt_handler.h"

//[[Rcpp::export]]
Rcpp::List mesh_multires_mcmc(
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
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
  
  if(d == 2){
    npars = 1 + 1; // sigmasq + phi
  } else {
    npars = 3 + 1; // sigmasq + alpha + beta + phi
  }
  
  if(q > 1){
    k = q * (q-1)/2;
    npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)
    Rcpp::Rcout << "Multivariate multiresolution not implemented yet.\n";
    throw 1;
  }

  Rcpp::Rcout << set_unif_bounds << endl;
  
  SpamTree mesh = SpamTree();
  
  mesh = SpamTree(y, X, Z, coords, blocking,
                  
                  parents, children, layer_names, layer_gibbs_group,
                  indexing,
                  
                  start_w, beta, theta, 1.0/tausq, sigmasq,
                  
                  use_alg,
                  verbose, debug);
  
  
  arma::mat b_mcmc = arma::zeros(X.n_cols, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(1, mcmc_keep);
  arma::mat sigmasq_mcmc = arma::zeros(1, mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(npars, mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_keep);
  
  arma::vec grps = arma::unique(layer_gibbs_group);
  int n_res = grps.n_elem;
  arma::cube eta_rpx_mcmc = arma::zeros(coords.n_rows, mcmc_keep, n_res);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
//*** omp parallel for
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, q);
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, 1);
  }
  
  mesh.get_loglik_comps_w( mesh.param_data );
  mesh.get_loglik_comps_w( mesh.alter_data );
  
  arma::vec param = mesh.param_data.theta;
  double current_loglik = tempr*mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  }
  
  double logaccept;
  
  double propos_count = 0;
  double accept_count = 0;
  double accept_ratio = 0;
  double propos_count_local = 0;
  double accept_count_local = 0;
  double accept_ratio_local = 0;
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  int msaved = 0;
  bool interrupted = false;
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  
  arma::vec sumparam = arma::zeros(npars);
  arma::mat prodparam = arma::zeros(npars, npars);
  arma::mat paramsd = mcmcsd; // proposal sd
  arma::vec sd_param = arma::zeros(mcmc +1); // mcmc sd
  
  double ll_upd_msg;
  
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  try { 
    for(m=0; m<mcmc; m++){
      
      mesh.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          mesh.predicting = true;
        } 
      }
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      ll_upd_msg = current_loglik;
      
      start = std::chrono::steady_clock::now();
      if(sample_w){
        mesh.gibbs_sample_w();
        mesh.get_loglik_w(mesh.param_data);
        current_loglik = tempr*mesh.param_data.loglik_w;
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
      if(sample_sigmasq & false){
        mesh.gibbs_sample_sigmasq();
        current_loglik = tempr*mesh.param_data.loglik_w;
        //Rcpp::Rcout << "loglik: " << current_loglik << "\n";
        //mesh.get_loglik_comps_w( mesh.param_data );
        //current_loglik = mesh.param_data.loglik_w;
        //Rcpp::Rcout << "recalc: " << mesh.param_data.loglik_w << "\n";
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_sigmasq & verbose){
        Rcpp::Rcout << "[sigmasq] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. \n";
        //Rcpp::Rcout << " >>>> CHECK from: " << mesh.param_data.loglik_w << endl;
        //mesh.get_loglik_comps_w( mesh.param_data );
        //Rcpp::Rcout << " >>>> CHECK with: " << mesh.param_data.loglik_w << endl;
        //current_loglik = mesh.param_data.loglik_w;
      }
      
      ll_upd_msg = current_loglik;
      start = std::chrono::steady_clock::now();
      if(sample_theta){
        propos_count++;
        propos_count_local++;
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          paramsd * arma::randn(npars), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        //Rcpp::Rcout << "new phi: " << new_param << endl;
        
        mesh.theta_update(mesh.alter_data, new_param); 
        
        mesh.get_loglik_comps_w( mesh.alter_data );
        
        bool accepted = !out_unif_bounds;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        if(!mesh.alter_data.cholfail){
          new_loglik = tempr*mesh.alter_data.loglik_w;
          current_loglik = tempr*mesh.param_data.loglik_w;
          
          if(isnan(current_loglik)){
            Rcpp::Rcout << "At nan loglik: error. \n";
            throw 1;
          }
          
          //prior_logratio = calc_prior_logratio(k, new_param, param, npars, dlim);
          jacobian       = calc_jacobian(new_param, param, set_unif_bounds);
          logaccept = new_loglik - current_loglik + //prior_logratio + 
            invgamma_logdens(new_param(0), 2, 2) -
            invgamma_logdens(mesh.param_data.theta(0), 2, 2) +
            jacobian;
          
          if(isnan(logaccept)){
            Rcpp::Rcout << new_param.t() << endl;
            Rcpp::Rcout << param.t() << endl;
            Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
            throw 1;
          }
          
          accepted = do_I_accept(logaccept);
        } else {
          accepted = false;
          num_chol_fails ++;
          printf("[warning] chol failure #%d at mh proposal -- auto rejected\n", num_chol_fails);
          Rcpp::Rcout << new_param.t() << "\n";
        }
      
        if(accepted){
          
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          accept_count++;
          accept_count_local++;
          current_loglik = new_loglik;
          mesh.accept_make_change();
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
        
        accept_ratio = accept_count/propos_count;
        accept_ratio_local = accept_count_local/propos_count_local;
        
        if(adapting){
          adapt(par_huvtransf_fwd(param, set_unif_bounds), sumparam, prodparam, paramsd, sd_param, m, accept_ratio); // **
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
      if((mesh.predicting == true) & sample_predicts){
        //Rcpp::Rcout << "[predict] with current theta: " << mesh.param_data.theta.t() << " vs old " << mesh.alter_data.theta.t() << endl;
        mesh.predict();
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_beta){
        mesh.gibbs_sample_beta();
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
      if(sample_tausq){
        mesh.gibbs_sample_tausq();
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
        
        printf("%5d-th iteration [ %dms ] ~ tsq=%.4f | MCMC acceptance %.2f%% (total: %.2f%%)\n", 
               m+1, itertime, 1.0/mesh.tausq_inv, accept_ratio_local*100, accept_ratio*100);
        for(int pp=0; pp<npars; pp++){
          printf("theta%1d=%.4f ", pp, mesh.param_data.theta(pp));
        }
        printf("\n");
        
        tick_mcmc = std::chrono::steady_clock::now();
      } else {
        if((m>0) & (mcmc > 200)){
          if(!(m % (mcmc / 20))){
            //Rcpp::Rcout << paramsd << endl;
            accept_count_local = 0;
            propos_count_local = 0;
            
            interrupted = checkInterrupt();
            if(interrupted){
              throw 1;
            }
            end_mcmc = std::chrono::steady_clock::now();
            if(true){
              int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
              int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
              printf("%.1f%% %dms (total: %dms) ~ MCMC acceptance %.2f%% (total: %.2f%%) \n",
                     floor(100.0*(m+0.0)/mcmc),
                     time_tick,
                     time_mcmc,
                     accept_ratio_local*100, accept_ratio*100);
              
              tick_mcmc = std::chrono::steady_clock::now();
            }
          } 
        }
      }
      
      
      //save
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          tausq_mcmc.col(msaved) = 1.0 / mesh.tausq_inv;
          //sigmasq_mcmc.col(msaved) = mesh.sigmasq;
          b_mcmc.col(msaved) = mesh.Bcoeff;
          theta_mcmc.col(msaved) = mesh.param_data.theta;
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mesh.w;
          yhat_mcmc(msaved) = mesh.X * mesh.Bcoeff + mesh.Zw;
          
          eta_rpx_mcmc.col(msaved) = mesh.eta_rpx;
          
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
      Rcpp::Named("paramsd") = paramsd,
      
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("parents_indexing") = mesh.parents_indexing,
      Rcpp::Named("children_indexing") = mesh.children_indexing
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

