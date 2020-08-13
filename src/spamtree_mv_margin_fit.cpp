
#include "spamtree_mv_model.h"
#include "multires_utils.h"
#include "interrupt_handler.h"

// inactive but dont erase
// needs update with new mh_adapt.h

/*
arma::mat rebuildC(const arma::field<arma::mat>& Cf, const arma::vec& block_names, const arma::uvec& block_is_reference, std::string part="all"){
  
  arma::uvec cdim = arma::zeros<arma::uvec>(Cf.n_cols + 1);
  arma::uvec rdim = arma::zeros<arma::uvec>(Cf.n_rows + 1);
  //Rcpp::Rcout << arma::size(Cf) << endl;
  
  for(int i=0; i<Cf.n_rows; i++){
    Rcpp::Rcout << i << endl;
    rdim(i+1) = Cf(i,i).n_rows;
    if(Cf(i,i).n_cols == Cf(i,i).n_rows){
      cdim(i+1) = Cf(i,i).n_cols;
    } else {
      cdim(i+1) = Cf(i,i).n_rows;
    }
  }
  arma::mat result = arma::zeros( arma::accu(rdim), arma::accu(cdim) );
  
  rdim = arma::cumsum(rdim);
  cdim = rdim;
  
  for(int i=0; i<Cf.n_rows; i++){
    int ncols = part=="all" ? Cf.n_cols : i+1;
    for(int j=0; j<ncols; j++){
      //Rcpp::Rcout << i << " " << j << endl;
      int fr = rdim(i);
      int lr = rdim(i+1)-1;
      int fc = cdim(j);
      int lc = cdim(j+1)-1;
      if(Cf(i,j).n_rows > 0){
        if(i==j){
          if(Cf(i,i).n_cols == Cf(i,i).n_rows){
            //Rcpp::Rcout << "1 " << endl;
            result.submat(fr, fc, lr, lc) = Cf(i, i);  
          } else {
            //Rcpp::Rcout << "2 " << endl;
            result.submat(fr, fc, lr, lc) = arma::diagmat(Cf(i,i).col(0));
          }
        } else {
          //Rcpp::Rcout << "3 " << endl;
          result.submat(fr, fc, lr, lc) = Cf(i, j);
        } 
      }
    }
  }
  return result;
}

arma::mat rebuildH(const arma::field<arma::mat>& Hf, const arma::vec& block_names, const arma::uvec& block_is_reference, std::string part="all"){
  
  arma::uvec cdim = arma::zeros<arma::uvec>(Hf.n_cols + 1);
  arma::uvec rdim = arma::zeros<arma::uvec>(Hf.n_rows + 1);
  //Rcpp::Rcout << arma::size(Hf) << endl;
  
  for(int i=0; i<Hf.n_rows; i++){
    for(int j=0; j<Hf.n_cols; j++){
      if(Hf(i, j).n_rows > 0){
        rdim(i+1) = Hf(i,j).n_rows;
        cdim(j+1) = Hf(i,j).n_cols;
      }
    }
  }
  
  //Rcpp::Rcout << arma::join_horiz(rdim, cdim) << endl;
  
  arma::mat result = arma::zeros( arma::accu(rdim), arma::accu(cdim) );
  
  rdim = arma::cumsum(rdim);
  cdim = arma::cumsum(cdim);
  
  for(int i=0; i<Hf.n_rows; i++){
    int ncols = part=="all" ? Hf.n_cols : i+1;
    for(int j=0; j<ncols; j++){
      Rcpp::Rcout << i << " " << j << endl;
      int fr = rdim(i);
      int lr = rdim(i+1)-1;
      int fc = cdim(j);
      int lc = cdim(j+1)-1;
      if(Hf(i,j).n_rows > 0){
        if(i==j){
          if(Hf(i,i).n_cols == Hf(i,i).n_rows){
            //Rcpp::Rcout << "1 " << endl;
            result.submat(fr, fc, lr, lc) = Hf(i, i);  
          } else {
            //Rcpp::Rcout << "2 " << endl;
            result.submat(fr, fc, lr, lc) = arma::diagmat(Hf(i,i).col(0));
          }
        } else {
          //Rcpp::Rcout << "3 " << fr << " " << fc << " " << lr << " " << lc << endl;
          result.submat(fr, fc, lr, lc) = Hf(i, j);
        } 
      }
    }
  }
  return result;
}

arma::mat rebuildR(const arma::field<arma::mat>& R){
  
  arma::uvec dims = arma::zeros<arma::uvec>(R.n_elem + 1);
  for(int i=0; i<R.n_elem; i++){
    dims(i+1) = R(i).n_rows;
  }
  
  arma::mat result = arma::zeros(arma::accu(dims), arma::accu(dims));
  dims = arma::cumsum(dims);
  
  for(int i=0; i<R.n_elem; i++){
    if(R(i).n_rows == R(i).n_cols){
      result.submat(dims(i), dims(i), dims(i+1)-1, dims(i+1)-1) = R(i);
    } else {
      result.submat(dims(i), dims(i), dims(i+1)-1, dims(i+1)-1) = arma::diagmat(R(i));  
    }
    
  }
  return result;
}


//[[Rcpp::export]]
arma::field<arma::mat> Hinverse(const arma::field<arma::mat>& ImH,
                   const arma::field<arma::uvec>& parents,
                   const arma::field<arma::uvec>& children,
                   int n_actual_groups,
                   const arma::field<arma::vec>& u_by_block_groups,
                   const arma::field<arma::uvec>& indexing){
  
  arma::field<arma::mat> ImHinv = ImH;
  
  //start from res 1 as res 0 has identity
  for(int res=1; res<n_actual_groups; res++){
    for(int g=0; g<u_by_block_groups(res).n_elem; g++){
      int u = u_by_block_groups(res)(g);
      Rcpp::Rcout << "u: " << u << endl;
      
      for(int p=0; p<parents(u).n_elem; p++){
        int pj = parents(u)(p);
        Rcpp::Rcout << "parent: " << pj << endl;
        arma::mat result = arma::zeros(indexing(u).n_elem, indexing(pj).n_elem);
        
        arma::uvec common_gp = arma::intersect(parents(u), arma::join_vert(arma::ones<arma::uvec>(1)*pj, children(pj)));
        Rcpp::Rcout << "common: " << endl << common_gp << endl;
        for(int gp=0; gp<common_gp.n_elem; gp++){
          int gpar = common_gp(gp);
          result -= ImH(u, gpar) * ImHinv(gpar, pj); 
        }
        //Rcpp::Rcout << "operating on " << endl << indexing(u) << endl << indexing(pj) << endl;
        ImHinv(u, pj) = result;
      }
    }
  }
  
  return ImHinv;
}

//[[Rcpp::export]]
Rcpp::List spamtree_mv_margin_mcmc(
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
  
  SpamTreeMV mtree = SpamTreeMV();
  
  arma::vec start_w_vec = arma::randn(y.n_elem);
  
  bool limited_tree = false;
  mtree = SpamTreeMV("gaussian", y, X, Z, coords, mv_id, 
                     blocking, gix_block, res_is_ref,
                     
                     parents, children, limited_tree, layer_names, layer_gibbs_group,
                     indexing,
                     
                     start_w_vec, beta, theta, 1.0/tausq, 
                     
                     use_alg, num_threads,
                     verbose, debug);
  
  
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
  
  mtree.get_loglik_comps_w( mtree.param_data );
  mtree.get_loglik_comps_w( mtree.alter_data );
  mtree.deal_with_w(true);
  mtree.get_loglik_w(mtree.param_data);
   
  arma::vec param = mtree.param_data.theta;//arma::join_vert( , mtree.tausq_inv );
  arma::vec predict_param = param;
  double current_loglik = tempr*mtree.param_data.loglik_w;
  //if(verbose & debug){
  //  Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  //}
  
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
  
  arma::vec sumparam = arma::zeros(param.n_elem);
  arma::mat prodparam = arma::zeros(param.n_elem, param.n_elem);
  arma::mat paramsd = metropolis_sd; // proposal sd
  arma::vec sd_param = arma::zeros(mcmc +1); // mcmc sd
  
  double ll_upd_msg;
  
  bool need_update = true;//arma::accu(abs(param - predict_param) > 1e-05);
  
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
        mtree.deal_with_w(need_update);
        mtree.get_loglik_w(mtree.param_data);
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
        propos_count++;
        propos_count_local++;
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          paramsd * arma::randn(param.n_elem), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        //Rcpp::Rcout << "new phi: " << new_param << endl;
        
        arma::vec theta_proposal = new_param.subvec(0, npars-1);
        //arma::vec tausqi_proposal = new_param.subvec(npars, param.n_elem-1);
        //arma::vec tausqi_original = param.subvec(npars, param.n_elem-1);
        
        mtree.theta_update(mtree.alter_data, theta_proposal);
        mtree.get_loglik_comps_w( mtree.alter_data );
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
        
        //prior_logratio = calc_prior_logratio(k, new_param, param, npars, dlim);
        jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
        logaccept = new_loglik - current_loglik + //prior_logratio + 
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
        
        accepted = do_I_accept(logaccept);
        
        if(accepted){
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          
          if(sample_tausq & false){
            start = std::chrono::steady_clock::now();
            mtree.gibbs_sample_tausq();
            end = std::chrono::steady_clock::now();
            if(verbose_mcmc & sample_tausq & verbose){
              Rcpp::Rcout << "[tausq] " 
                          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                          << endl; 
            }
          }
          
          accept_count++;
          accept_count_local++;
          current_loglik = new_loglik;
          mtree.accept_make_change();
          //mtree.tausq_inv = tausqi_proposal;
          
          param = new_param;
          need_update = true;
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us.\n"; 
          } 
        } else {
          need_update = false;
          //mtree.tausq_inv = tausqi_original;
          
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
      
      if((mtree.predicting == true) & sample_predicts){
        // tell predict() if theta has changed because if not, we can avoid recalculating
        mtree.predict(need_update);
        predict_param = param;
      }
      
      if(sample_tausq & true){
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
      
      need_update = true;
      
      if(printall){
        //Rcpp::checkUserInterrupt();
        interrupted = checkInterrupt();
        if(interrupted){
          throw 1;
        }
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick_mcmc ).count();
        
        printf("%5d-th iteration [ %dms ] MCMC acceptance %.2f%% (total: %.2f%%)\n", 
               m+1, itertime, accept_ratio_local*100, accept_ratio*100);
        for(int pp=0; pp<npars; pp++){
          printf("theta%1d=%.4f ", pp, mtree.param_data.theta(pp));
        }
        for(int tt=0; tt<q; tt++){
          printf("tausq%1d=%.4f ", tt, 1.0/mtree.tausq_inv(tt));
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
    
    mtree.fill_precision_blocks(mtree.param_data);
    Rcpp::Rcout << arma::size(mtree.param_data.Ciblocks) << endl;
    arma::mat Cisaved = rebuildC(mtree.param_data.Ciblocks, mtree.block_names, mtree.block_is_reference);
    
    mtree.decompose_margin_precision(mtree.param_data);
    
    arma::mat Hsaved = rebuildH(mtree.param_data.Hblocks, mtree.block_names, mtree.block_is_reference);
    
    
    arma::field<arma::mat> ImH(mtree.param_data.Hblocks.n_rows, mtree.param_data.Hblocks.n_cols);
    for(int i=0; i<mtree.param_data.Hblocks.n_rows; i++){
      for(int j=0; j<mtree.param_data.Hblocks.n_cols; j++){
        if(mtree.param_data.Hblocks.n_rows > 0){
          if(i==j){
            ImH(i,i) = arma::eye(indexing(i).n_rows, indexing(i).n_rows);
          } else {
            ImH(i,j) = -mtree.param_data.Hblocks(i,j);
          }
        }
      }
    }
    
    arma::field<arma::mat> ImHinv = Hinverse(ImH,
                                           mtree.parents, mtree.children, mtree.n_actual_groups,
                                           mtree.u_by_block_groups, mtree.indexing);
      
    arma::mat ImHinvsaved = rebuildH(ImHinv, mtree.block_names, mtree.block_is_reference);
    
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
      
      Rcpp::Named("paramsd") = paramsd,
      Rcpp::Named("block_ct_obs") = mtree.block_ct_obs,
      
      Rcpp::Named("indexing") = mtree.indexing,
      Rcpp::Named("parents_indexing") = mtree.parents_indexing,
      
      Rcpp::Named("block_names") = mtree.block_names,
      Rcpp::Named("block_is_reference") = mtree.block_is_reference,
      Rcpp::Named("u_by_block_groups") = mtree.u_by_block_groups,
      Rcpp::Named("n_actual_groups") = mtree.n_actual_groups,
      Rcpp::Named("Ciblocks") = Cisaved,
      Rcpp::Named("Hblocks") = Hsaved,
      Rcpp::Named("ImHinv") = ImHinvsaved,
      Rcpp::Named("Riblocks") = rebuildR(mtree.param_data.Riblocks), //rebuilder(mtree.param_data.Riblocks, mtree.block_names, mtree.block_is_reference),
      
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

*/