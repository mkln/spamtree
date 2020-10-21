
Eigen::VectorXd armavec_to_vectorxd(arma::vec arma_A) {
  
  Eigen::VectorXd eigen_B = Eigen::Map<Eigen::VectorXd>(arma_A.memptr(),
                                                        arma_A.n_elem);
  return eigen_B;
}

void expand_grid_with_values_(arma::umat& locs,
                              arma::vec& vals,
                              
                              int rowstart, int rowend,
                              const arma::uvec& x1,
                              const arma::uvec& x2,
                              const arma::mat& values){
  
  for(int i=rowstart; i<rowend; i++){
    arma::uvec ix;
    try {
      ix = arma::ind2sub(arma::size(values), i-rowstart);
    } catch (...) {
      Rcpp::Rcout << arma::size(values) << " " << i-rowstart << " " << i <<" " << rowstart << " " << rowend << endl;
      throw 1;
    }
    locs(0, i) = x1(ix(0));
    locs(1, i) = x2(ix(1));
    vals(i) = values(ix(0), ix(1));
  }
}

Rcpp::List SpamTreeMV::Cinv(){
  
  arma::uvec oneton = arma::regspace<arma::uvec>(0, 1, n-1);
  
  //omp_set_num_threads(num_threads);
  
  //int n_blocks = block_names.n_elem;
  //arma::uvec qvblock_c = mv_id-1;
  //arma::field<arma::uvec> parents_indexing(n_blocks);
  
  arma::uvec Adims = arma::zeros<arma::uvec>(n_blocks+1);
  arma::uvec Ddims = arma::zeros<arma::uvec>(n_blocks+1);
  
  arma::uvec sort_index = field_v_concat_uv(indexing);
  arma::uvec reverse_ix = arma::reverse(sort_index);
  arma::field<arma::uvec> cindexing(n_blocks);
  arma::field<arma::uvec> pindexing(n_blocks);
  
  int istart=0;
  for(int i=0; i<n_blocks; i++){
    cindexing(i) = indexing(i); //arma::regspace<arma::uvec>(istart, 1, istart+indexing(i).n_elem-1);
    istart += indexing(i).n_elem;
  }
  
  //***#pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = i;//block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = cindexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      pindexing(u) = field_v_concat_uv(pixs);
      Adims(i+1) = indexing(u).n_elem * parents_indexing(u).n_elem;
    }
    Ddims(i+1) = indexing(u).n_elem * indexing(u).n_elem;
  }
  
  int Asize = arma::accu(Adims);
  Adims = arma::cumsum(Adims);
  
  arma::umat Hlocs = arma::zeros<arma::umat>(2, Asize);
  arma::vec Hvals = arma::zeros(Asize);
  
  int Dsize = arma::accu(Ddims);
  Ddims = arma::cumsum(Ddims);
  
  arma::umat Dlocs2 = arma::zeros<arma::umat>(2, Dsize);
  arma::vec Dvals2 = arma::zeros(Dsize);
  
  int errtype=-1;
  
  for(int g=0; g<n_actual_groups; g++){
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      if(parents(u).n_elem == 0){
        arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
        
        try{
          param_data.Kxx_invchol(u) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
          param_data.Kxx_inv(u) = param_data.Kxx_invchol(u).t() * param_data.Kxx_invchol(u);
          param_data.Rcc_invchol(u) = param_data.Kxx_invchol(u); 
          param_data.w_cond_prec(u) = param_data.Kxx_inv(u);
          
          param_data.ccholprecdiag(u) = param_data.Rcc_invchol(u).diag();
        } catch(...){
          errtype = 1;
        }
        
        expand_grid_with_values_(Dlocs2, Dvals2, Ddims(u), Ddims(u+1),
                                 cindexing(u), cindexing(u), param_data.w_cond_prec(u));
        
      } else {
        int last_par = parents(u)(parents(u).n_elem - 1);
        Covariancef_inplace(param_data.Kxc(u), coords, qvblock_c, parents_indexing(u), indexing(u), covpars, false);
        param_data.w_cond_mean_K(u) = param_data.Kxc(u).t() * param_data.Kxx_inv(last_par);
        
        if(res_is_ref(g) == 1){
          //start = std::chrono::steady_clock::now();
          arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
          
          try {
            param_data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(
              Kcc - param_data.w_cond_mean_K(u) * param_data.Kxc(u)), "lower")));
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            if(children(u).n_elem > 0){
              if(limited_tree){
                param_data.Kxx_inv(u) = arma::inv_sympd(Kcc);
              } else {
                invchol_block_inplace_direct(param_data.Kxx_invchol(u), param_data.Kxx_invchol(last_par), 
                                             param_data.w_cond_mean_K(u), param_data.Rcc_invchol(u));
                param_data.Kxx_inv(u) = param_data.Kxx_invchol(u).t() * param_data.Kxx_invchol(u);
              }
              
              param_data.has_updated(u) = 1;
            }
            param_data.w_cond_prec(u) = param_data.Rcc_invchol(u).t() * param_data.Rcc_invchol(u);
          } catch(...){
            errtype = 2;
          }
          
          expand_grid_with_values_(Hlocs, Hvals, Adims(u), Adims(u+1),
                                   cindexing(u), pindexing(u), param_data.w_cond_mean_K(u));
          
          expand_grid_with_values_(Dlocs2, Dvals2, Ddims(u), Ddims(u+1),
                                   cindexing(u), cindexing(u), param_data.w_cond_prec(u));
          
        } else {
          // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
          arma::uvec ones = arma::ones<arma::uvec>(1);
          arma::vec diag_condprec = arma::zeros(indexing(u).n_elem);
          Rcpp::Rcout << "- "<< endl;
          for(int ix=0; ix<indexing(u).n_elem; ix++){
            arma::uvec uix = ones * indexing(u)(ix);
            //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
            int first_ix = ix;
            int last_ix = ix;
            //start = std::chrono::steady_clock::now();
            arma::mat Kcc = Covariancef(coords, qvblock_c, uix, uix, covpars, true);
            
            arma::mat Kcx_xxi_xc = param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * param_data.Kxc(u).cols(first_ix, last_ix);
            arma::mat Rinvchol;
            try {
              Rinvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc - Kcx_xxi_xc), "lower")));
              param_data.ccholprecdiag(u).subvec(first_ix, last_ix) = Rinvchol.diag();
              param_data.w_cond_prec_noref(u)(ix) = Rinvchol.t() * Rinvchol;
              diag_condprec(ix) = param_data.w_cond_prec_noref(u)(ix)(0,0);
            } catch(...){
              errtype = 3;
            }
          }
          
          expand_grid_with_values_(Hlocs, Hvals, Adims(u), Adims(u+1),
                                   cindexing(u), pindexing(u), param_data.w_cond_mean_K(u));
          
          expand_grid_with_values_(Dlocs2, Dvals2, Ddims(u), Ddims(u+1),
                                   cindexing(u), cindexing(u), arma::diagmat(diag_condprec));
          
        }
      }
    }
  }
  
  // EIGEN
  Rcpp::Rcout << "n: " << n << endl;
  Eigen::SparseMatrix<double> I_eig(n, n);
  I_eig.setIdentity();
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  std::vector<T> tripletList_Dic2;
  std::vector<T> tripletList_RR;
  
  tripletList_RR.reserve(I_eig.cols());
  for(int i=0; i<I_eig.cols(); i++){
    tripletList_RR.push_back(
      T(i, I_eig.cols()-i-1, 1)
    );
  }
  
  
  Eigen::SparseMatrix<double> RR(n,n);
  RR.setFromTriplets(tripletList_RR.begin(), tripletList_RR.end());
  
  
  tripletList_H.reserve(Hlocs.n_cols);
  for(int i=0; i<Hlocs.n_cols; i++){
    tripletList_H.push_back(T(Hlocs(0, i), Hlocs(1, i), Hvals(i)));
  }
  Eigen::SparseMatrix<double> He(n,n);
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  tripletList_Dic2.reserve(Dlocs2.n_cols);
  for(int i=0; i<Dlocs2.n_cols; i++){
    tripletList_Dic2.push_back(T(Dlocs2(0, i), Dlocs2(1, i), Dvals2(i)));
  }
  Eigen::SparseMatrix<double> Di(n,n);
  Di.setFromTriplets(tripletList_Dic2.begin(), tripletList_Dic2.end());
  
  Eigen::SparseMatrix<double> L = (I_eig-He).triangularView<Eigen::Lower>().transpose();
  
  Eigen::SparseMatrix<double> Ci = L * Di *  L.transpose();
  
  // perform LDLT
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>, Eigen::Lower > choler;
  
  start = std::chrono::steady_clock::now();
  choler.compute(Ci);
  Eigen::VectorXd ye = armavec_to_vectorxd(y);
  Eigen::VectorXd Ciy = choler.solve(ye);
  //Eigen::SparseMatrix<double> LL1 = choler.matrixL();
  end = std::chrono::steady_clock::now();
  Rcpp::Rcout << "Cholesky std: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << "us.\n";
  
  // Eigen::NaturalOrdering<int> 
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower> choler2;
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> Cir = (RR * Ci * RR).transpose();
  choler2.compute(Cir);
  Eigen::SparseMatrix<double> LL2 = choler2.matrixL().triangularView<Eigen::Lower>();//(RR * choler.matrixL() * RR).transpose();
  end = std::chrono::steady_clock::now();
  Rcpp::Rcout << "Cholesky rev: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << "us.\n";
  
  Rcpp::Rcout << "returning " << endl;
  
  return Rcpp::List::create(
    //Rcpp::Named("Ci") = Ci,
    //Rcpp::Named("Cir") = Cir,
    // Rcpp::Named("RR") = RR,
    //Rcpp::Named("H") = He,
    //Rcpp::Named("IminusH") = I_eig - He,
    //Rcpp::Named("Di") = Di,
    Rcpp::Named("sort_index") = sort_index,
    Rcpp::Named("reverse_ix") = reverse_ix,
    //Rcpp::Named("LL1") = LL1,
    //Rcpp::Named("LL2") = LL2,
    Rcpp::Named("cindexing") = cindexing
  );
}





//[[Rcpp::export]]
Rcpp::List Cinv_builtin(
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
  
  arma::vec start_w_vec = arma::zeros(coords.n_rows);
  
  SpamTreeMV mtree(y, X, Z, coords, mv_id, 
                   blocking, gix_block, res_is_ref,
                   
                   parents, children, limited_tree, 
                   layer_names, layer_gibbs_group,
                   indexing,
                   
                   start_w_vec, beta, theta, 1.0/tausq, 
                   
                   use_alg, num_threads,
                   verbose, debug);
  
  mtree.get_loglik_comps_w( mtree.param_data );
  return mtree.Cinv();
  
}

