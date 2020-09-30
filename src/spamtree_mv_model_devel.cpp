#define ARMA_DONT_PRINT_ERRORS
#include "spamtree_mv_model_devel.h"



void SpamTreeMVdevel::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

SpamTreeMVdevel::SpamTreeMVdevel(){
  
}

SpamTreeMVdevel::SpamTreeMVdevel(
  const std::string& family_in,
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  const arma::uvec& mv_id_in,
  
  const arma::uvec& blocking_in,
  const arma::uvec& gix_block_in,
  const arma::uvec& res_is_ref_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  bool limited_tree_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  
  const arma::mat& w_in,
  const arma::vec& beta_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  
  char use_alg_in='S',
  int max_num_threads_in=4,
  bool v=false,
  bool debugging=false){
  
  
  
  use_alg = use_alg_in; //S:standard, P:residual process, R: rp recursive
  max_num_threads = max_num_threads_in;
  
  message("~ SpamTreeMVdevel initialization.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  
  verbose = v;
  debug = debugging;
  
  message("SpamTreeMVdevel::SpamTreeMVdevel assign values.");
  
  family = family_in;
  
  y                   = y_in;
  X                   = X_in;
  Z                   = Z_in;
  
  coords              = coords_in;
  mv_id               = mv_id_in;
  qvblock_c           = mv_id-1;
  blocking            = blocking_in;
  
  // for every row, this indicates which mini-block INSIDE the block it corresponds to
  // basically groups tied coordinates together
  gix_block           = gix_block_in;
  
  res_is_ref          = res_is_ref_in;
  parents             = parents_in;
  children            = children_in;
  limited_tree        = limited_tree_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_actual_groups     = n_gibbs_groups;
  n_blocks            = block_names.n_elem;
  
  Rcpp::Rcout << n_gibbs_groups << " tree levels. " << endl;
  //Rcpp::Rcout << block_groups_labels << endl;
  
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  arma::uvec mv_id_uniques = arma::unique(mv_id);
  q  = mv_id_uniques.n_elem;//Z.n_cols;
  dd = coords.n_cols;
  
  ix_by_q = arma::field<arma::uvec>(q);
  ix_by_q_a = arma::field<arma::uvec>(q);
  arma::uvec qvblock_c_available = qvblock_c.rows(na_ix_all);
  for(int j=0; j<q; j++){
    ix_by_q(j) = arma::find(qvblock_c == j);
    ix_by_q_a(j) = arma::find(qvblock_c_available == j);
  }
  
  //Zw      = arma::zeros(coords.n_rows);
  //eta_rpx = arma::zeros(coords.n_rows, q, n_gibbs_groups);
  
  indexing    = indexing_in;
  
  /*
   if(dd == 2){
   if(q > 2){
   npars = 1+3;
   } else {
   npars = 1+1;
   }
   } else {
   if(q > 2){
   npars = 1+5;
   } else {
   npars = 1+3; // sigmasq + alpha + beta + phi
   }
   }*/
  
  if(dd == 2){
    int n_cbase = q > 2? 3: 1;
    npars = 3*q + n_cbase;
  } else {
    Rcpp::Rcout << "d>2 not implemented yet " << endl;
    throw 1;
  }
  
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  // init
  
  //Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  this_is_jth_child = arma::field<arma::uvec> (n_blocks);
  
  dim_by_parent = arma::field<arma::vec> (n_blocks);
  //dim_by_child = arma::field<arma::vec> (n_blocks);
  
  tausq_inv        = arma::ones(q) * tausq_inv_in;
  tausq_inv_long   = arma::ones(y.n_elem) * tausq_inv_in;
  
  XB = arma::zeros(coords.n_rows);
  Bcoeff           = arma::zeros(p, q);
  
  for(int j=0; j<q; j++){
    //Rcpp::Rcout << " -> " << ix_by_q(j).n_elem << endl;
    //Rcpp::Rcout << arma::size(X) << " " << arma::size(XB) << " " << endl;
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * beta_in;
    Bcoeff.col(j) = beta_in;
  }
  w                = w_in;
  
  predicting = true;
  
  // now elaborate
  message("SpamTreeMVdevel::SpamTreeMVdevel : init_indexing()");
  init_indexing();
  
  message("SpamTreeMVdevel::SpamTreeMVdevel : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  
  
  // prior for beta
  XtX = arma::field<arma::mat>(q);
  for(int j=0; j<q; j++){
    XtX(j)   = X_available.rows(ix_by_q_a(j)).t() * 
      X_available.rows(ix_by_q_a(j));
  }
  
  Vi    = .01 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  message("SpamTreeMVdevel::SpamTreeMVdevel : make_gibbs_groups()");
  make_gibbs_groups();
  message("SpamTreeMVdevel::SpamTreeMVdevel : init_finalize()");
  init_finalize();
  
  init_model_data(theta_in);
  
  //find_common_descendants();
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "SpamTreeMVdevel::SpamTreeMVdevel initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}

void SpamTreeMVdevel::make_gibbs_groups(){
  message("[make_gibbs_groups] check that graph makes sense\n");
  // checks -- errors not allowed. use check_groups.cpp to fix errors.
  for(int g=0; g<n_gibbs_groups; g++){
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(indexing(u).n_elem > 0){ //**
          
          for(int pp=0; pp<parents(u).n_elem; pp++){
            if(block_groups(parents(u)(pp)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                          << ": same group (" << block_groups(u) 
                          << ")." << endl;
              throw 1;
            }
          }
          for(int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << endl;
              throw 1;
            }
          }
        }
      }
    }
  }
  
  message("[make_gibbs_groups] manage groups for fitting algs\n");
  arma::field<arma::vec> u_by_block_groups_temp (n_gibbs_groups);
  arma::uvec there_are_blocks = arma::ones<arma::uvec>(n_gibbs_groups);
  
  /// create list of groups for gibbs
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups_temp(g) = arma::zeros(0);
    
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        //if(block_ct_obs(u) > 0){ //**
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups_temp(g) = arma::join_vert(u_by_block_groups_temp(g), uhere);
        //} 
      }
    }
  }
  
  u_by_block_groups = u_by_block_groups_temp;
  
  message("[make_gibbs_groups] list nonempty, predicting, and reference blocks\n");  
  //blocks_not_empty = arma::find(block_ct_obs > 0);
  
  block_is_reference = arma::ones<arma::uvec>(n_blocks);//(blocks_not_empty.n_elem);
  
  // resolutions that are not reference
  arma::uvec which_not_reference = arma::find(res_is_ref == 0);
  for(int g=0; g<n_actual_groups; g++){
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      if(g < n_actual_groups-1){
        block_is_reference(u) = 0;
      } 
    }
  }
  
  blocks_predicting = arma::zeros<arma::uvec>(u_by_block_groups(n_actual_groups-1).n_elem);//arma::find(block_ct_obs == 0);
  for(int i=0; i<blocks_predicting.n_elem; i++){
    blocks_predicting(i) = block_ct_obs(i) == 0? 1 : 0;
  }
  
  message("[make_gibbs_groups] done.\n");
}

void SpamTreeMVdevel::na_study(){
  // prepare stuff for NA management
  message("[na_study] NA management for predictions\n");
  
  block_ct_obs = arma::zeros<arma::uvec>(n_blocks);//(y_blocks.n_elem);
  na_1_blocks = arma::field<arma::uvec>(n_blocks);
  Ib = arma::field<arma::sp_mat>(n_blocks);
  for(int i=0; i<n_blocks; i++){
    arma::vec yvec = y.rows(indexing(i));
    arma::uvec y_not_na = arma::find_finite(yvec);
    //Rcpp::Rcout << "2"<< endl;
    block_ct_obs(i) = y_not_na.n_elem;
    na_1_blocks(i) = arma::zeros<arma::uvec>(yvec.n_elem);
    //Rcpp::Rcout << "3"<< endl;
    if(y_not_na.n_elem > 0){
      na_1_blocks(i).elem(y_not_na).fill(1);
    }
    Ib(i) = arma::eye<arma::sp_mat>(indexing(i).n_elem, indexing(i).n_elem);
    for(int j=0; j<Ib(i).n_cols; j++){
      if(na_1_blocks(i)(j) == 0){
        Ib(i)(j,j) = 0;//1e-6;
      }
    }
  }
  
  message("[na_study] done.\n");
}

void SpamTreeMVdevel::init_indexing(){
  //arma::field<arma::uvec> qvblock(n_blocks);
  Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  message("[init_indexing] indexing, parent_indexing, children_indexing");
  //pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      //Rcpp::Rcout << "n. parents " << parents(u).n_elem << endl;
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
    
    if(children(u).n_elem > 0){
      arma::field<arma::uvec> cixs(children(u).n_elem);
      for(int ci=0; ci<children(u).n_elem; ci++){
        //Rcpp::Rcout << "n. children " << children(u).n_elem << endl;
        //Rcpp::Rcout << "--> " << children(u)(ci) << endl;
        cixs(ci) = indexing(children(u)(ci));//arma::find( blocking == children(u)(ci)+1 ); // children are 0-indexed 
      }
      children_indexing(u) = field_v_concat_uv(cixs);
    }
    
  }
  
}

void SpamTreeMVdevel::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  //pragma omp parallel for //**
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
      dim_by_parent(u) = arma::zeros(parents(u).n_elem + 1);
      for(int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;//coords_blocks(parents(u)(j)).n_rows;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  
  message("[init_finalize] u_is_which_col_f");
  
  //pragma omp parallel for // **
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    
    //if(indexing(u).n_elem > 0){//***
    // children-parent relationship variables
    u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (children(u).n_elem);
    this_is_jth_child(u) = arma::zeros<arma::uvec> (parents(u).n_elem);
    
    for(int c=0; c<children(u).n_elem; c++){
      int child = children(u)(c);
      //Rcpp::Rcout << "child: " << child << "\n";
      // which parent of child is u which we are sampling
      arma::uvec u_is_which = arma::find(parents(child) == u, 1, "first"); 
      
      // which columns correspond to it
      int firstcol = dim_by_parent(child)(u_is_which(0));
      int lastcol = dim_by_parent(child)(u_is_which(0)+1);
      
      int dimen = parents_indexing(child).n_elem;
      
      // / / /
      arma::uvec result = arma::regspace<arma::uvec>(0, dimen-1);
      arma::uvec rowsel = arma::zeros<arma::uvec>(result.n_rows);
      rowsel.subvec(firstcol, lastcol-1).fill(1);
      arma::uvec result_local = result.rows(arma::find(rowsel==1));
      arma::uvec result_other = result.rows(arma::find(rowsel==0));
      u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
      u_is_which_col_f(u)(c)(0) = result_local; // u parent of c is in these columns for c
      u_is_which_col_f(u)(c)(1) = result_other; // u parent of c is NOT in these columns for c
    }
    
      for(int p=0; p<parents(u).n_elem; p++){
        int up = parents(u)(p);
        arma::uvec cuv = arma::find(children(up) == u, 1, "first");
        this_is_jth_child(u)(p) = cuv(0); 
      }
  }
}

void SpamTreeMVdevel::init_model_data(const arma::vec& theta_in){
  
  message("[init_model_data]");
  // data for metropolis steps and predictions
  // block params
  
  param_data.has_updated   = arma::zeros<arma::uvec>(n_blocks);
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.Kxc           = arma::field<arma::mat> (n_blocks);
  param_data.Kxx_inv       = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.Kcc = arma::field<arma::mat>(n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.Kxx_invchol = arma::field<arma::mat> (n_blocks); // storing the inv choleskys of {parents(w), w} (which is parent for children(w))
  param_data.Rcc_invchol = arma::field<arma::mat> (n_blocks); 
  param_data.ccholprecdiag = arma::field<arma::vec> (n_blocks);
  
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.AK_uP_all = arma::field<arma::mat> (n_blocks);
  param_data.AK_uP_u_all = arma::field<arma::mat> (n_blocks);
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  param_data.ll_y_all       = 0;
  param_data.Ddiag = arma::field<arma::vec> (n_blocks);//***
  param_data.theta          = theta_in;
  
  param_data.Sigi_children = arma::field<arma::cube> (n_blocks);
  param_data.Smu_children = arma::field<arma::mat> (n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    if(children(i).n_elem > 0){
      param_data.Sigi_children(i) = arma::zeros(indexing(i).n_elem,
                               indexing(i).n_elem, children(i).n_elem);
      param_data.Smu_children(i) = arma::zeros(indexing(i).n_elem,
                              children(i).n_elem);
    }
    
    int u = block_names(i)-1;
    param_data.Kxx_invchol(u) = arma::zeros(parents_indexing(u).n_elem + indexing(u).n_elem, //Kxx_invchol(last_par).n_rows + Kcc.n_rows, 
                           parents_indexing(u).n_elem + indexing(u).n_elem);//Kxx_invchol(last_par).n_cols + Kcc.n_cols);
  
    param_data.w_cond_mean_K(u) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
    param_data.Kxc(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    param_data.ccholprecdiag(u) = arma::zeros(indexing(u).n_elem);
    
  
    param_data.w_cond_prec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    param_data.Rcc_invchol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    param_data.Sigi_chol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);

    //if(block_ct_obs(u) > 0){

    param_data.Ddiag(u) = arma::zeros(indexing(u).n_elem);
  //}

    param_data.AK_uP_all(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    param_data.AK_uP_u_all(u) = param_data.AK_uP_all(u) * param_data.w_cond_mean_K(u);
  }
  
  
  alter_data                = param_data; 
  
  Rcpp::Rcout << "init_model_data: indexing elements: " << indexing.n_elem << endl;
  
}


void SpamTreeMVdevel::get_loglik_w(SpamTreeMVDataDevel& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  get_loglik_w_std(data);
}

void SpamTreeMVdevel::get_loglik_w_std(SpamTreeMVDataDevel& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
  
  //arma::uvec blocks_not_empty = arma::find(block_ct_obs > 0);
  //***#pragma omp parallel for //**
  for(int g=0; g<n_actual_groups-1; g++){
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      //arma::mat w_x = arma::vectorise( arma::trans( w.rows(indexing(u)) ) );
      arma::vec w_x = w.rows(indexing(u));
      
      if(parents(u).n_elem > 0){
        w_x -= data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
      }
      
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      data.loglik_w_comps(u) = (indexing(u).n_elem + .0) * hl2pi -.5 * data.wcore(u);
    }
  }
  
  data.logdetCi = arma::accu(data.logdetCi_comps);
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps) + data.ll_y_all;
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  //print_data(data);
}

void SpamTreeMVdevel::theta_transform(const SpamTreeMVDataDevel& data){
  // from vector to all covariance components
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  arma::vec cparams = data.theta.subvec(0, npars - 1);
  n_cbase = q > 2? 3: 1;
  ai1 = cparams.subvec(0, q-1);
  ai2 = cparams.subvec(q, 2*q-1);
  phi_i = cparams.subvec(2*q, 3*q-1);
  thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
  
  if(k>0){
    Dmat = vec_to_symmat(data.theta.subvec(npars, npars + k - 1));
  } else {
    Dmat = arma::zeros(1,1);
  }
}

bool SpamTreeMVdevel::get_loglik_comps_w(SpamTreeMVDataDevel& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  return get_loglik_comps_w_std(data);
}

bool SpamTreeMVdevel::get_loglik_comps_w_std(SpamTreeMVDataDevel& data){
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_w_std] start. ");
  
  // arma::vec timings = arma::zeros(7);
  theta_transform(data);
  // cycle through the resolutions starting from the bottom
  
  int errtype = -1;
  
  arma::vec timings = arma::zeros(n_actual_groups);
  arma::vec timings2 = arma::zeros(n_actual_groups);
  
  arma::vec ll_y = arma::zeros(y.n_rows);
  //Rcpp::Rcout << "about to enter for loop " << endl;
  //Rcpp::Rcout << u_by_block_groups.n_elem << endl;
  for(int g=0; g<n_actual_groups; g++){
    //Rcpp::Rcout << g << endl;
    start = std::chrono::steady_clock::now();
//***#pragma omp parallel for 
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      arma::vec w_x = w.rows(indexing(u));
      
      if(parents(u).n_elem == 0){
        arma::mat Kcc = mvCovAG20107(coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
        data.Kcc(u) = Kcc;
        try{
          data.Kxx_invchol(u) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
          data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
          data.Rcc_invchol(u) = data.Kxx_invchol(u); 
          data.w_cond_prec(u) = data.Kxx_inv(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
          data.logdetCi_comps(u) = arma::accu(log(data.ccholprecdiag(u)));
          data.loglik_w_comps(u) = (indexing(u).n_elem+.0) 
            * hl2pi -.5 * data.wcore(u);
          //end = std::chrono::steady_clock::now();
          //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          
        } catch(...){
          errtype = 1;
          throw 1;
        }
      } else {
        int last_par = parents(u)(parents(u).n_elem - 1);
        //arma::mat LAi = Kxx_invchol(last_par);
        
        mvCovAG20107_inplace(data.Kxc(u), coords, qvblock_c, parents_indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, false);
        arma::vec w_pars = w.rows(parents_indexing(u));
        
        data.w_cond_mean_K(u) = data.Kxc(u).t() * data.Kxx_inv(last_par);
        w_x -= data.w_cond_mean_K(u) * w_pars;
        //Rcpp::Rcout << "done for branch " << endl;
        if(res_is_ref(g) == 1){
          
          arma::mat Kcc = mvCovAG20107(coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
          data.Kcc(u) = Kcc;
          //try {
              //Rcpp::Rcout << "branch 1" << endl;
            //Rcpp::Rcout << arma::size(Kcc) << " " << arma::size(data.w_cond_mean_K(u)) <<
             // " " << arma::size(data.Kxc(u)) << endl;
            
            data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(
              Kcc - data.w_cond_mean_K(u) * data.Kxc(u)), "lower")));
            
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            //Rcpp::Rcout << "branch 2" << endl;
            //if(children(u).n_elem > 0){
              //Rcpp::Rcout << "branch 3 with children" << endl;
              //if(limited_tree){
              //  data.Kxx_inv(u) = arma::inv_sympd(Kcc);
              //} else {
              //Rcpp::Rcout << arma::size(data.Kxx_invchol(u)) << " " << 
              //  arma::size(data.Kxx_invchol(last_par)) << " " << 
              //  arma::size(data.w_cond_mean_K(u)) << " " << 
              //  arma::size(data.Rcc_invchol(u)) << endl;
              
                invchol_block_inplace_direct(data.Kxx_invchol(u), data.Kxx_invchol(last_par), 
                                             data.w_cond_mean_K(u), data.Rcc_invchol(u));
                
                //Rcpp::Rcout << "aand ";
                data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
                //Rcpp::Rcout << "done with Kxx_inv" << endl;
              //}
              //Rcpp::Rcout << "branch 5" << endl;
              
              data.has_updated(u) = 1;
            //}
            
            //Rcpp::Rcout << "other comps ";
            data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);
            
            data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
            data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
            data.logdetCi_comps(u) = arma::accu(log(data.ccholprecdiag(u)));
            data.loglik_w_comps(u) = (indexing(u).n_elem+.0) 
              * hl2pi -.5 * data.wcore(u);
        } else {
          // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
          // all observed locations are in these sets.
          // we integrate out the latent effect and use the output likelihood instead.
          
        //start = std::chrono::steady_clock::now();
        //Rcpp::Rcout << arma::size(data.w_cond_mean_K(u)) << " " << arma::size(data.Kxc(u)) << endl;
          data.wcore(u) = 0;
          arma::uvec ones = arma::ones<arma::uvec>(1);
          for(int ix=0; ix<indexing(u).n_elem; ix++){
            if(na_1_blocks(u)(ix) == 1){ 
              // compute likelihood contribution of this observation
              arma::uvec uix = ones * indexing(u)(ix);
              //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
              int first_ix = ix;
              int last_ix = ix;
              //start = std::chrono::steady_clock::now();
              arma::mat Kcc = mvCovAG20107(coords, qvblock_c, uix, uix, ai1, ai2, phi_i, thetamv, Dmat, true);
              //end = std::chrono::steady_clock::now();
              //timings(5) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

              //start = std::chrono::steady_clock::now();
              param_data.Ddiag(u)(ix) = arma::conv_to<double>::from(
                data.w_cond_mean_K(u).rows(first_ix, last_ix) * data.Kxc(u).cols(first_ix, last_ix) );
              double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv_long(indexing(u)(ix));
              double ytilde =  
                arma::conv_to<double>::from(y(indexing(u)(ix)) - XB.row(indexing(u)(ix)));// - Zw.row(indexing_obs(u)(i)));
              ll_y(indexing(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
              
            }
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    timings(g) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  if(errtype > 0){
    Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << endl;
    Rcpp::Rcout << "ai1: " << ai1.t() << endl
                << "ai2: " << ai2.t() << endl
                << "phi_i: " << phi_i.t() << endl
                << "thetamv: " << thetamv.t() << endl
                << "and Dmat: " << Dmat << endl;
    Rcpp::Rcout << " -- auto rejected and proceeding." << endl;
    return false;
  }
  
  //Rcpp::Rcout << "timings: " << timings.t() << endl;
  //Rcpp::Rcout << "total timings: " << arma::accu(timings) << endl;
  data.logdetCi = arma::accu(data.logdetCi_comps.subvec(0, n_blocks-1));
  data.ll_y_all = arma::accu(ll_y);
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps.subvec(0, n_blocks-1)) + data.ll_y_all;
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_comps_w_std] " << errtype << " "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n"
                << timings.t() << endl
                << timings2.t() << endl;
  }
  
  return true;
}

void SpamTreeMVdevel::deal_with_w(bool need_update){
  // Gibbs samplers:
  // S=standard gibbs (cheapest), 
  // P=residual process, 
  // R=residual process using recursive functions
  // Back-fitting plus IRLS:
  // I=back-fitting and iterated reweighted least squares
  gibbs_sample_w_std(true);
  
  /*
 switch(use_alg){
 case 'S':
 gibbs_sample_w_std();
 break;
 case 'P':
 gibbs_sample_w_rpx();
 break;*/
  //case 'I': 
  //  bfirls_w();
  //  break;
  //}
}


void SpamTreeMVdevel::gibbs_sample_w_std(bool need_update){
  // backward sampling?
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] sampling " << endl;
  }
  
  bigrnorm = arma::randn(coords.n_rows);
  theta_transform(param_data);
  arma::vec timings = arma::zeros(8);
  
  int errtype = -1;
  //Rcpp::Rcout << res_is_ref << endl;
  //Rcpp::Rcout << "n groups " << n_actual_groups << endl;
  
  for(int g=n_actual_groups-1; g>=0; g--){
    //***#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //Rcpp::Rcout << g << " " << u << " " << indexing(u).n_elem << endl;
      //Rcpp::Rcout << res_is_ref.t() << endl;
      //arma::mat Dtaui = arma::diagmat(tausq_inv_long.rows(indexing(u)));
      //arma::sp_mat Ztausq = spmat_by_diagmat(Zblock(u).t(), tausq_inv_long.rows(indexing(u)));
      if(res_is_ref(g) == 0){
        //if(blocks_predicting(i)==0){
          w.rows(indexing(u)) = param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        //}
      } else {
        //start = std::chrono::steady_clock::now();
        // reference set. full dependence
        arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
        // Sigi_p
        
        arma::mat Sigi_tot = param_data.w_cond_prec(u);
        if(parents(u).n_elem > 0){
          param_data.AK_uP_all(u) = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
        }
        
        // this is a reference set therefore it must have children
        //Rcpp::Rcout << children(u) << endl;
        //Rcpp::Rcout << arma::size(param_data.Sigi_children(u)) << endl;
        Sigi_tot += arma::sum(param_data.Sigi_children(u), 2);
        
        //Sigi_tot.diag() += tausq_inv_long.rows(indexing(u)); //Ztausq * Zblock(u);
        // go check out observations
        
        //int up = parents(u)(p);
        int cc = children(u).n_elem-1;
        int uc = children(u)(cc);
        int c_ix = children(u).n_elem-1;//this_is_jth_child(uc)(cc);
        
        
        arma::vec w_pars_of_child = w.rows(parents_indexing(uc));
        arma::vec w_opars_of_child = w_pars_of_child.rows(u_is_which_col_f(u)(c_ix)(1));
        
        arma::mat Hthis = param_data.w_cond_mean_K(uc).cols(u_is_which_col_f(u)(c_ix)(0));
        arma::mat Hother = param_data.w_cond_mean_K(uc).cols(u_is_which_col_f(u)(c_ix)(1));
        //arma::mat Kxcother = param_data.Kxc(u).rows(u_is_which_col_f(up)(c_ix)(1));
        
        arma::sp_mat tsqD = spmat_by_diagmat(Ib(uc), tausq_inv_long(indexing(uc)));
        arma::mat message_to_parent =  Hthis.t() * tsqD * Hthis;
        
        arma::mat meanmessage = Hthis.t() * tsqD * (y.rows(indexing(uc)) - XB.rows(indexing(uc)) - Hother * w_opars_of_child);
        
        Sigi_tot += message_to_parent;
        Smu_tot += meanmessage;
      
      
        // ------------------------- 
        
        
        start = std::chrono::steady_clock::now();
        if(parents(u).n_elem>0){
          Smu_tot += param_data.AK_uP_all(u).t() * w.rows(parents_indexing(u));//param_data.w_cond_mean(u);
          // for updating the parents that have this block as child
        }
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        
        //if(children(u).n_elem > 0){
          Smu_tot += arma::sum(param_data.Smu_children(u), 1);
        //}

        
        try {
          param_data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        } catch(...){
          errtype = 10;
        }
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        arma::mat Sigi_chol = param_data.Sigi_chol(u);
        
        arma::vec rnvec = bigrnorm.rows(indexing(u));//arma::randn(indexing(u).n_elem);//arma::vectorise(rand_norm_mat.rows(indexing(u)));
        
        w.rows(indexing(u)) = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        // message the parents
        if(parents(u).n_elem > 0){
          //start = std::chrono::steady_clock::now();
          if(need_update){
            param_data.AK_uP_u_all(u) = param_data.AK_uP_all(u) * param_data.w_cond_mean_K(u);
          }
          
          arma::vec w_par = w.rows(parents_indexing(u));
          
          for(int p=0; p<parents(u).n_elem; p++){
            // up is the parent
            int up = parents(u)(p);
            
            int c_ix = this_is_jth_child(u)(p);
            
            if(need_update){
              param_data.Sigi_children(up).slice(c_ix) = 
                param_data.AK_uP_u_all(u).submat(u_is_which_col_f(up)(c_ix)(0), 
                                       u_is_which_col_f(up)(c_ix)(0));
            }
            
            //start = std::chrono::steady_clock::now();
            param_data.Smu_children(up).col(c_ix) = 
              param_data.AK_uP_all(u).rows(u_is_which_col_f(up)(c_ix)(0)) * w.rows(indexing(u)) -  
              param_data.AK_uP_u_all(u).submat(u_is_which_col_f(up)(c_ix)(0), 
                                     u_is_which_col_f(up)(c_ix)(1)) * w_par.rows(u_is_which_col_f(up)(c_ix)(1));
            
          }
          //end = std::chrono::steady_clock::now();
          //timings(7) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        } // endif
      
      } // endif reference
      
    } // members loop
  } // group loop
  
  if(errtype > 0){
    Rcpp::Rcout << errtype << endl;
    throw 1;
  }
  
  if(verbose){
    Rcpp::Rcout << "timings: " << timings.t() << endl;
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}


void SpamTreeMVdevel::predict(bool theta_update=true){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  predict_std(true, theta_update);
}

void SpamTreeMVdevel::predict_std(bool sampling=true, bool theta_update=true){
  start_overall = std::chrono::steady_clock::now();
  message("[predict_std] start. ");
  
  //arma::vec timings = arma::zeros(5);
  theta_transform(param_data);
  
  //arma::vec timings = arma::zeros(4);
  
  // cycle through the resolutions starting from the bottom
  //arma::uvec predicting_blocks = arma::find(block_ct_obs == 0);
//***#pragma omp parallel for
  for(int i=0; i<blocks_predicting.n_elem; i++){
    int u = blocks_predicting(i);
    // meaning this block must be predicted
    //start = std::chrono::steady_clock::now();
    //start = std::chrono::steady_clock::now();
    if(theta_update){
      
      mvCovAG20107_inplace(param_data.Kxc(u), coords, qvblock_c, 
                           parents_indexing(u), indexing(u), 
                           ai1, ai2, phi_i, thetamv, Dmat, false);
      
      int u_par = parents(u)(parents(u).n_elem - 1);
      //start = std::chrono::steady_clock::now();
      // updating the parent
      if(param_data.has_updated(u) == 0){

          // parents of this block have no children so they have not been updated
          int u_gp = parents(u_par)(parents(u_par).n_elem - 1);
          invchol_block_inplace_direct(param_data.Kxx_invchol(u_par), param_data.Kxx_invchol(u_gp), 
                                       param_data.w_cond_mean_K(u_par), param_data.Rcc_invchol(u_par));
          param_data.Kxx_inv(u_par) = param_data.Kxx_invchol(u_par).t() * param_data.Kxx_invchol(u_par);
        
      }
      param_data.w_cond_mean_K(u) = param_data.Kxc(u).t() * param_data.Kxx_inv(u_par);
    }
    
    arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
    //end = std::chrono::steady_clock::now();
    //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    //start = std::chrono::steady_clock::now();
    if(sampling){
      arma::uvec ones = arma::ones<arma::uvec>(1);
      for(int ix=0; ix<indexing(u).n_elem; ix++){
        arma::uvec uix = ones * indexing(u)(ix);
        int first_ix = ix;
        int last_ix = ix;
        arma::mat Kcc = mvCovAG20107(coords, qvblock_c, uix, uix, ai1, ai2, phi_i, thetamv, Dmat, true);
        //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
        arma::mat Rchol;
        arma::mat Ktemp = Kcc - 
          param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * 
          param_data.Kxc(u).cols(first_ix, last_ix);
        try {
          Rchol = arma::chol(arma::symmatu(Ktemp), "lower");
        } catch(...){
          Rcpp::Rcout << Ktemp << endl;
          Ktemp(0,0) = 0;
          Rchol = arma::zeros(1,1);
        }
        //arma::vec rnvec = arma::randn(1);
        
        w.row(indexing(u)(ix)) = param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * w_par + Rchol * bigrnorm(indexing(u)(ix));
      }
    } else {
      w.rows(indexing(u)) = param_data.w_cond_mean_K(u) * w_par;
    }
  }
  
  //Rcpp::Rcout << "prediction timings: " << timings.t() << endl;
  //Rcpp::Rcout << arma::accu(timings) << endl;
  
  //Zw = w;//armarowsum(Z % w);
  
  if(verbose){
    
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict_std] done "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. \n";
    
  }
  
}

void SpamTreeMVdevel::deal_with_beta(){
  //switch(use_alg){
  //case 'I':
  //  bfirls_beta();
  //  break;
  //default:
  gibbs_sample_beta();
  //  break;
  //}
}

void SpamTreeMVdevel::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X_available.rows(ix_by_q_a(j)).t() * 
      (y_available.rows(ix_by_q_a(j)) - w.rows(ix_by_q_a(j)));
    
    arma::vec Bmu = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j);
    Bcoeff.col(j) = Bmu + Sigma_chol_Bcoeff.t() * arma::randn(p);
    //Rcpp::Rcout << "j: " << j << endl
    //     << Bmu << endl;
    
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * Bcoeff.col(j);
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void SpamTreeMVdevel::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  for(int j=0; j<q; j++){
    arma::vec Zw_availab = w.rows(na_ix_all);
    arma::vec XB_availab = XB.rows(na_ix_all);
    arma::mat yrr = y_available.rows(ix_by_q_a(j)) - XB_availab.rows(ix_by_q_a(j)) - Zw_availab.rows(ix_by_q_a(j));
    double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
    double aparam = 2.0001 + ix_by_q_a(j).n_elem/2.0;
    double bparam = 1.0/( 1.0 + .5 * bcore );
    
    Rcpp::RNGScope scope;
    tausq_inv(j) = R::rgamma(aparam, bparam);
    // fill all (not just available) corresponding to same variable.
    tausq_inv_long.rows(ix_by_q(j)).fill(tausq_inv(j));
    
    if(verbose){
      end = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[gibbs_sample_tausq] " << j << ", "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                  << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                  << endl;
    }
  }
}


void SpamTreeMVdevel::update_ll_after_beta_tausq(){
  int g = n_actual_groups - 1;
  arma::vec ll_y = arma::zeros(w.n_rows);
  for(int i=0; i<u_by_block_groups(g).n_elem; i++){
    int u = u_by_block_groups(g)(i);
    
    for(int ix=0; ix<indexing(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv_long(indexing(u)(ix));
        //Rcpp::Rcout << "llcontrib 2" << endl;
        double ytilde =  
          arma::conv_to<double>::from(y(indexing(u)(ix)) - XB.row(indexing(u)(ix)));// - Zw.row(indexing_obs(u)(i)));
        ll_y(indexing(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
      }
    }
  }
  param_data.ll_y_all = arma::accu(ll_y);
  param_data.loglik_w = param_data.logdetCi + 
    arma::accu(param_data.loglik_w_comps.subvec(0, n_blocks-1)) + 
    param_data.ll_y_all;
  
}

void SpamTreeMVdevel::theta_update(SpamTreeMVDataDevel& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void SpamTreeMVdevel::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void SpamTreeMVdevel::beta_update(const arma::mat& new_beta){ 
  Bcoeff = new_beta;
}

void SpamTreeMVdevel::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}