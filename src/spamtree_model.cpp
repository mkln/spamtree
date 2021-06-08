#define ARMA_DONT_PRINT_ERRORS
#include "spamtree_model.h"

SpamTreeMV::SpamTreeMV(){
  
}

SpamTreeMV::SpamTreeMV(
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
  
  verbose = v;
  debug = debugging;
  
  start_overall = std::chrono::steady_clock::now();
  if(verbose & debug){
    Rcpp::Rcout << "~ SpamTreeMV initialization.\n";
  }
  
  
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
  
  if(verbose){
    Rcpp::Rcout << n_gibbs_groups << " tree branching levels. " << "\n";
  }
  

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
  
  if(verbose){
    Rprintf("%d observed locations, %d to predict, %d total\n",
            n, y.n_elem-n, y.n_elem);
  }
  
  
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
    //Rcpp::Rcout << " -> " << ix_by_q(j).n_elem << "\n";
    //Rcpp::Rcout << arma::size(X) << " " << arma::size(XB) << " " << "\n";
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * beta_in;
    Bcoeff.col(j) = beta_in;
  }
  w                = w_in;
  
  predicting = true;
  
  // now elaborate
  if(verbose & debug){
    Rcpp::Rcout << "SpamTreeMV::SpamTreeMV : init_indexing()\n";
  }
  
  init_indexing();
  
  if(verbose & debug){
    Rcpp::Rcout << "SpamTreeMV::SpamTreeMV : na_study()\n";
  }
  
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
  
  if(verbose & debug){
    Rcpp::Rcout << "SpamTreeMV::SpamTreeMV : make_gibbs_groups()\n";
  }
  
  make_gibbs_groups();
  
  if(verbose & debug){
    Rcpp::Rcout << "SpamTreeMV::SpamTreeMV : init_finalize()\n";
  }
  
  init_finalize();
  
  init_model_data(theta_in);
  
  //find_common_descendants();
  
  // initialize covariance model
  if(dd==3){
    covariance_model = 2;
  } else {
    covariance_model = -1; // let it be decided auto
  }
  covpars = CovarianceParams(dd, q, covariance_model);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "SpamTreeMV::SpamTreeMV initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}

void SpamTreeMV::make_gibbs_groups(){
  
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] check\n";
  }
  
  // checks -- errors not allowed. use check_groups.cpp to fix errors.
  for(int g=0; g<n_gibbs_groups; g++){
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(indexing(u).n_elem > 0){ //**
          
          for(unsigned int pp=0; pp<parents(u).n_elem; pp++){
            if(block_groups(parents(u)(pp)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              throw 1;
            }
          }
          for(unsigned int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              throw 1;
            }
          }
        }
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] manage groups for fitting algs\n";
  }
  
  arma::field<arma::vec> u_by_block_groups_temp (n_gibbs_groups);
  arma::uvec there_are_blocks = arma::zeros<arma::uvec>(n_gibbs_groups);
  /// create list of groups for gibbs
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups_temp(g) = arma::zeros(0);
    
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(block_ct_obs(u) > 0){ //**
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups_temp(g) = arma::join_vert(u_by_block_groups_temp(g), uhere);
        } 
      }
    }
    
    if(u_by_block_groups_temp(g).n_elem > 0){
      there_are_blocks(g) = 1;
    } else {
      there_are_blocks(g) = 0;
    }
  }
  
  n_actual_groups = arma::accu(there_are_blocks);
  u_by_block_groups = arma::field<arma::vec> (n_actual_groups);
  for(int g=0; g<n_actual_groups; g++){
    u_by_block_groups(g) = u_by_block_groups_temp(g);
    int res_num_blocks = u_by_block_groups(g).n_elem;
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] list nonempty, predicting, and reference blocks\n";
  }
  
  blocks_not_empty = arma::find(block_ct_obs > 0);
  blocks_predicting = arma::find(block_ct_obs == 0);
  block_is_reference = arma::ones<arma::uvec>(n_blocks);//(blocks_not_empty.n_elem);
  
  // resolutions that are not reference
  arma::uvec which_not_reference = arma::find(res_is_ref == 0);
  int ne=0;
  int pr=0;
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(block_ct_obs(u) > 0){
      blocks_not_empty(ne) = u;
      ne ++;
      // determine if this block is reference
      for(unsigned int r=0; r<which_not_reference.n_elem; r++){
        // non-reference include predictions but we are not including those in u_by_block_groups
        if(which_not_reference(r) < u_by_block_groups.n_elem){
          arma::uvec lookforit = arma::find(u_by_block_groups(which_not_reference(r)) == u);
          if(lookforit.n_elem > 0){
            // found block in non-reference last-res
            block_is_reference(u) = 0;
            break;
          }
        }
      }
    } else {
      blocks_predicting(pr) = u;
      pr ++;
      block_is_reference(u) = 0;
    }
  }
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] done.\n";
  }
  
}

void SpamTreeMV::na_study(){
  
  block_ct_obs = arma::zeros<arma::uvec>(n_blocks);//(y_blocks.n_elem);
  
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    arma::vec yvec = y.rows(indexing(i));
    arma::uvec y_not_na = arma::find_finite(yvec);
    block_ct_obs(i) = y_not_na.n_elem;
  }
  
}

void SpamTreeMV::init_indexing(){
  //arma::field<arma::uvec> qvblock(n_blocks);
  Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  if(verbose & debug){
    Rcpp::Rcout << "init_indexing\n"; 
  }
  //pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    //Rcpp::Rcout << u << "\n";
    if(parents(u).n_elem > 0){
      //Rcpp::Rcout << "n. parents " << parents(u).n_elem << "\n";
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(unsigned int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
    
    if(children(u).n_elem > 0){
      arma::field<arma::uvec> cixs(children(u).n_elem);
      for(unsigned int ci=0; ci<children(u).n_elem; ci++){
        //Rcpp::Rcout << "n. children " << children(u).n_elem << "\n";
        //Rcpp::Rcout << "--> " << children(u)(ci) << "\n";
        cixs(ci) = indexing(children(u)(ci));//arma::find( blocking == children(u)(ci)+1 ); // children are 0-indexed 
      }
      children_indexing(u) = field_v_concat_uv(cixs);
    }
    
    //qvblock(u) = arma::field<arma::uvec>(indexing(u).n_elem);
    //Zblock(u) = ZifyMV( Z.rows(indexing(u)), gix_block.elem(indexing(u)));
    //Rcpp::Rcout << "doing Z" << "\n";
    //Rcpp::Rcout << arma::mat(Zblock(u)) << "\n";
  }
  
}

void SpamTreeMV::init_finalize(){
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_finalize] dim_by_parent, parents_coords, children_coords\n";
  }
  
  //pragma omp parallel for //**
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
      dim_by_parent(u) = arma::zeros(parents(u).n_elem + 1);
      for(unsigned int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;//coords_blocks(parents(u)(j)).n_rows;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_finalize] u_is_which_col_f\n";
  }
  
  //pragma omp parallel for // **
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    
    //if(indexing(u).n_elem > 0){//***
    // children-parent relationship variables
    u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (children(u).n_elem);
    this_is_jth_child(u) = arma::zeros<arma::uvec> (parents(u).n_elem);
    
    for(unsigned int c=0; c<children(u).n_elem; c++){
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
    
    if(block_ct_obs(u) > 0){
      for(unsigned int p=0; p<parents(u).n_elem; p++){
        int up = parents(u)(p);
        arma::uvec cuv = arma::find(children(up) == u, 1, "first");
        this_is_jth_child(u)(p) = cuv(0); 
      }
    }
    //}
  }
}

void SpamTreeMV::init_model_data(const arma::vec& theta_in){
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_model_data]\n";
  }
  
  // data for metropolis steps and predictions
  // block params
  
  param_data.has_updated   = arma::zeros<arma::uvec>(n_blocks);
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.Kxc           = arma::field<arma::mat> (n_blocks);
  param_data.Kxx_inv       = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec_noref   = arma::field<arma::field<arma::mat> > (n_blocks);
  param_data.Kxx_invchol = arma::field<arma::mat> (n_blocks); // storing the inv choleskys of {parents(w), w} (which is parent for children(w))
  param_data.Rcc_invchol = arma::field<arma::mat> (n_blocks); 
  param_data.ccholprecdiag = arma::field<arma::vec> (n_blocks);
  
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.Sigi_chol_noref = arma::field<arma::field<arma::mat> >(n_blocks);
  param_data.AK_uP_all = arma::field<arma::mat> (n_blocks);
  param_data.AK_uP_u_all = arma::field<arma::mat> (n_blocks);
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
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
    if(block_ct_obs(u)>0){
      param_data.Kxx_invchol(u) = arma::zeros(parents_indexing(u).n_elem + indexing(u).n_elem, //Kxx_invchol(last_par).n_rows + Kcc.n_rows, 
                             parents_indexing(u).n_elem + indexing(u).n_elem);//Kxx_invchol(last_par).n_cols + Kcc.n_cols);
      //Kxx_chol(u) = Kxx_invchol(u);
    }
    
    param_data.w_cond_mean_K(u) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
    param_data.Kxc(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    param_data.ccholprecdiag(u) = arma::zeros(indexing(u).n_elem);
    
    if(block_is_reference(u) == 1){
      param_data.w_cond_prec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      param_data.Rcc_invchol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      param_data.Sigi_chol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    } else {
      if(block_ct_obs(u) > 0){
        param_data.w_cond_prec_noref(u) = arma::field<arma::mat>(indexing(u).n_elem);
        param_data.Sigi_chol_noref(u) = arma::field<arma::mat>(indexing(u).n_elem);
        for(unsigned int j=0; j<indexing(u).n_elem; j++){
          param_data.w_cond_prec_noref(u)(j) = arma::zeros(q, q);
          param_data.Sigi_chol_noref(u)(j) = arma::zeros(q, q);
        }
      }
    }
    param_data.AK_uP_all(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    param_data.AK_uP_u_all(u) = param_data.AK_uP_all(u) * param_data.w_cond_mean_K(u);
  }
  
  // stuff for marginalized likelihood
  //param_data.Ciblocks = arma::field<arma::mat>(n_blocks, n_blocks);
  //param_data.Hblocks = arma::field<arma::mat>(n_blocks, n_blocks);
  //param_data.Riblocks = arma::field<arma::mat>(n_blocks);
  
  alter_data                = param_data; 
  
  //Rcpp::Rcout << "init_model_data: indexing elements: " << indexing.n_elem << "\n";
  
}


arma::mat SpamTreeMV::Ci_ij(int block_i, int block_j, SpamTreeMVData& data){
  // this is a (slow) utility to compute the ij block of the precision matrix
  arma::vec cparams = data.theta;
  covpars.transform(cparams);
  
  arma::uvec common_descendants = block_common_descendants(block_i, block_j); 

  //Rcpp::Rcout << "common " << "\n";
  //Rcpp::Rcout << common_descendants << "\n";
  
  int n_cd = common_descendants.n_elem;
  
  if(n_cd > 0){
    arma::mat result;
    
    if((block_i == block_j) & (block_is_reference(block_i)==0)){
      // block diagonal at non-reference blocks, save memory by not saving all the zeros. 
      // save as column vector representing the diagonal here
      result = arma::zeros(indexing(block_i).n_elem, 1);
      for(unsigned int i=0; i<data.w_cond_prec_noref(block_i).n_elem; i++){
        result(i, 0) = data.w_cond_prec_noref(block_i)(i)(0,0);
      }
      return result;
    } else {
      // one of the two is not reference, proceed in full
      result = arma::zeros(indexing(block_i).n_elem, indexing(block_j).n_elem);
      for(int cd=0; cd < n_cd; cd++){
        int block_k = common_descendants(cd);
        //Rcpp::Rcout << "descendant: " << block_k << "\n";
        
        //Rcpp::Rcout << indexing(block_k).n_elem << " " << indexing(block_i).n_elem << " " << indexing(block_j).n_elem << "\n";
        
        arma::mat Iki = arma::zeros(indexing(block_k).n_elem, indexing(block_i).n_elem);
        if(block_k == block_i){
          Iki.diag() += 1;
        }
        arma::mat Ikj = arma::zeros(indexing(block_k).n_elem, indexing(block_j).n_elem);
        if(block_k == block_j){
          Ikj.diag() += 1;
        }
        
        int n_loc_par = parents_indexing(block_k).n_elem; //0;
        //for(int p = 0; p<parents(block_k).n_elem; p++){
        //  n_loc_par += indexing( parents(block_k)(p) ).n_elem;
        //}
        
        //arma::uvec parents_indexing = arma::zeros<arma::uvec>(n_loc_par);
        // whichone: identifies which parent it is
        
        //arma::uvec parents_whichone = arma::zeros<arma::uvec>(n_loc_par);
        //int start_ix = 0;
        //Rcpp::Rcout << "building parent locations " << "\n";
        //for(int p = 0; p<parents(block_k).n_elem; p++){
        //Rcpp::Rcout << "parent " << parents(block_k)(p) << " is n. " << p << "\n";
        //  arma::uvec block_here = indexing(parents(block_k)(p));
        //Rcpp::Rcout << "size: " << block_here.n_elem << "\n";
        //  int n_par_block = block_here.n_elem;
        //parents_indexing.rows(start_ix, start_ix + n_par_block-1) = block_here;
        //  parents_whichone.rows(start_ix, start_ix + n_par_block-1) += p;
        //  start_ix += n_par_block;
        //}
        
        //arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(block_k), indexing(block_k), ai1, ai2, phi_i, thetamv, Dmat, true);
        //arma::mat Kxxi = arma::inv_sympd(Covariancef(coords, qvblock_c, parents_indexing(block_k), parents_indexing(block_k), ai1, ai2, phi_i, thetamv, Dmat, true));
        //arma::mat Kcx = Covariancef(coords, qvblock_c, indexing(block_k), parents_indexing(block_k), ai1, ai2, phi_i, thetamv, Dmat, false);
        arma::mat H_k = data.w_cond_mean_K(block_k); //Kcx * Kxxi;
        arma::mat Ri_k;
        
        if(block_is_reference(block_k)){
          Ri_k = data.w_cond_prec(block_k);
        } else {
          Ri_k = arma::zeros(data.w_cond_prec_noref(block_k).n_elem, 1); 
          for(unsigned int s=0; s<Ri_k.n_rows; s++){
            //Rcpp::Rcout << arma::size(data.w_cond_prec_noref(block_k)(s)) << "\n";
            Ri_k(s, 0) = data.w_cond_prec_noref(block_k)(s)(0,0);
          }
        }
        
        //arma::uvec find_k_in_children_i = arma::find(i_descendants == block_k, 1, "first");
        //arma::uvec find_k_in_children_j = arma::find(j_descendants == block_k, 1, "first");
        
        //arma::uvec find_i_in_parents_k = arma::find(parents(block_k) == block_i, 1, "first");
        //arma::uvec find_j_in_parents_k = arma::find(parents(block_k) == block_j, 1, "first");
        
        //int pn_i_k = find_i_in_parents_k.n_elem > 0 ? arma::conv_to<int>::from( find_i_in_parents_k ) : -1;
        //int pn_j_k = find_j_in_parents_k.n_elem > 0 ? arma::conv_to<int>::from( find_j_in_parents_k ) : -1;
        
        //arma::uvec pn_i = arma::find(parents_whichone == pn_i_k);
        //arma::uvec pn_j = arma::find(parents_whichone == pn_j_k);
        
        // 
        arma::uvec find_k_in_children_i = arma::find(children(block_i) == block_k, 1, "first");
        arma::uvec find_k_in_children_j = arma::find(children(block_j) == block_k, 1, "first");
        
        int cn_i_k = find_k_in_children_i.n_elem > 0 ? arma::conv_to<int>::from( find_k_in_children_i ) : -1;
        int cn_j_k = find_k_in_children_j.n_elem > 0 ? arma::conv_to<int>::from( find_k_in_children_j ) : -1;
        
        arma::uvec i_colsel_k;
        if(cn_i_k > -1){
          i_colsel_k = u_is_which_col_f(block_i)(cn_i_k)(0);
        } else {
          i_colsel_k.reset();
        }
        arma::uvec j_colsel_k;
        if(cn_j_k > -1){
          j_colsel_k = u_is_which_col_f(block_j)(cn_j_k)(0);
        } else {
          j_colsel_k.reset();
        }
        
        arma::mat H_k_owed_i = H_k.cols(i_colsel_k); //H_k.cols(pn_i);
        arma::mat H_k_owed_j = H_k.cols(j_colsel_k); //H_k.cols(pn_j);
        
        //if(pn_i.n_elem == 0){
        //Rcpp::Rcout << "No i" << "\n";
        //  H_k_owed_i = arma::zeros(arma::size(Iki));
        //}
        //if(pn_j.n_elem == 0){
        //Rcpp::Rcout << "No j" << "\n";
        //  H_k_owed_j = arma::zeros(arma::size(Ikj));
        //}
        
        //Rcpp::Rcout << cd << " parset size: " << parents_indexing.n_elem << " " << arma::size(H_k) << "\n";
        //Rcpp::Rcout << parents_whichone.t() << "\n";
        //Rcpp::Rcout << "pn: " << pn_i_k << " " << pn_j_k << "\n";
        //Rcpp::Rcout << pn_i << "\n" << pn_j << "\n";
        //Rcpp::Rcout << arma::size(Iki) << " " << arma::size(Ikj) << "\n";
        //Rcpp::Rcout << arma::size(H_k_owed_i) << " " << arma::size(H_k_owed_j) << "\n";
        //Rcpp::Rcout << "--" << "\n";
        
        arma::mat IminusH_ki, IminusH_kj;
        if(H_k_owed_i.n_cols > 0){//(block_k != block_i){
          IminusH_ki = Iki-H_k_owed_i;
        } else {
          IminusH_ki = Iki;
        }
        if(H_k_owed_j.n_cols > 0){ //(block_k != block_j){
          IminusH_kj = Ikj-H_k_owed_j;
        } else {
          IminusH_kj = Ikj;
        }
        
        if(block_is_reference(block_k)){
          result += IminusH_ki.t() * Ri_k * IminusH_kj;
        } else {
          arma::mat Rik_IminusH = arma::zeros(Ri_k.n_rows, IminusH_kj.n_cols);
          for(unsigned int s=0; s<Ri_k.n_rows; s++){
            Rik_IminusH.row(s) = Ri_k(s,0) * IminusH_kj.row(s);
          }
          result += IminusH_ki.t() * Rik_IminusH;
        }
        //Rcpp::Rcout << Ri_k << "\n";
        //Rcpp::Rcout << IminusH_ki.t() << "\n";
      }
      return result;
    }
    
  } else {
    // no common descendants = empty
    arma::mat result;
    result.reset();
    return result;
  }
  
}


void SpamTreeMV::find_common_descendants(){
  block_common_descendants = arma::field<arma::uvec>(n_blocks, n_blocks);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int i=0; i<n_blocks; i++){
    int ui = block_names(i) - 1;
    //if(block_is_reference(ui) == 1){
      for(int j=0; j<=i; j++){
        int uj = block_names(j) - 1;
        //if(block_is_reference(uj)){
          arma::uvec i_descendants = arma::join_vert(oneuv * ui, children(ui));
          arma::uvec j_descendants = arma::join_vert(oneuv * uj, children(uj));
          block_common_descendants(ui, uj) = arma::intersect(i_descendants, j_descendants);
        //}
      }
    //}
  }
}

void SpamTreeMV::fill_precision_blocks(SpamTreeMVData& data){
  // this builds a block representation of the marginal likelihood precision
  if(verbose & debug){
    Rcpp::Rcout << "[SpamTreeMV::fill_precision_blocks] start\n";
  }
  for(int i=0; i<n_blocks; i++){
    int ui = block_names(i) - 1;
    //if(block_is_reference(ui) == 1){
      for(int j=0; j<=i; j++){
        int uj = block_names(j) - 1;
        //Rcpp::Rcout << "ui: " << ui << " uj: " << uj << "\n";
        //if(block_is_reference(uj)){
          data.Ciblocks(ui, uj) = Ci_ij(ui, uj, data);
        //}
        if(i == j){
          if(block_is_reference(ui) == 1){
            data.Ciblocks(ui, ui).diag() += tausq_inv_long.rows(indexing(ui));
          } else {
            data.Ciblocks(ui, ui) += tausq_inv_long.rows(indexing(ui));
          }
        } else {
          data.Ciblocks(uj, ui) = arma::trans(data.Ciblocks(ui, uj));
        }
      }
    //}
    
  }
  if(verbose & debug){
    Rcpp::Rcout << "[SpamTreeMV::fill_precision_blocks] done\n";
  }
  
}

void SpamTreeMV::decompose_margin_precision(SpamTreeMVData& data){
  if(verbose & debug){
    Rcpp::Rcout << "[SpamTreeMV::decompose_margin_precision] start\n";
  }
  
  for(int g=n_actual_groups-1; g>=0; g--){
//////***#pragma omp parallel for
    for(unsigned int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      data.Riblocks(u) = data.Ciblocks(u, u);
      arma::mat Dinv;
      
      if(block_is_reference(u) == 1){
        Dinv = arma::inv_sympd(data.Riblocks(u));
      } else {
        Dinv = 1.0/data.Riblocks(u); // this is a vector representing the diagonal of Riblocks(u)
      }
      
      for(unsigned int p=0; p<parents(u).n_elem; p++){
        int pj = parents(u)(p);
        arma::mat HRi_pj = -data.Ciblocks(u, pj).t();
        
        if(block_is_reference(u) == 1){
          data.Hblocks(u, pj) = - Dinv * data.Ciblocks(u, pj); 
        } else {
          data.Hblocks(u, pj) = arma::zeros(indexing(u).n_rows, indexing(pj).n_rows);
          for(unsigned int r=0; r<indexing(u).n_rows; r++){
            data.Hblocks(u, pj).row(r) = - Dinv(r) * data.Ciblocks(u, pj).row(r);
          }
        } 
        
        for(unsigned int g=0; g<parents(u).n_elem; g++){
          int gj = parents(u)(g);
          if(data.Ciblocks(pj, gj).n_rows > 0){
            if(data.Hblocks(u, gj).n_rows > 0){
              data.Ciblocks(pj, gj) = data.Ciblocks(pj, gj) - HRi_pj * data.Hblocks(u, gj);
            }
            data.Ciblocks(gj, pj) = arma::trans(data.Ciblocks(pj, gj));
          }
        }
      }
    }
  }
  if(verbose & debug){
    Rcpp::Rcout << "[SpamTreeMV::decompose_margin_precision] done\n ";  
  }
  
}


void SpamTreeMV::get_loglik_w(SpamTreeMVData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  get_loglik_w_std(data);
}

void SpamTreeMV::get_loglik_w_std(SpamTreeMVData& data){
  start = std::chrono::steady_clock::now();
  if(verbose & debug){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
  
  //arma::uvec blocks_not_empty = arma::find(block_ct_obs > 0);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<blocks_not_empty.n_elem; i++){  
    int u = blocks_not_empty(i);
    
    //arma::mat w_x = arma::vectorise( arma::trans( w.rows(indexing(u)) ) );
    arma::vec w_x = w.rows(indexing(u));
    
    if(parents(u).n_elem > 0){
      w_x -= data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
    }
    
    if(block_is_reference(u) == 1){
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
    } else {
      // really only need the (q=1: diagonals, q>1: block-diagonals) of precision since we assume
      // conditional independence here.
      data.wcore(u) = 0;
      arma::uvec ones = arma::ones<arma::uvec>(1);
      for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
        data.wcore(u) += w_x(ix) * data.w_cond_prec_noref(u)(ix)(0,0) *  w_x(ix);
      }
    }
    data.loglik_w_comps(u) = (indexing(u).n_elem + .0) * hl2pi -.5 * data.wcore(u);
  }
  
  data.logdetCi = arma::accu(data.logdetCi_comps);
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  //print_data(data);
}


bool SpamTreeMV::get_loglik_comps_w(SpamTreeMVData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  return get_loglik_comps_w_std(data);
}

bool SpamTreeMV::get_loglik_comps_w_std(SpamTreeMVData& data){
  start_overall = std::chrono::steady_clock::now();
  
  if(verbose & debug){
    Rcpp::Rcout << "[get_loglik_comps_w_std] start. \n";
  }
  
  // arma::vec timings = arma::zeros(7);
  arma::vec cparams = data.theta;
  covpars.transform(cparams);
  // cycle through the resolutions starting from the bottom
  
  int errtype = -1;
  
  for(int g=0; g<n_actual_groups; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(block_ct_obs(u) > 0){
      //Rcpp::Rcout << "u: " << u << " obs: " << block_ct_obs(u) << " parents: " << parents(u).t() << "\n"; 
      //arma::mat cond_mean;
      //arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
      arma::vec w_x = w.rows(indexing(u));
      //Rcpp::Rcout << "step 1\n";
      
      if(parents(u).n_elem == 0){
        //start = std::chrono::steady_clock::now();
        arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
        
        try{
          data.Kxx_invchol(u) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
          data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
          data.Rcc_invchol(u) = data.Kxx_invchol(u); 
          data.w_cond_prec(u) = data.Kxx_inv(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
          //end = std::chrono::steady_clock::now();
          //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          
        } catch(...){
          errtype = 1;
        }
        data.has_updated(u) = 1;
        
      } else {
        //Rcpp::Rcout << "step 2\n";
        int last_par = parents(u)(parents(u).n_elem - 1);
        //arma::mat LAi = Kxx_invchol(last_par);
        
        Covariancef_inplace(data.Kxc(u), coords, qvblock_c, parents_indexing(u), indexing(u), covpars, false);
        arma::vec w_pars = w.rows(parents_indexing(u));
        data.w_cond_mean_K(u) = data.Kxc(u).t() * data.Kxx_inv(last_par);
        w_x -= data.w_cond_mean_K(u) * w_pars;
        
        if(res_is_ref(g) == 1){
          //start = std::chrono::steady_clock::now();
          arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
          
          try {
              
            data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(
              Kcc - data.w_cond_mean_K(u) * data.Kxc(u)), "lower")));
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            if(children(u).n_elem > 0){
              if(limited_tree){
                data.Kxx_inv(u) = arma::inv_sympd(Kcc);
              } else {
                invchol_block_inplace_direct(data.Kxx_invchol(u), data.Kxx_invchol(last_par), 
                                             data.w_cond_mean_K(u), data.Rcc_invchol(u));
                data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
              }
              
              data.has_updated(u) = 1;
            }
            
            data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);
            data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
            data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              
          } catch(...){
            errtype = 2;
          }
        
          
        } else {
          // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
          //start = std::chrono::steady_clock::now();
          data.wcore(u) = 0;
          arma::uvec ones = arma::ones<arma::uvec>(1);
          for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
            arma::uvec uix = ones * indexing(u)(ix);
            //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
            int first_ix = ix;
            int last_ix = ix;
            //start = std::chrono::steady_clock::now();
            arma::mat Kcc = Covariancef(coords, qvblock_c, uix, uix, covpars, true);
            //end = std::chrono::steady_clock::now();
            //timings(5) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            //start = std::chrono::steady_clock::now();
            arma::mat Kcx_xxi_xc = data.w_cond_mean_K(u).rows(first_ix, last_ix) * data.Kxc(u).cols(first_ix, last_ix);
            //Rcpp::Rcout << "step 3o " << arma::size(cond_mean_K_sub) << " " << arma::size(data.Kxc(u).cols(first_ix, last_ix)) <<  "\n";
            arma::mat Rinvchol;
            try {
              Rinvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc - Kcx_xxi_xc), "lower")));
              
              data.ccholprecdiag(u).subvec(first_ix, last_ix) = Rinvchol.diag();
              
              //data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) = Rinvchol.t() * Rinvchol;
              data.w_cond_prec_noref(u)(ix) = Rinvchol.t() * Rinvchol;
              
              data.wcore(u) += arma::conv_to<double>::from(
                w_x.rows(first_ix, last_ix).t() * 
                  data.w_cond_prec_noref(u)(ix) *
                  w_x.rows(first_ix, last_ix));
              
              //end = std::chrono::steady_clock::now();
              //timings(6) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            } catch(...){
              errtype = 3;
            }
            
          }
          
        }
      }
      
      data.logdetCi_comps(u) = arma::accu(log(data.ccholprecdiag(u)));
      data.loglik_w_comps(u) = (indexing(u).n_elem+.0) 
        * hl2pi -.5 * data.wcore(u);
    }
    
    if(errtype > 0){
      if(verbose & debug){
        Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
        Rcpp::Rcout << "ai1: " << covpars.ai1.t() << "\n"
                    << "ai2: " << covpars.ai2.t() << "\n"
                    << "phi_i: " << covpars.phi_i.t() << "\n"
                    << "thetamv: " << covpars.thetamv.t() << "\n"
                    << "and Dmat: " << covpars.Dmat << "\n";
        Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
      }
      return false;
    }
  }
  
  //Rcpp::Rcout << "timings: " << timings.t() << "\n";
  //Rcpp::Rcout << "total timings: " << arma::accu(timings) << "\n";
  data.logdetCi = arma::accu(data.logdetCi_comps.subvec(0, n_blocks-1));
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps.subvec(0, n_blocks-1));
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_comps_w_std] " << errtype << " "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
  return true;
}

void SpamTreeMV::deal_with_w(bool need_update){
  // Gibbs samplers:
  // S=standard gibbs (cheapest), 
  // P=residual process, 
  // R=residual process using recursive functions
  // Back-fitting plus IRLS:
  // I=back-fitting and iterated reweighted least squares
  gibbs_sample_w_std(need_update);
}


void SpamTreeMV::gibbs_sample_w_std(bool need_update){
  // backward sampling?
  start_overall = std::chrono::steady_clock::now();
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w] sampling " << "\n";
  }
  
  bigrnorm = arma::randn(coords.n_rows);
  
  arma::vec timings = arma::zeros(8);
  
  int errtype = -1;
  
  for(int g=n_actual_groups-1; g>=0; g--){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      //Rcpp::Rcout << "u: " << u << " g: " << g << "\n";
      //arma::mat Dtaui = arma::diagmat(tausq_inv_long.rows(indexing(u)));
      //arma::sp_mat Ztausq = spmat_by_diagmat(Zblock(u).t(), tausq_inv_long.rows(indexing(u)));
      
      //arma::mat AK_uP_all;
      //Rcpp::Rcout << "u " << u << "\n";
      if(res_is_ref(g) == 1){
        //start = std::chrono::steady_clock::now();
        //Rcpp::Rcout << "step 1 " << "\n";
        // reference set. full dependence
        arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
        // Sigi_p
        
        arma::mat Sigi_tot = param_data.w_cond_prec(u);
        if(parents(u).n_elem > 0){
          param_data.AK_uP_all(u) = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
        }
        if(children(u).n_elem > 0){
          Sigi_tot += arma::sum(param_data.Sigi_children(u), 2);
        }
        Sigi_tot.diag() += tausq_inv_long.rows(indexing(u)); //Ztausq * Zblock(u);
        
        try {
          param_data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        } catch(...){
          errtype = 10;
        }
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        start = std::chrono::steady_clock::now();
        if(parents(u).n_elem>0){
          Smu_tot += param_data.AK_uP_all(u).t() * w.rows(parents_indexing(u));//param_data.w_cond_mean(u);
          // for updating the parents that have this block as child
        }
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        //Rcpp::Rcout << "step 2 " << "\n";
        if(children(u).n_elem > 0){
          Smu_tot += arma::sum(param_data.Smu_children(u), 1);
        }
        
        Smu_tot += //Ztausq *
          tausq_inv_long.rows(indexing(u)) %
          ( y.rows(indexing(u)) - XB.rows(indexing(u)) );
        //end = std::chrono::steady_clock::now();
        //timings(2) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        arma::mat Sigi_chol = param_data.Sigi_chol(u);
        //Rcpp::Rcout << "step 4 " << "\n";
        arma::vec rnvec = bigrnorm.rows(indexing(u));//arma::randn(indexing(u).n_elem);//arma::vectorise(rand_norm_mat.rows(indexing(u)));
        
        w.rows(indexing(u)) = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        
        //end = std::chrono::steady_clock::now();
        //timings(3) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
      } else {
        // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
        //start = std::chrono::steady_clock::now();
        arma::vec rnvec = bigrnorm.rows(indexing(u));//arma::randn(indexing(u).n_elem);
        
        arma::uvec ones = arma::ones<arma::uvec>(1);
        arma::vec tsq_Zt_y_XB = //Ztausq * 
          tausq_inv_long.rows(indexing(u)) % (y.rows(indexing(u)) - XB.rows(indexing(u)));
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        arma::mat cond_mean_K_wpar = param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        //end = std::chrono::steady_clock::now();
        //timings(2) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //AK_uP_all = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
        
        //arma::uvec gix_here = gix_block.elem(indexing(u));
        //arma::uvec unique_coords = arma::unique(gix_here);
        
        for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
          //for(int ix=0; ix<unique_coords.n_elem; ix++){
          
          //start = std::chrono::steady_clock::now();
          //arma::uvec uix = ones * indexing(u)(ix);
          //int first_ix = ix;
          //int last_ix = ix;
          //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
          //arma::mat ZZ = Z.row(indexing(u)(ix));
          
          double tsqi = tausq_inv_long(indexing(u)(ix));
          arma::mat Sigi_tot = param_data.w_cond_prec_noref(u)(ix) +//param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) + 
            tsqi;/// * ZZ.t() * ZZ; 
          arma::mat Smu_tot = param_data.w_cond_prec_noref(u)(ix) * //param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) * 
            cond_mean_K_wpar(ix) +//.rows(first_ix, last_ix) +
            tsq_Zt_y_XB(ix);//.rows(first_ix, last_ix);
          //end = std::chrono::steady_clock::now();
          //timings(3) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
          //start = std::chrono::steady_clock::now();
          try {
          param_data.Sigi_chol_noref(u)(ix) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Sigi_tot), "lower")));
          } catch(...){
            errtype = 11;
          }
          arma::mat Sigi_chol = param_data.Sigi_chol_noref(u)(ix);
          
          w.row(indexing(u)(ix)) = Sigi_chol.t() * Sigi_chol * Smu_tot +  
            Sigi_chol.t() * rnvec(ix);//.rows(first_ix, last_ix);
          //end = std::chrono::steady_clock::now();
          //timings(4) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
            arma::mat AK_uP_here = arma::trans(param_data.w_cond_mean_K(u).row(ix)) *//.rows(first_ix, last_ix)) *
              param_data.w_cond_prec_noref(u)(ix);
            param_data.AK_uP_all(u).col(ix) = //.cols(first_ix, last_ix) = 
              AK_uP_here; //param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) *
          
          //start = std::chrono::steady_clock::now();
          
          //9.9621e+06
          //end = std::chrono::steady_clock::now();
          //timings(5) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
      }
      
      // update Sigi_children and Smu_children for parents so we can sample them correctly
      if(parents(u).n_elem > 0){
        //start = std::chrono::steady_clock::now();
        
        if(need_update){
          param_data.AK_uP_u_all(u) = param_data.AK_uP_all(u) * param_data.w_cond_mean_K(u);
        }
        
        arma::vec w_par = w.rows(parents_indexing(u));
        
        //end = std::chrono::steady_clock::now();
        //timings(4) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        // update parents' Sigi_children and Smu_children
        
        for(unsigned int p=0; p<parents(u).n_elem; p++){
          //Rcpp::Rcout << "u: " << u << " " << parents_indexing(u).n_elem << "\n";
          // up is the parent
          int up = parents(u)(p);
          //Rcpp::Rcout << "parent: " << up << "\n";
          //start = std::chrono::steady_clock::now();
          // up has other children, which one is u?
          //arma::uvec cuv = arma::find(children(up) == u,1,"first");
          //int c_ix = cuv(0); 
          int c_ix = this_is_jth_child(u)(p);
          
          //end = std::chrono::steady_clock::now();
          //timings(5) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
          //start = std::chrono::steady_clock::now();
          //Rcpp::Rcout << "filling" << "\n";
          if(need_update){
            param_data.Sigi_children(up).slice(c_ix) = 
              param_data.AK_uP_u_all(u).submat(u_is_which_col_f(up)(c_ix)(0), 
                                     u_is_which_col_f(up)(c_ix)(0));
          }
          //arma::mat AK_uP = param_data.AK_uP_all(u).rows(u_is_which_col_f(up)(c_ix)(0));
          //arma::vec w_par_child_select = w_par.rows(u_is_which_col_f(up)(c_ix)(1));
          //end = std::chrono::steady_clock::now();
          //timings(6) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
          //start = std::chrono::steady_clock::now();
          param_data.Smu_children(up).col(c_ix) = 
            param_data.AK_uP_all(u).rows(u_is_which_col_f(up)(c_ix)(0)) * w.rows(indexing(u)) -  
            param_data.AK_uP_u_all(u).submat(u_is_which_col_f(up)(c_ix)(0), 
                                   u_is_which_col_f(up)(c_ix)(1)) * w_par.rows(u_is_which_col_f(up)(c_ix)(1));
          
          //end = std::chrono::steady_clock::now();
          //timings(7) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        
        
      }
      
    }
  }
  
  if(errtype > 0){
    Rcpp::stop("Error at gibbs_sample_w");
  }
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}


void SpamTreeMV::predict(bool theta_update=true){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  predict_std(true, theta_update);
}

void SpamTreeMV::predict_std(bool sampling=true, bool theta_update=true){
  start_overall = std::chrono::steady_clock::now();
  
  if(verbose & debug){
    Rcpp::Rcout << "predict_std \n";
  }
  //arma::vec timings = arma::zeros(5);
  arma::vec cparams = param_data.theta;
  covpars.transform(cparams);
  
  //arma::vec timings = arma::zeros(4);
  
  // cycle through the resolutions starting from the bottom
  //arma::uvec predicting_blocks = arma::find(block_ct_obs == 0);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<blocks_predicting.n_elem; i++){
    int u = blocks_predicting(i);
    // meaning this block must be predicted
    //start = std::chrono::steady_clock::now();
    //start = std::chrono::steady_clock::now();
    if(theta_update){
      
      Covariancef_inplace(param_data.Kxc(u), coords, qvblock_c, 
                           parents_indexing(u), indexing(u), 
                           covpars, false);
      
      //end = std::chrono::steady_clock::now();
      //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      //end = std::chrono::steady_clock::now();
      //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      // full
      //arma::mat Kpp = xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
      //arma::mat Kppi = arma::inv_sympd(Kpp);
      //arma::mat Kalt = Kcc - Kxc.t() * Kppi * Kxc;
      int u_par = parents(u)(parents(u).n_elem - 1);
      //start = std::chrono::steady_clock::now();
      // updating the parent
      if(param_data.has_updated(u_par) == 0){
        if(limited_tree){
          
          arma::mat Kxx = Covariancef(coords, qvblock_c, indexing(u_par), indexing(u_par), covpars, true);
          param_data.Kxx_inv(u_par) = arma::inv_sympd(Kxx);
        } else {
          // parents of this block have no children so they have not been updated
          int u_gp = parents(u_par)(parents(u_par).n_elem - 1);
          invchol_block_inplace_direct(param_data.Kxx_invchol(u_par), param_data.Kxx_invchol(u_gp), 
                                       param_data.w_cond_mean_K(u_par), param_data.Rcc_invchol(u_par));
          param_data.Kxx_inv(u_par) = param_data.Kxx_invchol(u_par).t() * param_data.Kxx_invchol(u_par);
        }
      }
      //end = std::chrono::steady_clock::now();
      //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      //end = std::chrono::steady_clock::now();
      //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      //start = std::chrono::steady_clock::now();
      //start = std::chrono::steady_clock::now();
      // marginal predictives
      
      param_data.w_cond_mean_K(u) = param_data.Kxc(u).t() * param_data.Kxx_inv(u_par);
    }
    
    arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
    //end = std::chrono::steady_clock::now();
    //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    //start = std::chrono::steady_clock::now();
    if(sampling){
      arma::uvec ones = arma::ones<arma::uvec>(1);
      for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
        arma::uvec uix = ones * indexing(u)(ix);
        int first_ix = ix;
        int last_ix = ix;
        arma::mat Kcc = Covariancef(coords, qvblock_c, uix, uix, covpars, true);
        //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
        arma::mat Rchol;
        arma::mat Ktemp = Kcc - 
          param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * 
          param_data.Kxc(u).cols(first_ix, last_ix);
        try {
          Rchol = arma::chol(arma::symmatu(Ktemp), "lower");
        } catch(...){
          //Rcpp::Rcout << Ktemp << "\n";
          Ktemp(0,0) = 0;
          Rchol = arma::zeros(1,1);
        }
        //arma::vec rnvec = arma::randn(1);
        
        w.row(indexing(u)(ix)) = param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * w_par + Rchol * bigrnorm(indexing(u)(ix));
      }
    } else {
      w.rows(indexing(u)) = param_data.w_cond_mean_K(u) * w_par;
    }
    //arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
    
    //arma::mat Rcc = Kcc - cond_mean_K * Kxc;
    //arma::mat Rchol = arma::chol(Rcc, "lower");
    // sample
    //arma::vec rnvec = arma::randn(indexing(u).n_elem);
    //arma::vec w_temp = cond_mean_K * w_par + Rchol * rnvec;
    
    //w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
    
    
    //end = std::chrono::steady_clock::now();
    //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  //Rcpp::Rcout << "prediction timings: " << timings.t() << "\n";
  //Rcpp::Rcout << arma::accu(timings) << "\n";
  
  //Zw = w;//armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict_std] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. \n";
    
  }
  
}

void SpamTreeMV::deal_with_beta(){
  gibbs_sample_beta();
}

void SpamTreeMV::gibbs_sample_beta(){
  if(verbose & debug){
    Rcpp::Rcout << "gibbs_sample_beta \n";
  }
  
  start = std::chrono::steady_clock::now();
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X_available.rows(ix_by_q_a(j)).t() * 
      (y_available.rows(ix_by_q_a(j)) - w.rows(ix_by_q_a(j)));
    
    arma::vec Bmu = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j);
    Bcoeff.col(j) = Bmu + Sigma_chol_Bcoeff.t() * arma::randn(p);
    //Rcpp::Rcout << "j: " << j << "\n"
    //     << Bmu << "\n";
    
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * Bcoeff.col(j);
  }
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void SpamTreeMV::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  for(int j=0; j<q; j++){
    arma::vec Zw_availab = w.rows(na_ix_all);
    arma::vec XB_availab = XB.rows(na_ix_all);
    arma::mat yrr = y_available.rows(ix_by_q_a(j)) - XB_availab.rows(ix_by_q_a(j)) - Zw_availab.rows(ix_by_q_a(j));
    double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
    double aparam = 2.01 + ix_by_q_a(j).n_elem/2.0;
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
                  << "\n";
    }
  }
}


void SpamTreeMV::theta_update(SpamTreeMVData& data, const arma::vec& new_param){
  data.theta = new_param;
}

void SpamTreeMV::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void SpamTreeMV::beta_update(const arma::mat& new_beta){ 
  Bcoeff = new_beta;
}

void SpamTreeMV::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}
