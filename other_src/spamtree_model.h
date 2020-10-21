#include <RcppArmadillo.h>
#include <omp.h>
#include <stdexcept>

#include "R.h"
#include "find_nan.h"
#include "mh_adapt.h"
#include "field_v_concatm.h"
#include "covariance_functions.h"
#include "multires_utils.h"
#include "irls.h"

// with indexing
// without block extensions (obs with NA are left in)

using namespace std;

const double hl2pi = -.5 * log(2 * M_PI);

class SpamTree {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  int npars;
  int sigmasq_mh;
  
  // data
  arma::vec y;
  arma::mat X;
  arma::mat Z;
  
  arma::mat y_available;
  arma::mat X_available;
  arma::mat Z_available;
  
  arma::mat coords;
  
  // block membership
  arma::uvec blocking;
  arma::uvec res_is_ref;
  
  arma::field<arma::sp_mat> Zblock;
  arma::cube eta_rpx;
  arma::vec Zw;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> parents_indexing; 
  arma::field<arma::uvec> children_indexing;
  
  // NA data
  arma::field<arma::vec> na_1_blocks; // indicator vector by block
  arma::field<arma::uvec> na_ix_blocks;
  arma::uvec na_ix_all;
  int n_loc_ne_blocks;
  
  // regression
  arma::mat XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  // dependence
  //arma::field<arma::sp_mat> Ib;
  arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  arma::vec                 block_groups; // same group = sample in parallel given all others
  arma::uvec                block_ct_obs; // 0 if no available obs in this block, >0=count how many available
  arma::uvec                blocks_not_empty; // list i indices of non-empty blocks
  arma::uvec                blocks_predicting;
  arma::uvec                block_is_reference;
  
  int                       n_gibbs_groups;
  int                       n_actual_groups;
  arma::field<arma::vec>    u_by_block_groups;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  // for each block's parents, which columns of children of c is u? and which instead are of other parents
  //arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_child; 
  
  arma::field<arma::vec> dim_by_parent;
  //arma::field<arma::vec> dim_by_child;
  
  // params
  arma::mat bigrnorm;
  arma::mat w;
  arma::vec Bcoeff; // sampled
  double    tausq_inv;
  double    sigmasq;
  
  // params with mh step
  SpamTreeData param_data; 
  SpamTreeData alter_data;
  
  // setup
  bool predicting;
  
  bool verbose;
  bool debug;
  
  // debug var
  Rcpp::List debug_stuff; 
  
  void message(string s);
  
  // init / indexing
  void init_indexing();
  void init_finalize();
  void init_model_data(const arma::vec&);
  
  //arma::uvec converged;
  std::string family;
  
  arma::vec cparams;
  arma::mat Dmat;
  void theta_transform(const SpamTreeData&);
  
  void na_study();
  void make_gibbs_groups();
  
  // MCMC
  char use_alg;
  int max_num_threads;
  
  
  
  void get_loglik_w(SpamTreeData& data);
  void get_loglik_w_std(SpamTreeData& data);
  
  void get_loglik_comps_w(SpamTreeData& data);
  void get_loglik_comps_w_std(SpamTreeData& data);
  
  void deal_with_w();
  void deal_with_beta();
  
  void gibbs_sample_w_std();
  void gibbs_sample_w_rpx();
  
  void gibbs_sample_beta();
  void gibbs_sample_sigmasq();
  void gibbs_sample_tausq();
  
  void bfirls_beta();
  void bfirls_w();
  
  void predict(bool);
  void predict_std(bool, bool);
  
  // changing the values, no sampling
  void tausq_update(double);
  void theta_update(SpamTreeData&, const arma::vec&);
  void beta_update(const arma::vec&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  SpamTree();
  
  // build everything
  SpamTree(
    const std::string& family,
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& blocking_in,
    const arma::uvec& res_is_ref_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    
    const arma::field<arma::uvec>& indexing_in,
    
    const arma::mat& w_in,
    const arma::vec& beta_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    double sigmasq_in,
    
    char use_alg_in,
    int max_num_threads_in,
    bool v,
    bool debugging);
  
};

void SpamTree::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

SpamTree::SpamTree(){
  
}

SpamTree::SpamTree(
  const std::string& family_in,
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  const arma::uvec& blocking_in,
  const arma::uvec& res_is_ref_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  
  const arma::mat& w_in,
  const arma::vec& beta_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  double sigmasq_in,
  
  char use_alg_in='S',
  int max_num_threads_in=4,
  bool v=false,
  bool debugging=false){
  
  
  
  use_alg = use_alg_in; //S:standard, P:residual process, R: rp recursive
  max_num_threads = max_num_threads_in;
  
  message("~ SpamTree initialization.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  
  verbose = v;
  debug = debugging;
  
  message("SpamTree::SpamTree assign values.");
  
  family = family_in;
  
  y                   = y_in;
  X                   = X_in;
  Z                   = Z_in;
  
  coords              = coords_in;
  blocking            = blocking_in;
  res_is_ref          = res_is_ref_in;
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_actual_groups     = n_gibbs_groups;
  n_blocks            = block_names.n_elem;
  
  Rcpp::Rcout << n_gibbs_groups << " resolution layers. " << endl;
  //Rcpp::Rcout << block_groups_labels << endl;
  
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  Z_available = Z.rows(na_ix_all);
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  q  = Z.n_cols;
  dd = coords.n_cols;
  
  Zw      = arma::zeros(coords.n_rows);
  eta_rpx = arma::zeros(coords.n_rows, q, n_gibbs_groups);
  
  indexing    = indexing_in;
  
  sigmasq_mh = 1;
  if(dd == 2){
    if(q > 2){
      npars = sigmasq_mh+3;
    } else {
      if(q==1){
        sigmasq_mh = 1;
        npars = sigmasq_mh+1;
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
  
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  // init
  
  //Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  //u_is_which_child    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  dim_by_parent = arma::field<arma::vec> (n_blocks);
  //dim_by_child = arma::field<arma::vec> (n_blocks);
  
  tausq_inv        = tausq_inv_in;
  sigmasq          = sigmasq_in;
  Bcoeff           = beta_in;
  w                = w_in;
  
  predicting = true;
  
  // now elaborate
  message("SpamTree::SpamTree : init_indexing()");
  init_indexing();
  
  message("SpamTree::SpamTree : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("SpamTree::SpamTree : init_finalize()");
  init_finalize();
  
  
  // prior for beta
  XtX   = X_available.t() * X_available;
  Vi    = .01 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  message("SpamTree::SpamTree : make_gibbs_groups()");
  make_gibbs_groups();

  init_model_data(theta_in);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "SpamTree::SpamTree initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}

void SpamTree::make_gibbs_groups(){
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
  
  message("[make_gibbs_groups] list nonempty, predicting, and reference blocks\n");  
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
      for(int r=0; r<which_not_reference.n_elem; r++){
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
  
  message("[make_gibbs_groups] done.\n");
}

void SpamTree::na_study(){
  // prepare stuff for NA management
  message("[na_study] NA management for predictions\n");
  
  na_1_blocks = arma::field<arma::vec> (n_blocks);//(y_blocks.n_elem);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);//(y_blocks.n_elem);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros<arma::uvec>(n_blocks);//(y_blocks.n_elem);
  
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    arma::vec yvec = y.rows(indexing(i));//y_blocks(i);
    na_1_blocks(i) = arma::zeros(yvec.n_elem);
    na_1_blocks(i).elem(arma::find_finite(yvec)).fill(1);
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
    
  }
  
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    block_ct_obs(i) = arma::accu(na_1_blocks(i)); //***
    if(block_ct_obs(i) > 0){
      n_loc_ne_blocks += indexing(i).n_elem;//coords_blocks(i).n_rows;
    }
  }
  
  message("[na_study] done.\n");
}

void SpamTree::init_indexing(){
  
  Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);

  message("[init_indexing] indexing, parent_indexing, children_indexing");
  
  //pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    //Rcpp::Rcout << u << endl;
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
    //Rcpp::Rcout << "doing Z" << endl;
    Zblock(u) = Zify( Z.rows(indexing(u)) );
  }
  
}

void SpamTree::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
  //pragma omp parallel for //**
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
      //Rcpp::Rcout << "Ib " << parents(u).n_elem << endl;
      //Ib(u) = arma::eye<arma::sp_mat>(coords_blocks(u).n_rows, coords_blocks(u).n_rows);
      //Ib(u) = arma::eye<arma::sp_mat>(indexing(u).n_elem, indexing(u).n_elem);
      //for(int j=0; j<Ib(u).n_cols; j++){
      //  if(na_1_blocks(u)(j) == 0){
      //    Ib(u)(j,j) = 0;//1e-6;
      //  }
      //}
      //Rcpp::Rcout << "dim_by_parent " << parents(u).n_elem << endl;
      // number of coords of the jth parent of the child
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
      u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (q*children(u).n_elem);
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
        arma::uvec temp = arma::regspace<arma::uvec>(0, q*dimen-1);
        arma::umat result = arma::trans(arma::umat(temp.memptr(), q, temp.n_elem/q));
        arma::uvec rowsel = arma::zeros<arma::uvec>(result.n_rows);
        rowsel.subvec(firstcol, lastcol-1).fill(1);
        arma::umat result_local = result.rows(arma::find(rowsel==1));
        arma::uvec result_local_flat = arma::vectorise(arma::trans(result_local));
        arma::umat result_other = result.rows(arma::find(rowsel==0));
        arma::uvec result_other_flat = arma::vectorise(arma::trans(result_other));
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = result_local_flat; // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = result_other_flat; // u parent of c is NOT in these columns for c
        
      }
      
    //}
  }
}

void SpamTree::init_model_data(const arma::vec& theta_in){
  
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
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  if(sigmasq_mh == 0){
    param_data.theta          = theta_in;
  } else {
    param_data.theta          = arma::join_vert(arma::ones(1) * sigmasq, theta_in);
  }
  
  //param_data.cholfail       = false;
  //param_data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  
  
  param_data.Sigi_children = arma::field<arma::cube> (n_blocks);
  param_data.Smu_children = arma::field<arma::mat> (n_blocks);

  //converged = arma::zeros<arma::uvec>(1+ n_blocks); // includes beta
  
  for(int i=0; i<n_blocks; i++){
    param_data.Sigi_children(i) = arma::zeros(q*indexing(i).n_elem,
                  q*indexing(i).n_elem, children(i).n_elem);
    param_data.Smu_children(i) = arma::zeros(q*indexing(i).n_elem,
                 children(i).n_elem);
    
    int u = block_names(i)-1;
    if(block_ct_obs(u)>0){
      param_data.Kxx_invchol(u) = arma::zeros(q*parents_indexing(u).n_elem + q*indexing(u).n_elem, //Kxx_invchol(last_par).n_rows + Kcc.n_rows, 
                             q*parents_indexing(u).n_elem + q*indexing(u).n_elem);//Kxx_invchol(last_par).n_cols + Kcc.n_cols);
      //Kxx_chol(u) = Kxx_invchol(u);
    }
    
    param_data.w_cond_mean_K(u) = arma::zeros(q*indexing(u).n_elem, q*parents_indexing(u).n_elem);
    param_data.Kxc(u) = arma::zeros(q*parents_indexing(u).n_elem, q*indexing(u).n_elem);
    param_data.ccholprecdiag(u) = arma::zeros(q*indexing(u).n_elem);
    
    if(block_is_reference(u) == 1){
      param_data.w_cond_prec(u) = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem);
      param_data.Rcc_invchol(u) = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem);
    } else {
      if(block_ct_obs(u) > 0){
        param_data.w_cond_prec_noref(u) = arma::field<arma::mat>(indexing(u).n_elem);
        for(int j=0; j<indexing(u).n_elem; j++){
          param_data.w_cond_prec_noref(u)(j) = arma::zeros(q, q);
        }
        
      }
    }
  }
  
  alter_data                = param_data; 
  
  Rcpp::Rcout << "init_model_data: indexing elements: " << indexing.n_elem << endl;
  
}

void SpamTree::get_loglik_w(SpamTreeData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  get_loglik_w_std(data);
}

void SpamTree::get_loglik_w_std(SpamTreeData& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
  
  //arma::uvec blocks_not_empty = arma::find(block_ct_obs > 0);
 //*** omp parallel for //**
  for(int i=0; i<blocks_not_empty.n_elem; i++){  
    int u = blocks_not_empty(i);
  
    arma::mat w_x = arma::vectorise( arma::trans( w.rows(indexing(u)) ) );
    
    if(parents(u).n_elem > 0){
      arma::vec w_pars = arma::vectorise( arma::trans( w.rows(parents_indexing(u))));//indexing(p)) ));
      w_x -= data.w_cond_mean_K(u) * w_pars;
    }
    
    if(block_is_reference(u) == 1){
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
    } else {
      // really only need the (q=1: diagonals, q>1: block-diagonals) of precision since we assume
      // conditional independence here.
      data.wcore(u) = 0;
      arma::uvec ones = arma::ones<arma::uvec>(1);
      for(int ix=0; ix<indexing(u).n_elem; ix++){
        arma::uvec uix = ones * indexing(u)(ix);
        //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
        int first_ix = ix*q;
        int last_ix = ix*q+q-1;
        data.wcore(u) += arma::conv_to<double>::from(
          arma::trans(w_x.rows(first_ix, last_ix)) * 
            //data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) * 
            data.w_cond_prec_noref(u)(ix) *
            w_x.rows(first_ix, last_ix));
      }
      /*
      double check1 = data.wcore(u);
      double check2 = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      Rcpp::Rcout << "u: " << u << " is reference? " << block_is_reference(u) << endl;
      if(check1 != check2){
        Rcpp::Rcout << data.w_cond_prec(u) << endl;
        Rcpp::Rcout << check1 << " " << check2 << endl;
        throw 1;
      }*/
      
    }
    data.loglik_w_comps(u) = (q*indexing(u).n_elem + .0) * hl2pi -.5 * data.wcore(u);
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

void SpamTree::theta_transform(const SpamTreeData& data){
  //arma::vec Kparam = data.theta; 
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2

  if(sigmasq_mh == 0){
    cparams = arma::join_vert(arma::ones(1) * sigmasq, 
                              data.theta.subvec(0, npars - 1));
  } else {
    cparams = data.theta.subvec(0, npars - 1);
  }
  
  
  if(k>0){
    Dmat = vec_to_symmat(data.theta.subvec(npars, npars + k - 1));
  } else {
    Dmat = arma::zeros(1,1);
  }
}

void SpamTree::get_loglik_comps_w(SpamTreeData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  get_loglik_comps_w_std(data);
}

void SpamTree::get_loglik_comps_w_std(SpamTreeData& data){
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_w_std] start. ");
  
  // arma::vec timings = arma::zeros(7);
  theta_transform(data);
  // cycle through the resolutions starting from the bottom

  for(int g=0; g<n_actual_groups; g++){
//***#pragma omp parallel for 
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(block_ct_obs(u) > 0){
      //Rcpp::Rcout << "u: " << u << " obs: " << block_ct_obs(u) << " parents: " << parents(u).t() << "\n"; 
      //arma::mat cond_mean;
      arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
      //Rcpp::Rcout << "step 1\n";
    
      if(parents(u).n_elem == 0){
        //start = std::chrono::steady_clock::now();
        arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
        data.Kxx_invchol(u) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
        data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
        data.Rcc_invchol(u) = data.Kxx_invchol(u); 
        data.w_cond_mean_K(u) = arma::zeros(0, 0);
        data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
        data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
        data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      } else {
        //Rcpp::Rcout << "step 2\n";
        int last_par = parents(u)(parents(u).n_elem - 1);
        //arma::mat LAi = Kxx_invchol(last_par);
        
        xCovHUV_inplace(data.Kxc(u), coords, parents_indexing(u), indexing(u), cparams, Dmat, false);
        arma::vec w_pars = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));//indexing(p))));
        data.w_cond_mean_K(u) = data.Kxc(u).t() * data.Kxx_inv(last_par);
        w_x -= data.w_cond_mean_K(u) * w_pars;
        
        if(res_is_ref(g) == 1){
          //start = std::chrono::steady_clock::now();
          arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
          data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc - data.w_cond_mean_K(u) * data.Kxc(u)), "lower")));
          //end = std::chrono::steady_clock::now();
          //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          if(children(u).n_elem > 0){
            invchol_block_inplace_direct(data.Kxx_invchol(u), data.Kxx_invchol(last_par), data.w_cond_mean_K(u), data.Rcc_invchol(u));
            data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
            data.has_updated(u) = 1;
          }
          
          data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
          //end = std::chrono::steady_clock::now();
          //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } else {
          // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
          //start = std::chrono::steady_clock::now();
          data.wcore(u) = 0;
          arma::uvec ones = arma::ones<arma::uvec>(1);
          for(int ix=0; ix<indexing(u).n_elem; ix++){
            arma::uvec uix = ones * indexing(u)(ix);
            //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
            int first_ix = ix*q;
            int last_ix = ix*q+q-1;
            //start = std::chrono::steady_clock::now();
            arma::mat Kcc = xCovHUV(coords, uix, uix, cparams, Dmat, true);
            //Rcpp::Rcout << "step 2o " << first_ix << " " << last_ix << "\n";
            //end = std::chrono::steady_clock::now();
            //timings(5) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            //start = std::chrono::steady_clock::now();
            arma::mat Kcx_xxi_xc = data.w_cond_mean_K(u).rows(first_ix, last_ix) * data.Kxc(u).cols(first_ix, last_ix);
            //Rcpp::Rcout << "step 3o " << arma::size(cond_mean_K_sub) << " " << arma::size(data.Kxc(u).cols(first_ix, last_ix)) <<  "\n";
            arma::mat Rinvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc - Kcx_xxi_xc), "lower")));
            //data.Rcc_invchol(u).submat(first_ix, first_ix, last_ix, last_ix) = Rinvchol;
            data.ccholprecdiag(u).subvec(first_ix, last_ix) = Rinvchol.diag();
            
            //data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) = Rinvchol.t() * Rinvchol;
            data.w_cond_prec_noref(u)(ix) = Rinvchol.t() * Rinvchol;
            
            data.wcore(u) += arma::conv_to<double>::from(
                w_x.rows(first_ix, last_ix).t() * 
                data.w_cond_prec_noref(u)(ix) *
                w_x.rows(first_ix, last_ix));
            
            //end = std::chrono::steady_clock::now();
            //timings(6) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          }
          
        }
      }
      
      data.logdetCi_comps(u) = arma::accu(log(data.ccholprecdiag(u)));
      data.loglik_w_comps(u) = (q*indexing(u).n_elem+.0) 
        * hl2pi -.5 * data.wcore(u);
    }
  }
  
  //Rcpp::Rcout << "timings: " << timings.t() << endl;
  //Rcpp::Rcout << "total timings: " << arma::accu(timings) << endl;
  data.logdetCi = arma::accu(data.logdetCi_comps.subvec(0, n_blocks-1));
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps.subvec(0, n_blocks-1));
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_comps_w_std] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}

void SpamTree::deal_with_w(){
  // Gibbs samplers:
  // S=standard gibbs (cheapest), 
  // P=residual process, 
  // R=residual process using recursive functions
  // Back-fitting plus IRLS:
  // I=back-fitting and iterated reweighted least squares
  switch(use_alg){
  case 'S':
    gibbs_sample_w_std();
    break;
  case 'P':
    gibbs_sample_w_rpx();
    break;
  //case 'I': 
  //  bfirls_w();
  //  break;
  }
}

/*
void SpamTree::gibbs_sample_w_std(){
  // keep seed
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] sampling big stdn matrix size " << n << "," << q << endl;
  }
  
  Rcpp::RNGScope scope;
  arma::mat rand_norm_mat = arma::randn(coords.n_rows, q);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  
  arma::vec timings = arma::zeros(3);
  
  for(int g=0; g<n_actual_groups; g++){
    if(res_is_ref(g) == 1){
      
      
//////***#pragma omp parallel for 
      for(int i=0; i<u_by_block_groups(g).n_elem; i++){
        int u = u_by_block_groups(g)(i);
    
        start = std::chrono::steady_clock::now();
        //Rcpp::Rcout << "step 2\n";
        arma::mat Smu_tot = arma::zeros(q*indexing(u).n_elem, 1);
        arma::mat Sigi_tot = param_data.w_cond_prec(u); // Sigi_p
        
        //Rcpp::Rcout << "step 3\n";
        arma::vec w_par;
        if(parents(u).n_elem>0){
          w_par = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));//indexing(p) ) ));
          Smu_tot += Sigi_tot * param_data.w_cond_mean_K(u) * w_par;//param_data.w_cond_mean(u);
        }
        
        //arma::uvec ug = arma::zeros<arma::uvec>(1) + g;
        // indexes being used as parents in this group
        end = std::chrono::steady_clock::now();
        timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        //arma::cube Sigi_children = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem, children(u).n_elem);
        //arma::cube Smu_children = arma::zeros(q*indexing(u).n_elem, 1, children(u).n_elem);
        start = std::chrono::steady_clock::now();
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //if(block_ct_obs(child) > 0){
          //if(block_groups(child) == block_groups(u)+1){// one res up
          //Rcpp::Rcout << "child: " << child << endl;
          //clog << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
          //Rcpp::Rcout << u_is_which_col_f(u)(c)(0).t() << "\n";
          //Rcpp::Rcout << u_is_which_col_f(u)(c)(1).t() << "\n";
          
          arma::mat w_cond_prec_child;
          if(block_is_reference(child)){
            w_cond_prec_child = param_data.w_cond_prec(child);
          } else {
            w_cond_prec_child = arma::zeros(q*indexing(child).n_elem, q*indexing(child).n_elem);
            for(int ix=0; ix<indexing(child).n_elem; ix++){
              int first_ix = ix*q;
              int last_ix = ix*q+q-1;
              w_cond_prec_child.submat(first_ix, first_ix, last_ix, last_ix) = 
                param_data.w_cond_prec_noref(child)(ix);
            }
          }
          
          //Rcpp::Rcout << "child step 1\n"; 
          arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
          //Rcpp::Rcout << "child step 2\n"; 
          arma::mat AK_uP = AK_u.t() * w_cond_prec_child;// param_data.w_cond_prec(child);
          arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
          //Rcpp::Rcout << "child step 3\n"; 
          arma::vec w_child = arma::vectorise(arma::trans( w.rows( indexing(child) ) ));
          
          //Rcpp::Rcout << "child step 4 \n";
          arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
          arma::vec w_par_child = arma::vectorise(arma::trans(w_parents_of_child));
          arma::vec w_par_child_select = w_par_child.rows(u_is_which_col_f(u)(c)(1));
          
          try {
            double checkdiff = abs(arma::accu( w_par_child.rows(u_is_which_col_f(u)(c)(0) ) - 
                                   arma::vectorise(arma::trans(w.rows(indexing(u))))));
            //Rcpp::Rcout << checkdiff << endl;
            if(checkdiff > 0){
              Rcpp::Rcout << w.rows(indexing(u)) << endl
                          << w_parents_of_child << endl;
              Rcpp::Rcout << u_is_which_col_f(u)(c) << endl;
            }
          } catch (...) {
            Rcpp::Rcout << u_is_which_col_f(u)(c) << endl;
            throw 1;
          }
          
          
          //Rcpp::Rcout << arma::size(Sigi_tot) << " " << arma::size(AK_u) << " " << arma::size(AK_uP) << "\n";
          //Rcpp::Rcout << "child step 5 \n";
          Sigi_tot += AK_uP * AK_u;
          //Rcpp::Rcout << "child step 6 \n";
          Smu_tot += AK_uP * (w_child - AK_others * w_par_child_select);
          //Sigi_children.slice(c) = AK_uP * AK_u;
          //Smu_children.slice(c) = AK_uP * (w_child - AK_others * w_par_child_select );
          //}
          //}
        }
        end = std::chrono::steady_clock::now();
        timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //Sigi_tot += arma::sum(Sigi_children, 2);
        //Smu_tot += arma::sum(Smu_children, 2);
        
        start = std::chrono::steady_clock::now();
        
        //Rcpp::Rcout << "step 4\n";
        Sigi_tot += tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u);
        Smu_tot += Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
          ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
        
        
        arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        
        //Rcpp::Rcout << "5 " << endl;
        // sample
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
        arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        
        //rand_norm_tracker(indexing(u), ug) += 1;
        //Rcpp::Rcout << w_temp.n_elem/q << " " << q << "\n";
        w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
        //Rcpp::Rcout << "step 6\n";
        
    end = std::chrono::steady_clock::now();
    timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      }
      
      
    } else {
      
      
      //Rcpp::Rcout << "step 1a" << endl;
//***#pragma omp parallel for
      for(int i=0; i<u_by_block_groups(g).n_elem; i++){
        int u = u_by_block_groups(g)(i);
        // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
        arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        arma::uvec ones = arma::ones<arma::uvec>(1);
        arma::mat tsq_Zt_y_XB = tausq_inv * Zblock(u).t() * (y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff);
        
          //Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
          //( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
        
        //arma::mat Smu_tot = param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * w_par +
        //  tsq_Zt_y_XB;
        
        //arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        //arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        //arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        //w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
        
        arma::mat cond_mean_K_wpar = param_data.w_cond_mean_K(u) * w_par;
        for(int ix=0; ix<indexing(u).n_elem; ix++){
          //Rcpp::Rcout << "step 2a\n";
          arma::uvec uix = ones * indexing(u)(ix);
          arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
          arma::mat ZZ = Z.row(indexing(u)(ix));
          
          arma::mat Sigi_tot = param_data.w_cond_prec_noref(u)(ix) + tausq_inv * ZZ.t() * ZZ; 
          arma::mat Smu_tot = param_data.w_cond_prec_noref(u)(ix) * cond_mean_K_wpar.rows(ix_q) +
            tsq_Zt_y_XB.rows(ix_q);
            //tausq_inv * ZZ.t() * ( y.row(indexing(u)(ix)) - X.row(indexing(u)(ix)) * Bcoeff );
          //Rcpp::Rcout << "step 3a" << endl;
          arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Sigi_tot), "lower")));
  
          arma::vec w_temp = Sigi_chol.t() * Sigi_chol * Smu_tot + Sigi_chol.t() * rnvec.rows(ix_q);
          w.row(indexing(u)(ix)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
        }
        //end = std::chrono::steady_clock::now();
        //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      }
      
    }
  }
  
  //Rcpp::Rcout << "timings: " << timings.t() << endl;
   
  //debug_stuff = Rcpp::List::create(
  //  Rcpp::Named("rand_norm_tracker") = rand_norm_tracker,
  //  Rcpp::Named("rand_child_tracker") = rand_child_tracker,
  //  Rcpp::Named("rand_norm_mat") = rand_norm_mat
  //);
  
  Zw = armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}
 
*/
void SpamTree::gibbs_sample_w_std(){
  // backward sampling?
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] sampling big stdn matrix size " << n << "," << q << endl;
  }
  
  Rcpp::RNGScope scope;
  bigrnorm = arma::randn(coords.n_rows, q);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  
  arma::vec timings = arma::zeros(8);
  
  for(int g=n_actual_groups-1; g>=0; g--){
  
  //***#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      arma::vec w_par;
      arma::mat AK_uP_all;
      
      if(res_is_ref(g) == 1){
        //start = std::chrono::steady_clock::now();
        
        // reference set. full dependence
        arma::mat Smu_tot = arma::zeros(q*indexing(u).n_elem, 1);
        arma::mat Sigi_tot = param_data.w_cond_prec(u); // Sigi_p
        
        if(parents(u).n_elem>0){
          w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
          Smu_tot += Sigi_tot * param_data.w_cond_mean_K(u) * w_par;//param_data.w_cond_mean(u);
          // for updating the parents that have this block as child
          AK_uP_all = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
        }
        
        if(children(u).n_elem > 0){
          Sigi_tot += arma::sum(param_data.Sigi_children(u), 2);
          Smu_tot += arma::sum(param_data.Smu_children(u), 1);
        }
        
        Sigi_tot += tausq_inv * Zblock(u).t() * Zblock(u);
        Smu_tot += tausq_inv * Zblock(u).t() * 
          ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff );
        
        arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        
        arma::vec rnvec = arma::vectorise(bigrnorm.rows(indexing(u)));
        arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        
        w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
        
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
      } else {
        // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
        //start = std::chrono::steady_clock::now();
        w_par = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));//indexing(p) ) ));
        arma::vec rnvec = arma::vectorise(bigrnorm.rows(indexing(u)));
        arma::uvec ones = arma::ones<arma::uvec>(1);
        arma::mat tsq_Zt_y_XB = tausq_inv * Zblock(u).t() * (y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff);
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        arma::mat cond_mean_K_wpar = param_data.w_cond_mean_K(u) * w_par;
        //end = std::chrono::steady_clock::now();
        //timings(2) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        AK_uP_all = arma::zeros(q*parents_indexing(u).n_elem, q*indexing(u).n_elem);
        //AK_uP_all = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
        
        for(int ix=0; ix<indexing(u).n_elem; ix++){
          //start = std::chrono::steady_clock::now();
          //arma::uvec uix = ones * indexing(u)(ix);
          int first_ix = ix*q;
          int last_ix = ix*q+q-1;
          //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
          arma::mat ZZ = Z.row(indexing(u)(ix));
          
          arma::mat Sigi_tot = param_data.w_cond_prec_noref(u)(ix) +//param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) + 
            tausq_inv * ZZ.t() * ZZ; 
          arma::mat Smu_tot  = param_data.w_cond_prec_noref(u)(ix) * //param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) * 
            cond_mean_K_wpar.rows(first_ix, last_ix) +
            tsq_Zt_y_XB.rows(first_ix, last_ix);
          //end = std::chrono::steady_clock::now();
          //timings(3) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
          //start = std::chrono::steady_clock::now();
          arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Sigi_tot), "lower")));
          
          arma::vec w_temp = Sigi_chol.t() * Sigi_chol * Smu_tot + Sigi_chol.t() * rnvec.rows(first_ix, last_ix);
          w.row(indexing(u)(ix)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
          //end = std::chrono::steady_clock::now();
          //timings(4) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          
          
          //start = std::chrono::steady_clock::now();
          arma::mat AK_uP_here = arma::trans(param_data.w_cond_mean_K(u).rows(first_ix, last_ix)) *
            param_data.w_cond_prec_noref(u)(ix);
          AK_uP_all.cols(first_ix, last_ix) = AK_uP_here; //param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) *
          
          //9.9621e+06
          //end = std::chrono::steady_clock::now();
          //timings(5) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
      }
      
      // update Sigi_children and Smu_children for parents so we can sample them correctly
      if(parents(u).n_elem > 0){
        arma::mat AK_uP_u_all = AK_uP_all * param_data.w_cond_mean_K(u);
        
        w_par = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));//indexing(p) ) ));
        //start = std::chrono::steady_clock::now();
        
        //end = std::chrono::steady_clock::now();
        //timings(6) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        // update parents' Sigi_children and Smu_children
        for(int p=0; p<parents(u).n_elem; p++){
          //Rcpp::Rcout << "u: " << u << " " << parents_indexing(u).n_elem << endl;
          // up is the parent
          int up = parents(u)(p);
          //Rcpp::Rcout << "parent: " << up << endl;
          // up has other children, which one is u?
          arma::uvec cuv = arma::find(children(up) == u);
          int c_ix = cuv(0); 
          //Rcpp::Rcout << "which child is this: " << c_ix << endl;
          
          arma::mat AK_uP = AK_uP_all.rows(u_is_which_col_f(up)(c_ix)(0));
          arma::vec w_par_child_select = w_par.rows(u_is_which_col_f(up)(c_ix)(1));
          
          //Rcpp::Rcout << "filling" << endl;
          param_data.Sigi_children(up).slice(c_ix) = 
            AK_uP_u_all.submat(u_is_which_col_f(up)(c_ix)(0), 
                               u_is_which_col_f(up)(c_ix)(0));
          
          arma::vec w_temp = arma::vectorise(arma::trans(w.rows(indexing(u))));
          param_data.Smu_children(up).col(c_ix) = AK_uP * w_temp -  
            AK_uP_u_all.submat(u_is_which_col_f(up)(c_ix)(0), 
                               u_is_which_col_f(up)(c_ix)(1)) * w_par_child_select;
          
        }
        //end = std::chrono::steady_clock::now();
        //timings(7) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      
    }
  }
  
  Zw = armarowsum(Z % w);
  
  if(verbose){
    Rcpp::Rcout << "timings: " << timings.t() << endl;
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}
 
void SpamTree::gibbs_sample_w_rpx(){
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_rpx] " << endl;
  }
  
  // covariance parameters
  theta_transform(param_data);
  
  Rcpp::RNGScope scope;
  bigrnorm = arma::randn(coords.n_rows, q);
  
  //arma::vec timings = arma::zeros(5);
  //Rcpp::Rcout << "starting loops \n";
  //Rcpp::Rcout << "groups: " << block_groups_labels.t() << endl;
  
  //arma::mat eta_rpx_mat = eta_rpx.col(0);
  
  for(int g=0; g<n_actual_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    
  //***#pragma omp parallel for 
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      // eta_rpx is cube (location x q x grp)
      
      //start = std::chrono::steady_clock::now();
      if(res_is_ref(g) == 1){
        // precision
        arma::mat Sigi_tot = tausq_inv * Zblock(u).t() * Zblock(u) + param_data.w_cond_prec(u);
        // Smean
        arma::mat eta_others_q = subcube_collapse_via_sum(eta_rpx, indexing(u), other_etas);
        arma::vec Zw_others = armarowsum(Z.rows(indexing(u)) % eta_others_q);
        arma::mat Smu_tot = tausq_inv * Zblock(u).t() *
          (y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff - Zw_others);
        
        arma::field<arma::mat> KchildKxx(children(u).n_elem);
        for(int c=0; c<children(u).n_elem; c++){
          int cx = children(u)(c);
          arma::mat Kchildc = xCovHUV(coords, indexing(cx), indexing(u), cparams, Dmat, false);
          
          arma::mat Kchildren;
          if(parents(u).n_elem > 0){
            // update children locations
            int lp = parents(u)(parents(u).n_elem - 1);
            
            //start = std::chrono::steady_clock::now();
            arma::mat Kchildx = xCovHUV(coords, indexing(cx), parents_indexing(u), cparams, Dmat, false);
            //end = std::chrono::steady_clock::now();
            //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            //start = std::chrono::steady_clock::now();
            Kchildren = Kchildc - Kchildx * param_data.Kxx_inv(lp) * param_data.Kxc(u);
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          } else {
            Kchildren = Kchildc;
          }
          KchildKxx(c) = Kchildren * param_data.w_cond_prec(u);
          
          // precision
          Sigi_tot += tausq_inv * KchildKxx(c).t() * Zblock(cx).t() * Zblock(cx) * KchildKxx(c);
          // Smean
          arma::mat eta_others_q = subcube_collapse_via_sum(eta_rpx, indexing(cx), other_etas);
          arma::vec Zw_others = armarowsum(Z.rows(indexing(cx)) % eta_others_q);
          
          Smu_tot += tausq_inv * KchildKxx(c).t() * Zblock(cx).t() *
            (y.rows(indexing(cx)) - X.rows(indexing(cx)) * Bcoeff - Zw_others);
        }
        
        arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        
        
        // sample independent residual
        //Rcpp::Rcout << "sampling size " << indexing(u).n_elem << endl;
        arma::vec rnvec = arma::vectorise(bigrnorm.rows(indexing(u)));
        
        arma::vec w_local = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); 
        arma::mat w_fillmat = arma::trans(arma::mat(w_local.memptr(), q, w_local.n_elem/q));
        
        // at this resolution we have a q matrix of new w, fill the cube.
        cube_fill(eta_rpx, indexing(u), grp, w_fillmat);
        //eta_rpx_mat.submat(indexing(u), grpuvec) = w_local;
        
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        for(int c=0; c<children(u).n_elem; c++){
          int cx = children(u)(c);
          arma::vec w_cx = KchildKxx(c) * w_local;
          
          //Rcpp::Rcout << arma::size(w_cx) << endl;
          arma::mat w_cx_fillmat = arma::trans(arma::mat(w_cx.memptr(), q, w_cx.n_elem/q));
          cube_fill(eta_rpx, indexing(cx), grp, w_cx_fillmat);
          //eta_rpx_mat.submat(indexing(cx), grpuvec) = w_cx;
        }
        
      } else {
        arma::mat eta_others_q = subcube_collapse_via_sum(eta_rpx, indexing(u), other_etas);
        arma::vec Zw_others = armarowsum(Z.rows(indexing(u)) % eta_others_q);
        arma::mat w_fillmat = arma::zeros(indexing(u).n_elem, q);
        
        for(int ix=0; ix<indexing(u).n_elem; ix++){
          //int first_ix = ix*q;
          //int last_ix = ix*q+q-1;
          arma::mat ZZ = Z.row(indexing(u)(ix));
          arma::mat Sigi_tot = param_data.w_cond_prec_noref(u)(ix) +//param_data.w_cond_prec(u).submat(first_ix, first_ix, last_ix, last_ix) + 
            tausq_inv * ZZ.t() * ZZ; 
          arma::mat Smu_tot = tausq_inv * ZZ.t() *
            (y.row(indexing(u)(ix)) - X.row(indexing(u)(ix)) * Bcoeff - Zw_others(ix));
          arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
          arma::vec rnvec = arma::vectorise(bigrnorm.row(indexing(u)(ix)));
          arma::vec w_local = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); 
          w_fillmat.row(ix) = arma::trans(arma::mat(w_local.memptr(), q, w_local.n_elem/q));
        }
        
        // at this resolution we have a q matrix of new w, fill the cube.
        cube_fill(eta_rpx, indexing(u), grp, w_fillmat);
        //eta_rpx_mat.submat(indexing(u), grpuvec) = w_local;
        
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      }
      
    }
  }
  
  //start = std::chrono::steady_clock::now();
  //w = armarowsum(eta_rpx_mat);
  //eta_rpx.col(0) = eta_rpx_mat;
  w = arma::sum(eta_rpx, 2); // sum all resolutions
  Zw = armarowsum(Z % w);
  
  if(verbose){
    //end = std::chrono::steady_clock::now();
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_rpx] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us and " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us \n";
    //Rcpp::Rcout << timings.t() << endl;
  }
}

void SpamTree::bfirls_w(){
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[bfirls_w] " << endl;
  }
  
  // covariance parameters
  theta_transform(param_data);
  
  for(int g=0; g<n_actual_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    
    //***#pragma omp parallel for 
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      // ######################################
      
      //start = std::chrono::steady_clock::now();
      if(res_is_ref(g) == 1){
        
        arma::uvec here_and_children = arma::join_vert(indexing(u), children_indexing(u));
        
        arma::mat eta_others_fctv = subcube_collapse_via_sum(eta_rpx, here_and_children, other_etas);
        arma::vec Zw_others = armarowsum(Z.rows(here_and_children) % eta_others_fctv);
        
        arma::vec y_fctv = y.rows(here_and_children);
        arma::vec Xb_fctv = X.rows(here_and_children) * Bcoeff;
        
        arma::field<arma::mat> ZK(1 + children(u).n_elem);
        ZK(0) = Zblock(u);
        
        arma::field<arma::mat> KchildKxx(children(u).n_elem);
        for(int c=0; c<children(u).n_elem; c++){
          int cx = children(u)(c);
          arma::mat Kchildc = xCovHUV(coords, indexing(cx), indexing(u), cparams, Dmat, false);
          arma::mat Kchildren;
          if(parents(u).n_elem > 0){
            // update children locations
            int lp = parents(u)(parents(u).n_elem - 1);
            //start = std::chrono::steady_clock::now();
            arma::mat Kchildx = xCovHUV(coords, indexing(cx), parents_indexing(u), cparams, Dmat, false);
            //end = std::chrono::steady_clock::now();
            //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            //start = std::chrono::steady_clock::now();
            Kchildren = Kchildc - Kchildx * param_data.Kxx_inv(lp) * param_data.Kxc(u);
            //end = std::chrono::steady_clock::now();
            //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          } else {
            Kchildren = Kchildc;
          }
          KchildKxx(c) = Kchildren * param_data.w_cond_prec(u);
          ZK(1+c) = Zblock(cx) * KchildKxx(c);
        }
        
        arma::mat ZKmat = field_v_concatm(ZK);
        arma::vec offset = Xb_fctv + Zw_others;
        //arma::mat eta_others_q = subcube_collapse_via_sum(eta_rpx, indexing(u), other_etas);
        arma::mat eta_here_q = subcube_collapse_via_sum(eta_rpx, indexing(u), grpuvec);
        
        double scales = family == "gaussian" ? 1.0/tausq_inv : 1.0;
        arma::vec eta_local = arma::vectorise(arma::trans(eta_here_q));
        arma::vec eta_update = irls_step(eta_local,
                                         y_fctv, ZKmat, offset,
                                         param_data.w_cond_prec(u),
                                         family, scales);
        
        //double convnorm = arma::norm(eta_here_q-eta_update);
        //Rcpp::Rcout << convnorm << endl;
        //converged(1+u) = convnorm < 1e-04;
        
        arma::field<arma::mat> w_upd_list(1+children(u).n_elem);
        
        w_upd_list(0) = arma::trans(arma::mat(eta_update.memptr(), q, eta_update.n_elem/q));
        for(int c=0; c<children(u).n_elem; c++){
          arma::vec child_update = KchildKxx(c) * eta_update;
          w_upd_list(1+c) = arma::trans(arma::mat(child_update.memptr(), q, child_update.n_elem/q));
        }
        
        arma::mat w_upd_mat = field_v_concatm(w_upd_list);
        cube_fill(eta_rpx, here_and_children, grp, w_upd_mat);
        
      } else {
        //start = std::chrono::steady_clock::now();
        
        arma::mat eta_here_q = subcube_collapse_via_sum(eta_rpx, indexing(u), grpuvec);
        arma::mat eta_others = subcube_collapse_via_sum(eta_rpx, indexing(u), other_etas);
        arma::vec Zw_others = armarowsum(Z.rows(indexing(u)) % eta_others);
        arma::mat w_fillmat = arma::zeros(indexing(u).n_elem, q);
        
        //double conv;
        for(int ix=0; ix<indexing(u).n_elem; ix++){
          arma::mat ZZ = Z.row(indexing(u)(ix));
          arma::vec offset = X.row(indexing(u)(ix)) * Bcoeff + Zw_others(ix);
          //arma::mat eta_others_q = subcube_collapse_via_sum(eta_rpx, indexing(u), other_etas);
          
          
          double scales = family == "gaussian" ? 1.0/tausq_inv : 1.0;
          
          arma::vec eta_local = arma::trans(eta_here_q.row(ix));
          arma::vec eta_update = irls_step(eta_local,
                                           y.row(indexing(u)(ix)), Z.row(indexing(u)(ix)), offset,
                                           param_data.w_cond_prec_noref(u)(ix),
                                           family, scales);
          
          w_fillmat.row(ix) = arma::trans(arma::mat(eta_update.memptr(), q, eta_update.n_elem/q));
          //conv += arma::accu(pow(eta_update - eta_local, 2));
        }
        
        //converged(1+u) = conv < 1e-04;
        
        cube_fill(eta_rpx, indexing(u), grp, w_fillmat);
        
      }
      
    }
    
    
    /*for(int j=0; j<q; j++){
      int nr = eta_rpx.n_rows;
      arma::vec rjvec = eta_rpx.subcube(0, j, g, nr-1, j, g);
      arma::uvec seekinto = arma::find(rjvec != 0);
      double rjvec_mean = arma::mean(rjvec(seekinto));
      eta_rpx.subcube(0, j, g, nr-1, j, g) = rjvec-rjvec_mean;
    }*/
    
  }
  
  //start = std::chrono::steady_clock::now();
  w = arma::sum(eta_rpx, 2); // sum all resolutions
  Zw = armarowsum(Z % w);
  
  if(verbose){
    //end = std::chrono::steady_clock::now();
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[bfirls_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us and " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us \n";
    //Rcpp::Rcout << timings.t() << endl;
  }
}

void SpamTree::predict(bool theta_update=true){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  predict_std(true, theta_update);
}

void SpamTree::predict_std(bool sampling=true, bool theta_update=true){
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
       xCovHUV_inplace(param_data.Kxc(u), coords, parents_indexing(u), indexing(u), cparams, Dmat, false);
    }
    //end = std::chrono::steady_clock::now();
    //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //end = std::chrono::steady_clock::now();
    //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // full
    //arma::mat Kpp = xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
    //arma::mat Kppi = arma::inv_sympd(Kpp);
    //arma::mat Kalt = Kcc - Kxc.t() * Kppi * Kxc;
    int u_par = parents(u)(parents(u).n_elem - 1);
    int u_gp = parents(u_par)(parents(u_par).n_elem - 1);
    
    //start = std::chrono::steady_clock::now();
    // updating the parent
    if(param_data.has_updated(u) == 0){
      // parents of this block have no children so they have not been updated
      invchol_block_inplace_direct(param_data.Kxx_invchol(u_par), param_data.Kxx_invchol(u_gp), 
                                   param_data.w_cond_mean_K(u_par), param_data.Rcc_invchol(u_par));
      param_data.Kxx_inv(u_par) = param_data.Kxx_invchol(u_par).t() * param_data.Kxx_invchol(u_par);
    }
    
    //end = std::chrono::steady_clock::now();
    //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //end = std::chrono::steady_clock::now();
    //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    //start = std::chrono::steady_clock::now();
    //start = std::chrono::steady_clock::now();
    // marginal predictives
    arma::uvec ones = arma::ones<arma::uvec>(1);
    arma::mat cond_mean_K = param_data.Kxc(u).t() * param_data.Kxx_inv(u_par);
    arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
    //end = std::chrono::steady_clock::now();
    //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    //start = std::chrono::steady_clock::now();
    if(sampling){
      for(int ix=0; ix<indexing(u).n_elem; ix++){
        arma::uvec uix = ones * indexing(u)(ix);
        int first_ix = ix*q;
        int last_ix = ix*q+q-1;
        arma::mat Kcc = xCovHUV(coords, uix, uix, cparams, Dmat, true);
        //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
        arma::mat Rchol = arma::chol(arma::symmatu(Kcc - cond_mean_K.rows(first_ix, last_ix) * param_data.Kxc(u).cols(first_ix, last_ix)), "lower");
        arma::vec rnvec = arma::vectorise(bigrnorm.rows(uix));
        
        arma::vec w_temp = cond_mean_K.rows(first_ix, last_ix) * w_par + Rchol * rnvec;
        w.row(indexing(u)(ix)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      }
    } else {
      arma::vec w_temp = cond_mean_K * w_par;
      w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
    }
      //arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
      
      //arma::mat Rcc = Kcc - cond_mean_K * Kxc;
      //arma::mat Rchol = arma::chol(Rcc, "lower");
      // sample
      //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
      //arma::vec w_temp = cond_mean_K * w_par + Rchol * rnvec;
      
      //w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
    
    
    //end = std::chrono::steady_clock::now();
    //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  //Rcpp::Rcout << "prediction timings: " << timings.t() << endl;
  //Rcpp::Rcout << arma::accu(timings) << endl;
  
  Zw = armarowsum(Z % w);
  
  if(verbose){
    
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict_std] done "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. \n";
    
  }
  
}

void SpamTree::deal_with_beta(){
  //switch(use_alg){
  //case 'I':
  //  bfirls_beta();
  //  break;
  //default:
    gibbs_sample_beta();
  //  break;
  //}
}

void SpamTree::bfirls_beta(){
  message("[bfirls_beta]");
  start = std::chrono::steady_clock::now();
  
  double scales = family == "gaussian" ? 1.0/tausq_inv : 1.0;
  
  arma::vec Bcoeff_new = irls_step(Bcoeff,
                                   y_available, 
                                   X_available, 
                                   Zw.rows(na_ix_all),
                                   Vi,
                                   family, scales);
  
  //converged(0) = arma::norm(Bcoeff-Bcoeff_new) < 1e-05;
  
  Bcoeff = Bcoeff_new;
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[bfirls_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void SpamTree::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv * XtX + Vi), "lower"); 
  arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
  
  arma::mat Xprecy = Vim + tausq_inv * X_available.t() * ( y_available - Zw.rows(na_ix_all));// + ywmeandiff );
  Rcpp::RNGScope scope;
  Bcoeff = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy + arma::randn(p));
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void SpamTree::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  arma::mat yrr = y_available - X_available * Bcoeff - Zw.rows(na_ix_all);
  double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
  double aparam = 2.01 + n/2.0;
  double bparam = 1.0/( 1.0 + .5 * bcore );
  
  Rcpp::RNGScope scope;
  tausq_inv = R::rgamma(aparam, bparam);
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv
                << endl;
  }
}

void SpamTree::gibbs_sample_sigmasq(){
  start = std::chrono::steady_clock::now();
  
  double oldsigmasq = sigmasq;
  double aparam = 2.01 + n_loc_ne_blocks*q/2.0; //
  double bparam = 1.0/( 1.0 + .5 * oldsigmasq * arma::accu( param_data.wcore ));
  
  Rcpp::RNGScope scope;
  sigmasq = 1.0/R::rgamma(aparam, bparam);
  
  double old_new_ratio = oldsigmasq / sigmasq;
  
  if(isnan(old_new_ratio)){
    Rcpp::Rcout << oldsigmasq << " -> " << sigmasq << " ? " << bparam << endl;
    Rcpp::Rcout << "Error with sigmasq" << endl;
    throw 1;
  }
  
  // change all K
//***#pragma omp parallel for
  for(int i=0; i<blocks_not_empty.n_elem; i++){
    int u = blocks_not_empty(i);
    //if(block_ct_obs(u) > 0){
    param_data.logdetCi_comps(u) += //block_ct_obs(u)//
      (q*indexing(u).n_elem + 0.0) 
      * 0.5*log(old_new_ratio);
    
    param_data.w_cond_prec(u) = old_new_ratio * param_data.w_cond_prec(u);
    
    if(children(u).n_elem > 0) {
      param_data.Kxx_inv(u) = old_new_ratio * param_data.Kxx_inv(u);
      param_data.Kxc(u) = 1.0/old_new_ratio * param_data.Kxc(u);
    }
    param_data.wcore(u) = old_new_ratio * param_data.wcore(u);//arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
    param_data.loglik_w_comps(u) = //block_ct_obs(u)//
      (q*indexing(u).n_elem + .0) * hl2pi -.5 * param_data.wcore(u);
  }
  
  param_data.logdetCi = arma::accu(param_data.logdetCi_comps);
  param_data.loglik_w = param_data.logdetCi + arma::accu(param_data.loglik_w_comps);
  
  //Rcpp::Rcout << "sigmasq wcore: " << arma::accu( param_data.wcore ) << endl; //##
  //Rcpp::Rcout << "sigmasq logdetCi: " << param_data.logdetCi << endl; //##
  //Rcpp::Rcout << "sigmasq loglik_w: " << param_data.loglik_w << endl; //##
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sigmasq] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                << aparam << " : " << bparam << " " << arma::accu( param_data.wcore ) << " --> " << sigmasq
                << endl;
  }
}

void SpamTree::theta_update(SpamTreeData& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void SpamTree::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void SpamTree::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void SpamTree::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}