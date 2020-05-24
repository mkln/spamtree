#include <RcppArmadillo.h>
#include <omp.h>
#include <stdexcept>

#include "R.h"
#include "find_nan.h"
#include "mh_adapt.h"
#include "field_v_concatm.h"
#include "nonseparable_huv_cov.h"
#include "multires_utils.h"
// with indexing
// without block extensions (obs with NA are left in)

using namespace std;

const double hl2pi = -.5 * log(2 * M_PI);

class LMTmesh {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  int npars;
  
  // data
  arma::mat y;
  arma::mat X;
  arma::mat Z;
  
  arma::mat y_available;
  arma::mat X_available;
  arma::mat Z_available;
  
  arma::mat coords;
  
  // block membership
  arma::uvec blocking;
  
  arma::field<arma::sp_mat> Zblock;
  arma::mat eta_rpx;
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
  arma::field<arma::sp_mat> Ib;
  arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  arma::vec                 block_groups; // same group = sample in parallel given all others
  arma::vec                 block_ct_obs; // 0 if no available obs in this block, >0=count how many available
  arma::uvec                blocks_not_empty; // list i indices of non-empty blocks
  arma::uvec                blocks_predicting;
  int                       n_gibbs_groups;
  arma::field<arma::vec>    u_by_block_groups;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  arma::field<arma::vec>    dim_by_parent;
  
  // params
  arma::mat w;
  arma::vec Bcoeff; // sampled
  double    tausq_inv;
  double    sigmasq;
  
  // recursive multires stuff
  std::vector<std::function<arma::mat(const MeshData&, 
                                      const arma::field<arma::uvec>&, 
                                      int, int, int, int, bool)> > xCrec;
  
  
  // params with mh step
  MeshData param_data; 
  MeshData alter_data;
  
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
  void init_multires();
  
  arma::vec cparams;
  arma::mat Dmat;
  void theta_transform(const MeshData&);
  void get_recursive_funcs();
  
  
  void na_study();
  void make_gibbs_groups();
  
  
  // MCMC
  char use_alg;
  
  void get_loglik_w(MeshData& data);
  void get_loglik_w_std(MeshData& data);
  void get_loglik_w_rec(MeshData& data);
  
  void get_loglik_comps_w(MeshData& data);
  void get_loglik_comps_w_std(MeshData& data);
  void get_loglik_comps_w_rec(MeshData& data);
  
  void gibbs_sample_w();
  void gibbs_sample_w_std();
  void gibbs_sample_w_rpx();
  void gibbs_sample_w_rec();
  
  void gibbs_sample_beta();
  void gibbs_sample_sigmasq();
  void gibbs_sample_tausq();
  
  void predict();
  void predict_std();
  void predict_rec();
  
  // changing the values, no sampling
  void tausq_update(double);
  void theta_update(MeshData&, const arma::vec&);
  void beta_update(const arma::vec&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  LMTmesh();
  
  // build everything
  LMTmesh(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& blocking_in,
    
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
    bool v,
    bool debugging);
  
};

void LMTmesh::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

LMTmesh::LMTmesh(){
  
}

LMTmesh::LMTmesh(
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  const arma::uvec& blocking_in,
  
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
  bool v=false,
  bool debugging=false){
  
  use_alg = use_alg_in; //S:standard, P:residual process, R: rp recursive
  
  message("~ LMTmesh initialization.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  
  verbose = v;
  debug = debugging;
  
  message("LMTmesh::LMTmesh assign values.");
  
  y                   = y_in;
  X                   = X_in;
  Z                   = Z_in;
  
  coords              = coords_in;
  blocking            = blocking_in;
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
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
  eta_rpx = arma::zeros(coords.n_rows, n_gibbs_groups);
  
  indexing    = indexing_in;
  
  if(dd == 2){
    if(q < 2){
      npars = 1;
    } else {
      npars = 5;
    }
  } else {
    if(q < 3){
      npars = 3;
    } else {
      npars = 5;
    }
  }
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  // init
  dim_by_parent       = arma::field<arma::vec> (n_blocks);
  Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  
  
  tausq_inv        = tausq_inv_in;
  sigmasq          = sigmasq_in;
  Bcoeff           = beta_in;
  w                = w_in;
  
  predicting = true;
  
  // now elaborate
  message("LMTmesh::LMTmesh : init_indexing()");
  init_indexing();
  
  message("LMTmesh::LMTmesh : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("LMTmesh::LMTmesh : init_finalize()");
  init_finalize();
  
  
  // prior for beta
  XtX   = X_available.t() * X_available;
  Vi    = .01 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  message("LMTmesh::LMTmesh : make_gibbs_groups()");
  make_gibbs_groups();
  
  
  // data for metropolis steps and predictions
  // block params
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.Kxc           = arma::field<arma::mat> (n_blocks);
  param_data.Kxx_inv       = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  param_data.theta          = arma::join_vert(arma::ones(1) * sigmasq, theta_in);
  param_data.cholfail       = false;
  param_data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  
  xCrec.reserve(n_blocks);
  init_multires();
  
  alter_data                = param_data; 
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "LMTmesh::LMTmesh initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}

void LMTmesh::make_gibbs_groups(){
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
  
  
  u_by_block_groups = arma::field<arma::vec> (n_gibbs_groups);
  /// create list of groups for gibbs
  #pragma omp parallel for
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups(g) = arma::zeros(0);
    
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(block_ct_obs(u) > 0){ //**
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups(g) = arma::join_vert(u_by_block_groups(g), uhere);
        } 
      }
    }
  }
}

void LMTmesh::na_study(){
  // prepare stuff for NA management
  message("[na_study] NA management for predictions\n");
  
  na_1_blocks = arma::field<arma::vec> (n_blocks);//(y_blocks.n_elem);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);//(y_blocks.n_elem);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);//(y_blocks.n_elem);
  
  #pragma omp parallel for
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
  
  blocks_not_empty = arma::find(block_ct_obs > 0);
  blocks_predicting = arma::find(block_ct_obs == 0);
  int ne=0;
  int pr=0;
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(block_ct_obs(u) > 0){
      blocks_not_empty(ne) = u;
      ne ++;
    } else {
      blocks_predicting(pr) = u;
      pr ++;
    }
  }
  
}

void LMTmesh::init_indexing(){
  
  Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);

  message("[init_indexing] indexing, parent_indexing, children_indexing");
  
  #pragma omp parallel for
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

void LMTmesh::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
  #pragma omp parallel for //**
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
      //Rcpp::Rcout << "Ib " << parents(u).n_elem << endl;
      //Ib(u) = arma::eye<arma::sp_mat>(coords_blocks(u).n_rows, coords_blocks(u).n_rows);
      Ib(u) = arma::eye<arma::sp_mat>(indexing(u).n_elem, indexing(u).n_elem);
      for(int j=0; j<Ib(u).n_cols; j++){
        if(na_1_blocks(u)(j) == 0){
          Ib(u)(j,j) = 0;//1e-6;
        }
      }
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
  
  #pragma omp parallel for // **
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    //Rcpp::Rcout << "block: " << u << "\n";
    
    //if(coords_blocks(u).n_elem > 0){ //**
    if(indexing(u).n_elem > 0){
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
        //Rcpp::Rcout << "from: " << firstcol << " to: " << lastcol << endl; 
        
        int dimen = parents_indexing(child).n_elem;
        arma::vec colix = arma::zeros(q*dimen);//parents_coords(child).n_rows);//(w_cond_mean_K(child).n_cols);
        //arma::uvec indx_scheme = arma::regspace<arma::uvec>(0, q, q*(dimen-1));
        
        for(int s=0; s<q; s++){
          //arma::uvec c_indices = s + indx_scheme.subvec(firstcol, lastcol-1);

          int shift = s * dimen;
          colix.subvec(shift + firstcol, shift + lastcol-1).fill(1);
          //colix.elem(c_indices).fill(1);
        }
        //Rcpp::Rcout << indx_scheme << "\n";
        //Rcpp::Rcout << colix << "\n";
        
        //Rcpp::Rcout << "visual representation of which ones we are looking at " << endl
        //            << colix.t() << endl;
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = arma::find(colix == 1); // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = arma::find(colix != 1); // u parent of c is NOT in these columns for c
        
      }
    }
  }
}

void LMTmesh::get_loglik_w(MeshData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  switch(use_alg){
  case 'S':
    get_loglik_w_std(data);
    break;
  case 'P':
    get_loglik_w_std(data);
    break;
  case 'R': 
    get_loglik_w_rec(data);
    break;
  }
}

void LMTmesh::get_loglik_w_std(MeshData& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
  
  //arma::uvec blocks_not_empty = arma::find(block_ct_obs > 0);
 #pragma omp parallel for //**
  for(int i=0; i<blocks_not_empty.n_elem; i++){
    int u = blocks_not_empty(i);
    //if(block_ct_obs(u) > 0){
    //int p = parents(u).n_elem > 0? parents(u)(parents(u).n_elem - 1) : -1;
    
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if((block_ct_obs(u) > 0) & calc_this_block){
    //if(block_ct_obs(u) > 0){
      //double expcore = -.5 * arma::conv_to<double>::from( (w_blocks(u) - data.w_cond_mean(u)).t() * data.w_cond_prec(u) * (w_blocks(u) - data.w_cond_mean(u)) );
      arma::mat w_x = arma::vectorise( arma::trans( w.rows(indexing(u)) ) );
      
      if(parents(u).n_elem > 0){
        arma::vec w_pars = arma::vectorise( arma::trans( w.rows(parents_indexing(u))));//indexing(p)) ));
        w_x -= data.w_cond_mean_K(u) * w_pars;
      }
      
      //w_x = w_x % na_1_blocks(u);
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      
      data.loglik_w_comps(u) = //block_ct_obs(u)//
        (q*indexing(u).n_elem + .0) 
        * hl2pi -.5 * data.wcore(u);
    //} else {
    //  data.wcore(u) = 0;
    //  data.loglik_w_comps(u) = 0;
    //}
    //}
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


void LMTmesh::get_loglik_w_rec(MeshData& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
  
  for(int g=0; g<n_gibbs_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    
    #pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      arma::vec eta_x = eta_rpx.submat(indexing(u), grpuvec);
      data.wcore(u) = arma::conv_to<double>::from(eta_x.t() * data.w_cond_prec(u) * eta_x);
      data.loglik_w_comps(u) = (q*indexing(u).n_elem + .0) * hl2pi -.5 * data.wcore(u);
    }
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


void LMTmesh::theta_transform(const MeshData& data){
  cparams = data.theta;
  Dmat = arma::zeros(1,1);
}

void LMTmesh::get_recursive_funcs(){

  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    
    if(parents(u).n_elem == 0){
      std::function<arma::mat(const MeshData&, const arma::field<arma::uvec>&, int, int, int, int, bool)> Khere = 
        [&](const MeshData& localdata, const arma::field<arma::uvec>& indexing, int ux, int i1, int i2, int depth, bool same){
          arma::mat result = xCovHUV(coords, indexing(i1), indexing(i2), cparams, Dmat, same);
          return result;
        };
        xCrec.at(u) = Khere;  
    } else {
      std::function<arma::mat(const MeshData&, const arma::field<arma::uvec>&, int, int, int, int, bool)> Khere = 
        [&](const MeshData& localdata, const arma::field<arma::uvec>& indexing, int ux, int i1, int i2, int depth, bool same){
          if(depth==0){
            arma::mat result = xCovHUV(coords, indexing(i1), indexing(i2), cparams, Dmat, i1==i2);
            return result;
          } else {
            int lpx = parents(ux)(parents(ux).n_elem - 1);
            same = i1==i2;
            arma::mat Kcc, Krem;
            if(same){
              Kcc  = xCrec.at(lpx)(localdata, indexing, lpx, i1, i2, depth-1, true);
              arma::mat Kc1x = xCrec.at(lpx)(localdata, indexing, lpx, i1, lpx, depth-1, false);
              Krem = Kc1x * localdata.w_cond_prec(lpx) * Kc1x.t();
            } else {
              Kcc  = xCrec.at(lpx)(localdata, indexing, lpx, i1, i2, depth-1, false);
              arma::mat Kc1x = xCrec.at(lpx)(localdata, indexing, lpx, i1, lpx, depth-1, false);
              arma::mat Kxc2 = xCrec.at(lpx)(localdata, indexing, lpx, lpx, i2, depth-1, false);
              Krem = Kc1x * localdata.w_cond_prec(lpx) * Kxc2;
            }
            arma::mat result = Kcc - Krem;
            return result;
          }
        };
        xCrec.at(u) = Khere;
    }
  }

}

void LMTmesh::init_multires(){
  //Kxx_chol = arma::field<arma::mat> (n_blocks);
  param_data.Kxx_invchol = arma::field<arma::mat> (n_blocks); // storing the inv choleskys of {parents(w), w} (which is parent for children(w))
  param_data.Rcc_invchol = arma::field<arma::mat> (n_blocks); 
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(block_ct_obs(u)>0){
      param_data.Kxx_invchol(u) = arma::zeros(parents_indexing(u).n_elem + indexing(u).n_elem, //Kxx_invchol(last_par).n_rows + Kcc.n_rows, 
                  parents_indexing(u).n_elem + indexing(u).n_elem);//Kxx_invchol(last_par).n_cols + Kcc.n_cols);
      //Kxx_chol(u) = Kxx_invchol(u);
    }
    std::function<arma::mat(const MeshData&, const arma::field<arma::uvec>&, int, int, int, int, bool)> dummy_foo = 
      [&](const MeshData&, const arma::field<arma::uvec>& indexing, int ux, int i1, int i2, int depth, bool same){
        return arma::zeros(0,0);
      };
    xCrec.push_back(dummy_foo);
  }
  
  Rcpp::Rcout << "init_multires: indexing elements: " << indexing.n_elem << endl;

}

void LMTmesh::get_loglik_comps_w(MeshData& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  switch(use_alg){
  case 'S':
    get_loglik_comps_w_std(data);
    break;
  case 'P':
    get_loglik_comps_w_std(data);
    break;
  case 'R': 
    get_loglik_comps_w_rec(data);
    break;
  }
}

void LMTmesh::get_loglik_comps_w_std(MeshData& data){
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_w] start. ");
  
  //arma::vec timings = arma::zeros(5);
  
  theta_transform(data);
  
  // inv chol of Kxx where x are the parents 
  
  //arma::field<arma::mat> Kcx_store(n_blocks);
  //arma::field<arma::mat> Rcc_invchol(n_blocks); // conditional precision of w | pa(w)
  //arma::field<arma::mat> Rcc_chol(n_blocks);
  //arma::field<arma::mat> Kxx_inv(n_blocks);
  
  // cycle through the resolutions starting from the bottom
  
  for(int g=0; g<n_gibbs_groups; g++){
#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(block_ct_obs(u) > 0){
      //start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "u: " << u << " obs: " << block_ct_obs(u) << " parents: " << parents(u).t() << "\n"; 
      arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
      arma::mat cond_mean_K, cond_mean;
      arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
      //Rcpp::Rcout << "step 1\n";
      //end = std::chrono::steady_clock::now();
      //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      if(parents(u).n_elem == 0){
        //arma::mat Kxx = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
        //Kcx_store(u) = 
        //Kxx_chol(u) = arma::chol(Kcc, "lower");
        //Rcc_chol(u) = Kxx_chol(u);
        data.Kxx_invchol(u) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
        data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
        data.Rcc_invchol(u) = data.Kxx_invchol(u); 
        cond_mean_K = arma::zeros(0, 0);
      } else {
        //Rcpp::Rcout << "step 2\n";
        int last_par = parents(u)(parents(u).n_elem - 1);
        //arma::mat LAi = Kxx_invchol(last_par);
        
        //start = std::chrono::steady_clock::now();
        data.Kxc(u) = xCovHUV(coords, parents_indexing(u), indexing(u), cparams, Dmat, false);
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        cond_mean_K = data.Kxc(u).t() * data.Kxx_inv(last_par);//Kxx_invchol(last_par).t() * Kxx_invchol(last_par);
        //end = std::chrono::steady_clock::now();
        //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc - cond_mean_K * data.Kxc(u)), "lower")));
        
        arma::vec w_pars = arma::vectorise(arma::trans( w.rows(parents_indexing(u))));//indexing(p))));
        w_x -= cond_mean_K * w_pars;
        //end = std::chrono::steady_clock::now();
        //timings(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if(children(u).n_elem > 0){
          
          //Kxx_invchol(u) = invchol_block(LAi, Kxc, Kcc);
          //arma::mat test = Kxx_invchol(u);
          //invchol_block_inplace(Kxx_chol(u), Kxx_invchol(u), Kxx_chol(last_par), LAiBt, Rcc_chol(u));
          invchol_block_inplace_direct(data.Kxx_invchol(u), data.Kxx_invchol(last_par), cond_mean_K, data.Rcc_invchol(u));
          data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
          /*
           arma::mat Kxx = xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
           arma::mat Kxxi = arma::inv_sympd( Kxx );
           double checker = arma::accu(abs(data.Kxx_inv(last_par) - Kxxi));
           if(checker > 1e-3){ 
           Rcpp::Rcout << "u: " << u << endl;
           Rcpp::Rcout << "in calc comps: " << checker << endl;
           throw 1;
           }*/
          
        }
      }
      
      
      //start = std::chrono::steady_clock::now();
      //Rcpp::Rcout << "step 6\n";
      //arma::mat cond_cholprec = Rcc_invchol(u);//
      //Rcpp::Rcout << "step 3\n";
      data.w_cond_mean_K(u) = cond_mean_K;
      data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
      
      //if(block_ct_obs(u) > 0){
      arma::vec ccholprecdiag = data.Rcc_invchol(u).diag();
      data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));
      
      //w_x = w_x % na_1_blocks(u);
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      data.loglik_w_comps(u) = (q*indexing(u).n_elem+.0) 
        * hl2pi -.5 * data.wcore(u);
      //}
      //end = std::chrono::steady_clock::now();
      //timings(4) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      //} else {
      // Rcpp::Rcout << "You should not see this." << endl;
      //  data.wcore(u) = 0;
      //  data.loglik_w_comps(u) = 0;
      //  data.logdetCi_comps(u) = 0;
      //}
    }
  }
  
  //Rcpp::Rcout << "timings: " << timings.t() << endl;
  //Rcpp::Rcout << "total timings: " << arma::accu(timings) << endl;
  data.logdetCi = arma::accu(data.logdetCi_comps.subvec(0, n_blocks-1));
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps.subvec(0, n_blocks-1));
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_comps_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}

void LMTmesh::get_loglik_comps_w_rec(MeshData& data){
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_w] start. ");
  
  //arma::vec timings = arma::zeros(5);
  theta_transform(data);
  get_recursive_funcs();
  
  int K = eta_rpx.n_cols; // n. resolutions
  int depth = K;
  // cycle through the resolutions starting from the bottom
  
  for(int g=0; g<n_gibbs_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    //arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    //arma::vec eta_others = armarowsum(eta_rpx.submat(indexing(u), other_etas));
    
#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //Rcpp::Rcout << "u: " << u << " at resolution " << grp << " with " << parents(u).n_elem << " parents. \n";
      
      arma::mat Kcc = xCrec.at(u)(data, indexing, u, u, u, depth, true);
      
      //Rcpp::Rcout << "getting w_x" << endl;
      arma::vec w_x = eta_rpx.submat(indexing(u), grpuvec);
      
      //Rcpp::Rcout << "getting Rcc_invchol" << endl;
      data.Rcc_invchol(u) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc), "lower")));
      
      //Rcpp::Rcout << "getting w_cond_prec" << endl;
      data.w_cond_prec(u) = data.Rcc_invchol(u).t() * data.Rcc_invchol(u);//Rcc_invchol(u).t() * Rcc_invchol(u);
      
      //if(block_ct_obs(u) > 0){
      //Rcpp::Rcout << "getting logdetCi_comps" << endl;
      arma::vec ccholprecdiag = data.Rcc_invchol(u).diag();
      data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));
      
      //Rcpp::Rcout << "getting wcore, loglik_w_comps" << endl;
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
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
    Rcpp::Rcout << "[get_loglik_comps_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}

void LMTmesh::gibbs_sample_w(){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  switch(use_alg){
  case 'S':
    gibbs_sample_w_std();
    break;
  case 'P':
    gibbs_sample_w_rpx();
    break;
  case 'R': 
    gibbs_sample_w_rec();
    break;
  }
}

void LMTmesh::gibbs_sample_w_std(){
  // keep seed
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] sampling big stdn matrix size " << n << "," << q << endl;
  }
  
  Rcpp::RNGScope scope;
  arma::mat rand_norm_mat = arma::randn(coords.n_rows, q);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  
  for(int g=0; g<n_gibbs_groups; g++){
    #pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //Rcpp::Rcout << "step 1 \n";
      
        //Rcpp::Rcout << "step 2\n";
        arma::mat Smu_tot = arma::zeros(q*indexing(u).n_elem, 1);
        arma::mat Sigi_tot = param_data.w_cond_prec(u); // Sigi_p
        
        //Rcpp::Rcout << "step 3\n";
        arma::vec w_par;
        if(parents(u).n_elem>0){
          w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
          Smu_tot += Sigi_tot * param_data.w_cond_mean_K(u) * w_par;//param_data.w_cond_mean(u);
        }
        
        //arma::uvec ug = arma::zeros<arma::uvec>(1) + g;
        // indexes being used as parents in this group
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //if(block_ct_obs(child) > 0){
          //if(block_groups(child) == block_groups(u)+1){// one res up
            //Rcpp::Rcout << "child: " << child << endl;
            //clog << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
            //Rcpp::Rcout << u_is_which_col_f(u)(c)(0).t() << "\n";
            //Rcpp::Rcout << u_is_which_col_f(u)(c)(1).t() << "\n";
            //Rcpp::Rcout << "child step 1\n"; 
            arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
            //Rcpp::Rcout << "child step 2\n"; 
            arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
            arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
            //Rcpp::Rcout << "child step 3\n"; 
            arma::vec w_child = arma::vectorise(arma::trans( w.rows( indexing(child) ) ));
            
            //Rcpp::Rcout << "child part 3 \n";
            arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
            arma::vec w_par_child = arma::vectorise(arma::trans(w_parents_of_child));
            arma::vec w_par_child_select = w_par_child.rows(u_is_which_col_f(u)(c)(1));
            //Rcpp::Rcout << arma::size(Sigi_tot) << " " << arma::size(AK_u) << " " << arma::size(AK_uP) << "\n";
            
            Sigi_tot += AK_uP * AK_u;
            Smu_tot += AK_uP * (w_child - AK_others * w_par_child_select );
          //}
          //}
        }
        
        //Rcpp::Rcout << "step 4\n";
        Sigi_tot += tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u);
        Smu_tot += Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
          ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
        
        arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        
        //Rcpp::Rcout << "5 " << endl;
        //Rcpp::Rcout << "step 6\n";
        // sample
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
        arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        
        //rand_norm_tracker(indexing(u), ug) += 1;
        //Rcpp::Rcout << w_temp.n_elem/q << " " << q << "\n";
        w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      
    }
  }
  
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

void LMTmesh::gibbs_sample_w_rpx(){
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_rpx] " << endl;
  }
  
  // covariance parameters
  theta_transform(param_data);
  
  Rcpp::RNGScope scope;
  arma::mat rand_norm_mat = arma::randn(coords.n_rows, q);
  
  //arma::vec timings = arma::zeros(5);
  //Rcpp::Rcout << "starting loops \n";
  //Rcpp::Rcout << "groups: " << block_groups_labels.t() << endl;
  
  for(int g=0; g<n_gibbs_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    
  #pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(block_ct_obs(u)>0){
      //Rcpp::Rcout << other_etas.t() << endl;
      //Rcpp::Rcout << arma::size(eta_rpx) << endl;
      //Rcpp::Rcout << indexing(u).min() << " to " << indexing(u).max() << endl;
      
      //start = std::chrono::steady_clock::now();
      arma::vec eta_others = armarowsum(eta_rpx.submat(indexing(u), other_etas));
      //Rcpp::Rcout << "precision " << endl;
      
      // precision
      arma::mat Sigi_tot = 
        tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u) + 
        param_data.w_cond_prec(u);
      arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
      
      // Smean
      arma::mat Smu_tot = Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
        ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff - eta_others));
      
      // sample independent residual
      //Rcpp::Rcout << "sampling size " << indexing(u).n_elem << endl;
      
      arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
      //arma::vec rnvec = arma::randn(indexing(u).n_elem);
      arma::vec w_local = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
      
      eta_rpx.submat(indexing(u), grpuvec) = w_local;
      
      //end = std::chrono::steady_clock::now();
      //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      for(int c=0; c<children(u).n_elem; c++){
        int cx = children(u)(c);
        //if(block_ct_obs(cx) > 0){
        //start = std::chrono::steady_clock::now();
        arma::mat Kchildc = xCovHUV(coords, indexing(cx), indexing(u), cparams, Dmat, false);
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
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
        
        eta_rpx.submat(indexing(cx), grpuvec) = Kchildren * param_data.w_cond_prec(u) * w_local;
        //} else {
        //  Rcpp::Rcout << "You should not be reading this. \n";
        //  Rcpp::Rcout << "u: " << u << " with child: " << cx << " (empty) " << endl;
        //}
      }
      
    }
  }
  
  //start = std::chrono::steady_clock::now();
  w = armarowsum(eta_rpx);
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

void LMTmesh::gibbs_sample_w_rec(){
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_rec] " << endl;
  }
  
  // covariance parameters
  theta_transform(param_data);
  get_recursive_funcs();
  
  int K = eta_rpx.n_cols;
  int depth = K;
  
  Rcpp::RNGScope scope;
  arma::mat rand_norm_mat = arma::randn(coords.n_rows, q);
  
  //arma::vec timings = arma::zeros(5);
  //Rcpp::Rcout << "starting loops \n";
  //Rcpp::Rcout << "groups: " << block_groups_labels.t() << endl;
  
  for(int g=0; g<n_gibbs_groups; g++){
    int grp = block_groups_labels(g) - 1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    
#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(block_ct_obs(u)>0){
      //Rcpp::Rcout << other_etas.t() << endl;
      //Rcpp::Rcout << arma::size(eta_rpx) << endl;
      //Rcpp::Rcout << indexing(u).min() << " to " << indexing(u).max() << endl;
      
      //start = std::chrono::steady_clock::now();
      arma::vec eta_others = armarowsum(eta_rpx.submat(indexing(u), other_etas));
      //Rcpp::Rcout << "precision " << endl;
      
      // precision
      arma::mat Sigi_tot = 
        tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u) + 
        param_data.w_cond_prec(u);
      arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
      
      // Smean
      arma::mat Smu_tot = Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
        ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff - eta_others));
      
      // sample independent residual
      //Rcpp::Rcout << "sampling size " << indexing(u).n_elem << endl;
      
      arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
      //arma::vec rnvec = arma::randn(indexing(u).n_elem);
      arma::vec w_local = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
      
      eta_rpx.submat(indexing(u), grpuvec) = w_local;
      
      //end = std::chrono::steady_clock::now();
      //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      for(int c=0; c<children(u).n_elem; c++){
        // look at children resolution to compare with depth
        int cx = children(u)(c);
        int child_res = block_groups(cx);
        int u_res = block_groups(u);
        int diff_res = child_res - u_res;
        
        if(diff_res <= depth){
          // using recursive functions
          // "incomplete multiresolution decomposition" obtained by setting depth < n_pars_cx-1
          arma::mat Kchildren = xCrec.at(u)(param_data, indexing, u, cx, u, depth, false);
          eta_rpx.submat(indexing(cx), grpuvec) = Kchildren * param_data.w_cond_prec(u) * w_local;
        } else {
          Rcpp::Rcout << "got c>depth with " << child_res << " res at child, " << u_res << " res here, and " << depth << " depth\n";
        }
      }
      
    }
  }
  
  //start = std::chrono::steady_clock::now();
  w = armarowsum(eta_rpx);
  Zw = armarowsum(Z % w);
  
  if(verbose){
    //end = std::chrono::steady_clock::now();
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_rec] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us and " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us \n";
    //Rcpp::Rcout << timings.t() << endl;
  }
}

void LMTmesh::predict(){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  switch(use_alg){
  case 'S':
    predict_std();
    break;
  case 'P':
    predict_std();
    break;
  case 'R': 
    predict_rec();
    break;
  }
}

void LMTmesh::predict_std(){
  start_overall = std::chrono::steady_clock::now();
  message("[predict] start. ");
  
  //arma::vec timings = arma::zeros(5);
  theta_transform(param_data);
  
  //arma::vec timings = arma::zeros(3);
  
  
  // cycle through the resolutions starting from the bottom
  //arma::uvec predicting_blocks = arma::find(block_ct_obs == 0);
#pragma omp parallel for
  for(int i=0; i<blocks_predicting.n_elem; i++){
    int u = blocks_predicting(i);
    // meaning this block must be predicted
    
    //start = std::chrono::steady_clock::now();
    arma::mat Kxc = xCovHUV(coords, parents_indexing(u), indexing(u), cparams, Dmat, false);
    //end = std::chrono::steady_clock::now();
    //timings(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // full
    //arma::mat Kpp = xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
    //arma::mat Kppi = arma::inv_sympd(Kpp);
    //arma::mat Kalt = Kcc - Kxc.t() * Kppi * Kxc;
    
    //start = std::chrono::steady_clock::now();
    // updating the parent
    int u_par = parents(u)(parents(u).n_elem - 1);
    int u_gp = parents(u_par)(parents(u_par).n_elem - 1);
    invchol_block_inplace_direct(param_data.Kxx_invchol(u_par), param_data.Kxx_invchol(u_gp), param_data.w_cond_mean_K(u_par), param_data.Rcc_invchol(u_par));
    param_data.Kxx_inv(u_par) = param_data.Kxx_invchol(u_par).t() * param_data.Kxx_invchol(u_par);
    //end = std::chrono::steady_clock::now();
    //timings(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    //start = std::chrono::steady_clock::now();
    // marginal predictives
    arma::uvec ones = arma::ones<arma::uvec>(1);
    arma::mat cond_mean_K = Kxc.t() * param_data.Kxx_inv(u_par);
    arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u))));//indexing(p) ) ));
    for(int ix=0; ix<indexing(u).n_elem; ix++){
      arma::uvec uix = ones * indexing(u)(ix);
      arma::mat Kcc = xCovHUV(coords, uix, uix, cparams, Dmat, true);
      double Rcc = arma::conv_to<double>::from(Kcc(0,0) - cond_mean_K.row(ix) * Kxc.col(ix));
      
      // sample
      arma::vec rnvec = arma::randn(1);
      w.row(indexing(u)(ix)) = cond_mean_K.row(ix) * w_par + sqrt(Rcc) * rnvec;
    }
    //end = std::chrono::steady_clock::now();
    //timings(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  //Rcpp::Rcout << timings.t() << endl;
  //Rcpp::Rcout << arma::accu(timings) << endl;
  
  Zw = armarowsum(Z % w);
  if(verbose){
    
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict] done "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. \n";
    
  }
  
}

void LMTmesh::predict_rec(){
  start_overall = std::chrono::steady_clock::now();
  message("[predict] start. ");
  
  //arma::vec timings = arma::zeros(5);
  theta_transform(param_data);
  get_recursive_funcs();
  
  //arma::vec timings = arma::zeros(3);
  int K = eta_rpx.n_cols;
  int depth = K;
  
  // cycle through the resolutions starting from the bottom
  //arma::uvec predicting_blocks = arma::find(block_ct_obs == 0);
#pragma omp parallel for
  for(int i=0; i<blocks_predicting.n_elem; i++){
    int u = blocks_predicting(i);
    // meaning this block must be predicted
    //Rcpp::Rcout << "u: " << u << " <-- predicting\n";
    
    int grp = eta_rpx.n_cols-1;
    arma::uvec grpuvec = arma::ones<arma::uvec>(1) * grp;
    arma::uvec other_etas = arma::find(block_groups_labels != grp+1);
    
    //Rcpp::Rcout << "get predictions from all parents\n";
    // get predictions from all parents

    //for(int p=0; p<depth; p++){
      //if(parents(u).n_elem - p > 1){
        //int px = parents(u).n_elem - p - 1;
        
      for(int px=0; px<parents(u).n_elem; px++){
        if(px < depth){
          int par = parents(u)(parents(u).n_elem - px - 1);
          //Rcpp::Rcout << "parent: " << par << endl;
          arma::uvec grpparvec = arma::ones<arma::uvec>(1) * (block_groups(par) - 1);
          //Rcpp::Rcout << "got Kcx: " << arma::size(Kcx) << " and prec: " << arma::size(param_data.w_cond_prec(par)) << endl;
          arma::mat Kcx = xCrec.at(par)(param_data, indexing, par, u, par, depth, false);
          eta_rpx.submat(indexing(u), grpparvec) = Kcx * param_data.w_cond_prec(par) * eta_rpx.submat(indexing(par), grpparvec);
          
        } else {
          Rcpp::Rcout << "got px >= depth with " << parents(u).n_elem << " parents and " << depth << " depth\n";
        }
      }  
      //}
    //}
    //arma::vec eta_others = armarowsum(eta_rpx.submat(indexing(u), other_etas));
    
    //Rcpp::Rcout << "then with the recursive covariance, residuals are independent\n";
    // then with the recursive covariance, residuals are independent
    arma::mat Kcc = xCrec.at(u)(param_data, indexing, u, u, u, depth, true);
    //Rcpp::Rcout << "finally\n";
    
    arma::vec Kcc_diag = Kcc.diag();
    
    arma::vec rnvec = arma::randn(indexing(u).n_elem);
    
    for(int ix=0; ix<indexing(u).n_elem; ix++){
      double Rcc = Kcc_diag(ix);
      // sample
      eta_rpx(indexing(u)(ix), grp) = sqrt(Rcc) * rnvec(ix);
    }
    
  }
  
  w = armarowsum(eta_rpx);
  Zw = armarowsum(Z % w);
  
  if(verbose){
    
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict] done "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. \n";
    
  }
  
}

void LMTmesh::gibbs_sample_beta(){
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

void LMTmesh::gibbs_sample_tausq(){
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

void LMTmesh::gibbs_sample_sigmasq(){
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
#pragma omp parallel for
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
      (q*indexing(u).n_elem + .0) 
      * hl2pi -.5 * param_data.wcore(u);
    //} else {
    //  param_data.wcore(u) = 0;
    //  param_data.loglik_w_comps(u) = 0;
    //}
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


void LMTmesh::theta_update(MeshData& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void LMTmesh::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void LMTmesh::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void LMTmesh::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}