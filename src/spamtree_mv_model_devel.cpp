#define ARMA_DONT_PRINT_ERRORS
#include "spamtree_mv_model_devel.h"


bool matsame(const arma::mat& x, const arma::mat& y, double tol=1e-5){
  if((arma::size(x) == arma::size(y)) & ((x.n_rows + x.n_cols)*x.n_rows*x.n_cols > 0)){
    arma::mat absdiff = abs(x-y);
    return absdiff.max() < tol;
  }
  return false;
}


arma::uvec mat_sortix(const arma::mat& x, const arma::urowvec& ascending) {
  // Order the elements of x by sorting starting from the first column
  // we order by first column and resolve ties with other columns
  // First create a vector of indices
  arma::uvec idx = arma::regspace<arma::uvec>(0, x.n_rows - 1);
  // Then sort that vector by the values of y and z
  std::sort(idx.begin(), idx.end(), [&](int i, int j){
    if ( x(i,0) == x(j,0) ) {
      if( x(i, 1) == x(j, 1) ){
        if( x(i, 2) == x(j, 2) ){
          if( x(i, 3) == x(j, 3) ){
            if(ascending(4) == 1){
              return x(i, 4) < x(j, 4);
            } else {
              return x(i, 4) > x(j, 4);
            }
          }
          if(ascending(3) == 1){
            return x(i, 3) < x(j, 3);
          } else {
            return x(i, 3) > x(j, 3);
          }
        }
        if(ascending(2) == 1){
          return x(i, 2) < x(j, 2);
        } else {
          return x(i, 2) > x(j, 2);
        }
      }
      if(ascending(1) == 1){
        return x(i, 1) < x(j, 1);
      } else {
        return x(i, 1) > x(j, 1);
      }
    }
    if(ascending(0) == 1){
      return x(i, 0) < x(j, 0);
    } else {
      return x(i, 0) > x(j, 0);
    }
  });
  // And return x in that order
  return idx;
}


arma::vec drowcol_uv(const arma::field<arma::umat>& diag_blocks){
  int M=diag_blocks.n_elem;
  arma::vec drow = arma::zeros(M+1);
  for(int i=0; i<M; i++){
    drow(i+1) = diag_blocks(i).n_rows;
  }
  drow = arma::cumsum(drow);
  return drow;
}

arma::umat field_v_concat_um(arma::field<arma::umat> const& fuv){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_uv(fuv);
  arma::umat result = arma::zeros<arma::umat>(ddims(fuv.n_elem), fuv(0).n_cols);
  for(int j=0; j<fuv.n_elem; j++){
    if(fuv(j).n_elem>0){
      result.rows(ddims(j), ddims(j+1)-1) = fuv(j);
    }
  }
  return result;
}


arma::umat block_rotation_group(const arma::mat& coords, 
                                const arma::field<arma::uvec>& indexing,
                                const arma::field<arma::uvec>& indexing_obs,
                                const arma::field<arma::uvec>& parents, 
                                const arma::vec& block_names, 
                                const arma::field<arma::vec>& u_by_block_groups,
                                int n_actual_groups){
  // based on the position of the block in the domain,
  // this function determines which sorting of the coordinate axes
  // makes the resulting covariance matrices be equal
  
  arma::umat rot_groups = arma::zeros<arma::umat>(block_names.n_elem, 3); // 2d space + mv
  
  for(int i=0; i<block_names.n_elem; i++){
    int u = block_names(i) - 1;
    arma::uvec indexing_u = indexing(u);
    arma::uvec at_last_level = arma::find(u_by_block_groups(n_actual_groups-1) == u, 1, "first");
    if(at_last_level.n_elem > 0){
      indexing_u = indexing_obs(u);
    }
      
    //Rcpp::Rcout << "i " << i << " u " << u << endl;
    //Rcpp::Rcout << arma::size(coords) << endl;
    if((indexing_u.n_elem > 0) & (parents(u).n_elem > 0)){
      int last_par = parents(u)(parents(u).n_elem - 1);
      arma::mat coords_parent;
      arma::rowvec parent_centroid;
      
      arma::mat coords_block;
      arma::rowvec block_centroid;
      
      try {
        coords_parent = coords.rows(indexing(last_par));
        parent_centroid = arma::sum(coords_parent, 0) / (.0+coords_parent.n_elem);
        
        coords_block = coords.rows(indexing_u);
        block_centroid = arma::sum(coords_block, 0) / (.0+coords_block.n_elem);
        
      } catch (...) {
        Rcpp::Rcout << "got error " << endl;
        Rcpp::Rcout << indexing(last_par).t() << endl
                    << indexing_u.t() << endl;
      }
      
      //if(parents(u).n_elem > 0){
      if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
        // top right: ascending=0,0
        rot_groups(u,0) = 0;
        rot_groups(u,1) = 0;
      }
      if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
        // top left: ascending=1,0
        rot_groups(u,0) = 1;
        rot_groups(u,1) = 0;
      }
      if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
        // bottom right: ascending=0,1
        rot_groups(u,0) = 0;
        rot_groups(u,1) = 1;
      }
      if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
        // bottom left : ascending=1,1
        rot_groups(u,0) = 1;
        rot_groups(u,1) = 1;
        //}
      }/* else {
 // *INACTIVE*
 int gp = parents(u)(parents(u).n_elem - 2);
 arma::mat coords_gp = coords.rows(indexing(gp));
 arma::rowvec gp_centroid = arma::sum(coords_gp, 0) / (.0+coords_gp.n_elem);
 if((block_centroid(0) > gp_centroid(0)) & (block_centroid(1) > gp_centroid(1))){
 // top right of gp
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top right
 rot_groups(u) = 1;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top left
 rot_groups(u) = 2;
 }
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom right
 rot_groups(u) = 3;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom left
 rot_groups(u) = 4;
 }
 // ---------------
 }
 if((block_centroid(0) < gp_centroid(0)) & (block_centroid(1) > gp_centroid(1))){
 // top left of gp
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top right
 rot_groups(u) = 4;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top left
 rot_groups(u) = 1;
 }
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom right
 rot_groups(u) = 3;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom left
 rot_groups(u) = 2;
 }
 // ---------------
 }
 if((block_centroid(0) > gp_centroid(0)) & (block_centroid(1) < gp_centroid(1))){
 // bottom right of gp
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top right
 rot_groups(u) = 2;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top left
 rot_groups(u) = 3;
 }
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom right
 rot_groups(u) = 1;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom left
 rot_groups(u) = 4;
 }
 // ---------------
 }
 if((block_centroid(0) < gp_centroid(0)) & (block_centroid(1) < gp_centroid(1))){
 // bottom left of gp
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top right
 rot_groups(u) = 3;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) > parent_centroid(1))){
 // top left
 rot_groups(u) = 2;
 }
 if((block_centroid(0) > parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom right
 rot_groups(u) = 4;
 }
 if((block_centroid(0) < parent_centroid(0)) & (block_centroid(1) < parent_centroid(1))){
 // bottom left
 rot_groups(u) = 1;
 }
 // ---------------
 }
      }*/
    }
  }
  return rot_groups;
}

arma::field<arma::umat> parents_indexing_order(const arma::mat& coords, const arma::uvec& qmv_id,
                                               const arma::umat& rot_groups, 
                                               const arma::field<arma::uvec>& indexing,
                                               const arma::field<arma::uvec>& indexing_obs,
                                               const arma::field<arma::uvec>& parents,
                                               const arma::vec& block_names){
  
  // this function sorts the parent indices using the order that makes
  // symmetric parent sets for different blocks
  // result in the same covariance matrices
  // first column: the sorted indices
  // second column: the sorting order so sorted.elem(sorted_order) restores the original
  
  arma::field<arma::umat> par_index_reorder(block_names.n_elem);
  
  for(int i=0; i<block_names.n_elem; i++){
    int u = block_names(i)-1;
    if((indexing(u).n_elem+indexing_obs(u).n_elem > 0) & (parents(u).n_elem > 0)){
      int pixstart = 0;
      arma::field<arma::umat> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        arma::mat coords_par = coords.rows(indexing(parents(u)(pi)));
        arma::vec mv_id = arma::conv_to<arma::vec>::from(qmv_id.rows(indexing(parents(u)(pi))));
        arma::mat cmv = arma::join_horiz(coords_par, mv_id);
        
        int rotate_like = 0;
        if((pi==0) & (parents(u).n_elem == 1)){
          rotate_like = u;
        } else {
          if((pi==0) & (parents(u).n_elem > 1)){
            rotate_like = parents(u)(1);
          } else {
            rotate_like = parents(u)(pi);
          }
        }
        //Rcpp::Rcout << arma::size(cmv) << " " << arma::size(rot_groupd) << endl;
        //Rcpp::Rcout << rotate_like << endl;
        arma::uvec sortix = mat_sortix(cmv, rot_groups.row(rotate_like)); // order with
        
        pixs(pi) = arma::zeros<arma::umat>(coords_par.n_rows, 2);
        pixs(pi).col(0) = indexing(parents(u)(pi)).elem(sortix); // sorted
        pixs(pi).col(1) = pixstart + arma::sort_index(sortix);//indexing(parents(u)(pi)); // unsorted
        pixstart = arma::max(pixs(pi).col(1)) + 1;
      }
      par_index_reorder(u) = field_v_concat_um(pixs);
    }
  }
  return par_index_reorder;
}

arma::field<arma::umat> indexing_order(const arma::mat& coords, 
                                       const arma::uvec& qmv_id,
                                       const arma::umat& rot_groups, 
                                       const arma::field<arma::uvec>& indexing,
                                       const arma::field<arma::uvec>& parents,
                                       const arma::vec& block_names){
  
  // this function sorts the block indices according to the order
  // specified via the function block_rotation_group
  // for each block
  // first column: the sorted indices
  // second column: the sorting order so sorted.elem(sorted_order) restores the original
  arma::field<arma::umat> index_reorder(block_names.n_elem);
  
  for(int i=0; i<block_names.n_elem; i++){
    int u = block_names(i)-1;
    if(indexing(u).n_elem > 0){
      arma::mat coords_ix = coords.rows(indexing(u));
      arma::vec mv_id = arma::conv_to<arma::vec>::from(qmv_id.rows(indexing(u)));
      arma::mat cmv = arma::join_horiz(coords_ix, mv_id);
      arma::uvec sortix = mat_sortix(cmv, rot_groups.row(u)); // order with
      index_reorder(u) = arma::zeros<arma::umat>(indexing(u).n_rows, 2);
      index_reorder(u).col(0) = indexing(u).elem(sortix);
      index_reorder(u).col(1) = arma::sort_index(sortix);//indexing(u);
    }
  }
  return index_reorder;
}




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
  const arma::field<arma::uvec>& indexing_obs_in,
  
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
  indexing_obs = indexing_obs_in;
  
  
  covpars = CovarianceParams(dd, q, -1);

  
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
  init_caching();
  
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
    arma::vec yvec = y.rows(indexing_obs(i));
    arma::uvec y_not_na = arma::find_finite(yvec);
    //Rcpp::Rcout << "2"<< endl;
    block_ct_obs(i) = y_not_na.n_elem;
    na_1_blocks(i) = arma::zeros<arma::uvec>(yvec.n_elem);
    //Rcpp::Rcout << "3"<< endl;
    if(y_not_na.n_elem > 0){
      na_1_blocks(i).elem(y_not_na).fill(1);
    }
    
    Ib(i) = arma::eye<arma::sp_mat>(indexing_obs(i).n_elem, indexing_obs(i).n_elem);
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
  //children_indexing = arma::field<arma::uvec> (n_blocks);
  
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
    /*
     if(children(u).n_elem > 0){
     arma::field<arma::uvec> cixs(children(u).n_elem);
     for(int ci=0; ci<children(u).n_elem; ci++){
     //Rcpp::Rcout << "n. children " << children(u).n_elem << endl;
     //Rcpp::Rcout << "--> " << children(u)(ci) << endl;
     cixs(ci) = indexing(children(u)(ci));//arma::find( blocking == children(u)(ci)+1 ); // children are 0-indexed 
     }
     children_indexing(u) = field_v_concat_uv(cixs);
     }*/
    
  }
  
}

void SpamTreeMVdevel::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  //pragma omp parallel for //**
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem + indexing_obs(u).n_elem > 0){
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
  Rcpp::Rcout << "initializing param_data " << endl;
  
  param_data.has_updated   = arma::zeros<arma::uvec>(n_blocks);
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.Kxc           = arma::field<arma::mat> (n_blocks);
  //param_data.Kxx_inv       = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  //param_data.Kcc = arma::field<arma::mat>(n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  //param_data.Kxx_invchol = arma::field<arma::mat> (n_blocks); // storing the inv choleskys of {parents(w), w} (which is parent for children(w))
  param_data.Rcc_invchol = arma::field<arma::mat> (n_blocks); 
  param_data.ccholprecdiag = arma::field<arma::vec> (n_blocks);
  
  //param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  //param_data.AK_uP_all = arma::field<arma::mat> (n_blocks);
  //param_data.AK_uP_u_all = arma::field<arma::mat> (n_blocks);
  
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
  
  Rcpp::Rcout << "initializing param_data : blocks " << n_blocks << endl;
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    
    if(children(i).n_elem > 0){
      param_data.Sigi_children(i) = arma::zeros(indexing(i).n_elem,
                               indexing(i).n_elem, children(i).n_elem);
      param_data.Smu_children(i) = arma::zeros(indexing(i).n_elem,
                              children(i).n_elem);
      //param_data.Kxx_invchol(u) = arma::zeros(parents_indexing(u).n_elem + indexing(u).n_elem, //Kxx_invchol(last_par).n_rows + Kcc.n_rows, 
       //                      parents_indexing(u).n_elem + indexing(u).n_elem);//Kxx_invchol(last_par).n_cols + Kcc.n_cols);
      param_data.w_cond_prec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      param_data.Rcc_invchol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      //param_data.Sigi_chol(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      
      //if(block_ct_obs(u) > 0){
      //param_data.Kcc(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
      
    }
    param_data.w_cond_mean_K(u) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
    param_data.Kxc(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    param_data.ccholprecdiag(u) = arma::zeros(indexing(u).n_elem);
    
    param_data.Ddiag(u) = arma::zeros(indexing_obs(u).n_elem);
    //}
    //param_data.AK_uP_all(u) = arma::zeros(parents_indexing(u).n_elem, indexing(u).n_elem);
    //param_data.AK_uP_u_all(u) = param_data.AK_uP_all(u) * param_data.w_cond_mean_K(u);
  }
  
  alter_data                = param_data; 
  
  Rcpp::Rcout << "init_model_data: indexing elements: " << indexing.n_elem << endl;
  message("[init_model_data]");
}

arma::field<arma::uvec> SpamTreeMVdevel::cacher(const arma::field<arma::mat>& source_mats){
  arma::field<arma::uvec> targ_caching(n_actual_groups);
  for(int g=0; g<n_actual_groups; g++){
    targ_caching(g) = -1*arma::ones<arma::uvec>(u_by_block_groups(g).n_elem);
    
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u_target = u_by_block_groups(g)(i);
      if(g==n_actual_groups - 1){
        targ_caching(g)(i) = u_target; // last level = observations = avoid caching
        // even though there may be some advantage with gridded obs...
      } else {
        bool foundit = false;
        // search within same resolution
        for(int j=0; j<i; j++){
          int u_prop = u_by_block_groups(g)(j);
          
          bool is_same = matsame( source_mats(u_target), source_mats(u_prop) );
          if(is_same){
            targ_caching(g)(i) = u_prop;
            //Rcpp::Rcout << "found same!" << endl;
            foundit = true;
            break;
          } 
        }
        if(!foundit){
          targ_caching(g)(i) = u_target;
        }
      }
    }
  }
  return targ_caching;
}

void SpamTreeMVdevel::init_caching(){
  message("[init_caching]");
  
  //cc_caching = -1 + arma::zeros<arma::uvec>(n_blocks);
  //cx_caching = -1 + arma::zeros<arma::uvec>(n_blocks);
  //cr_caching = -1 + arma::zeros<arma::uvec>(n_blocks);
  
  // rotations and common sorting
  
  //arma::umat rotgroup; // output from block_rotation_group
  //arma::field<arma::umat> parents_indexing_rotated; // output from parents_indexing_order
  //arma::field<arma::umat> indexing_rotated; // output from indexing_order
  message("[block_rotation_group]");
  rotgroup = block_rotation_group(coords, indexing, indexing_obs, parents, block_names, u_by_block_groups, n_actual_groups);
  message("[parents_indexing_order]");
  parents_indexing_rotated = parents_indexing_order(coords, qvblock_c, rotgroup, indexing, indexing_obs, parents, block_names);
  message("[indexing_order]");
  indexing_rotated = indexing_order(coords, qvblock_c, rotgroup, indexing, parents, block_names);
  
  indexing_obs_rotated = indexing_order(coords, qvblock_c, rotgroup, indexing_obs, parents, block_names);
  message("rotations done.");
  
  //Kcc_perms = arma::field<arma::umat>(n_blocks, 2); // the two permutations
  //arma::field<arma::umat> Kcx_perms(n_blocks, 2);
  //arma::field<arma::umat> Kcr_perms(n_blocks,2);
  // just need some values here
  
  arma::vec cparams = param_data.theta;
  covpars.transform(cparams);
  
  
  // this section initializes the caching and is only run once
  arma::field<arma::mat> Kcc_skeletons(n_blocks);
  arma::field<arma::mat> Kcx_skeletons(n_blocks);
  arma::field<arma::mat> Kcr_skeletons(n_blocks);
  //arma::field<arma::mat> Kxx_skeletons(n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    arma::uvec indexing_u = indexing(u);
    arma::uvec block_rotix;
    arma::uvec at_last_level = arma::find(u_by_block_groups(n_actual_groups-1) == u, 1, "first");
    
    
    if(at_last_level.n_elem > 0){
      indexing_u = indexing_obs(u);
      block_rotix = indexing_obs_rotated(u).col(0);
    } else {
      block_rotix = indexing_rotated(u).col(0);
      
    }
    
    //Rcpp::Rcout << "i: " << i << " u: " << u << endl;
    arma::mat cc_temp = Covariancef(coords, qvblock_c, indexing_u, indexing_u, covpars, true);
    
    //Kcc_perms(u,0) = lexisorter_rows(cc_temp);
    //Kcc_perms(u,1) = lexisorter_rows(cc_temp.t());
    Kcc_skeletons(u) = //Kcc_perms(u,0) * 
      cc_temp;// * Kcc_perms(u,1).t();
    
    if(parents(u).n_elem > 0){
      
      arma::uvec par_rotix = parents_indexing_rotated(u).col(0);
      
      arma::mat cx_temp = Covariancef(coords, qvblock_c, block_rotix, par_rotix,//indexing(u), parents_indexing(u), 
                                      covpars, false);
      
      //Kcx_perms(u,0) = lexisorter_rows(cx_temp);
      //Kcx_perms(u,1) = lexisorter_rows(cx_temp.t());
      Kcx_skeletons(u) = //Kcx_perms(u,0) * 
        cx_temp;// * Kcx_perms(u,1).t();
      
      arma::mat xx_temp = Covariancef(coords, qvblock_c, par_rotix, par_rotix, //parents_indexing(u), parents_indexing(u), 
                                      covpars, true);
      arma::mat cr_temp = cc_temp - cx_temp *// xx_temp * 
        cx_temp.t();
      //Kcr_perms(u,0) = lexisorter_rows(cr_temp);
      //Kcr_perms(u,1) = lexisorter_rows(cr_temp.t());
      Kcr_skeletons(u) = //Kcr_perms(u,0) * 
        cr_temp;// * Kcr_perms(u,1).t();
    }
    
    /*
     if(children(u).n_elem >0){
     // we need to have Kxxi later
     // first child stores info
     // (all first-gen children have the same parent set)
     int firstchild = children(u)(0);
     arma::uvec par_rotix = parents_indexing_rotated(firstchild).col(0);
     Kxx_skeletons(u) = Covariancef(coords, qvblock_c, 
     par_rotix,par_rotix, 
     covpars, true);
     }*/
    
  }
  /*
   cc_caching = arma::field<arma::uvec> (n_actual_groups);
   for(int g=0; g<n_actual_groups; g++){
   cc_caching(g) = -1*arma::ones<arma::uvec>(u_by_block_groups(g).n_elem);
   for(int i=0; i<u_by_block_groups(g).n_elem; i++){
   int u_target = u_by_block_groups(g)(i);
   bool foundit = false;
   // search within same resolution
   for(int j=0; j<i; j++){
   int u_prop = u_by_block_groups(g)(j);
   
   bool is_same = matsame( Kcc_skeletons(u_target), Kcc_skeletons(u_prop) );
   if(is_same){
   cc_caching(g)(i) = u_prop;
   //Rcpp::Rcout << "found same!" << endl;
   foundit = true;
   break;
   } 
   }
   if(!foundit){
   cc_caching(g)(i) = u_target;
   }
   }
   }
   */
  
  message("running cache cc ");
  cc_caching = cacher(Kcc_skeletons);
  
  message("running cache cx ");
  cx_caching = cacher(Kcx_skeletons);
  
  message("running cache cr ");
  cr_caching = cacher(Kcr_skeletons);
  
  //Rcpp::Rcout << "running cache xx " << endl;
  //xxi_caching = cacher(Kxx_skeletons);
  
  cc_caching_uniques = arma::field<arma::uvec>(n_actual_groups);
  cr_caching_uniques = arma::field<arma::uvec>(n_actual_groups);
  cx_caching_uniques = arma::field<arma::uvec>(n_actual_groups);
  //xxi_caching_uniques = arma::field<arma::uvec>(n_actual_groups);
  for(int g=0; g<n_actual_groups; g++){
    cc_caching_uniques(g) = arma::unique(cc_caching(g)); // these are u values
    cr_caching_uniques(g) = arma::unique(cr_caching(g));
    cx_caching_uniques(g) = arma::unique(cx_caching(g));
    //xxi_caching_uniques(g) = arma::unique(xxi_caching(g));
  }
  // -- end initialization --
  
  // 
  for(int g=0; g<n_actual_groups; g++){
    Rcpp::Rcout << u_by_block_groups(g).n_elem << " " << 
      arma::size(cc_caching_uniques(g)) << " " << 
        arma::size(cx_caching_uniques(g)) << " " << 
          arma::size(cr_caching_uniques(g)) << " " << endl;
    //arma::size(xxi_caching_uniques(g)) << endl;
  }
  //
  
  // store some stuff so we dont have to find it all the time later
  store_find_lastpar_main = arma::field<arma::field<arma::uvec> >(n_actual_groups);
  store_find_lastpar_on_cache = arma::field<arma::field<arma::uvec> >(n_actual_groups);
  store_find_u = arma::field<arma::field<arma::uvec> >(n_actual_groups);
  store_found_cx = arma::field<arma::field<arma::uvec> >(n_actual_groups);
  store_found_cc = arma::field<arma::field<arma::uvec> >(n_actual_groups);
  for(int g=0; g<n_actual_groups; g++){
    store_find_lastpar_main(g) = arma::field<arma::uvec>(cr_caching_uniques(g).n_elem);
    store_find_lastpar_on_cache(g) = arma::field<arma::uvec>(cr_caching_uniques(g).n_elem);
    store_find_u(g) = arma::field<arma::uvec>(cr_caching_uniques(g).n_elem);
    store_found_cx(g) = arma::field<arma::uvec>(cr_caching_uniques(g).n_elem);
    store_found_cc(g) = arma::field<arma::uvec>(cr_caching_uniques(g).n_elem);
    for(int i=0; i<cr_caching_uniques(g).n_elem; i++){
      int u = cr_caching_uniques(g)(i);
      if(parents(u).n_elem > 0){
        int lastpar = parents(u)(parents(u).n_elem - 1);
        // lastpar has been cached using another prototype
        store_find_lastpar_main(g)(i) = arma::find(u_by_block_groups(g-1) == lastpar, 1, "first");
        int u_lastpar_on_cache = cr_caching(g-1)(store_find_lastpar_main(g)(i)(0));
        store_find_lastpar_on_cache(g)(i) = arma::find(cr_caching_uniques(g-1) == u_lastpar_on_cache, 1, "first");
        store_find_u(g)(i) = arma::find(u_by_block_groups(g) == u, 1, "first");
        store_found_cx(g)(i) = arma::find(cx_caching_uniques(g) == cx_caching(g)(store_find_u(g)(i)(0)), 1, "first");
        store_found_cc(g)(i) = arma::find(cc_caching_uniques(g) == cc_caching(g)(store_find_u(g)(i)(0)), 1, "first");
      }
    }
  }
  
  /*
   Rcpp::Rcout << "storing cache xx, root " << endl;
   int g=0;
   Kxxi_invchol(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
   for(int i=0; i<cr_caching_uniques(g).n_elem; i++){
   int u = cr_caching_uniques(g)(i);
   arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
   Kxxi_invchol(g)(i) = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
   }*/
  
  // this checks that the cached actually corresponds to what we want
  /*
   for(int g=0; g<n_actual_groups; g++){
   for(int i=0; i<u_by_block_groups(g).n_elem; i++){
   int u = u_by_block_groups(g)(i);
   Rcpp::Rcout << "g: " << g << ", u: " << u << " step 1 " << endl;
   
   arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
   
   if(parents(u).n_elem > 0){
   Rcpp::Rcout << "g: " << g << ", u: " << u << " step 2 " << endl;
   
   bool check2 = false;
   
   arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
   arma::mat Kcx = Covariancef(coords, qvblock_c, indexing(u), parents_indexing(u), covpars, false);
   arma::mat Kxx = Covariancef(coords, qvblock_c, parents_indexing(u), parents_indexing(u), covpars, true);
   
   arma::mat target = Kcx * arma::inv_sympd(Kxx);
   
   
   arma::uvec block_rotix_reverse = indexing_rotated(u).col(1);
   arma::uvec par_rotix_reverse = parents_indexing_rotated(u).col(1);
   
   int uj = cr_caching(g)(i);
   arma::uvec find_uj_in_cached = arma::find(cr_caching_uniques(g) == uj);
   Rcpp::Rcout << "u: " << u << " same as: " << uj << " which is at " << find_uj_in_cached.t() << endl;
   int found_ix = find_uj_in_cached(0);
   arma::mat retrieve = //Kcr_perms(u,0).t() * 
   H_cached(g)(found_ix).submat(block_rotix_reverse, par_rotix_reverse);// * Kcr_perms(u, 1);
   
   check2 = matsame(retrieve, target);
   if(!check2){
   Rcpp::Rcout << "u: " << u << " with " << uj << endl;
   Rcpp::Rcout << arma::accu(abs(retrieve-target)) << endl;
   Rcpp::Rcout << "err H" << endl;
   throw 1;
   }
   
   target = arma::inv_sympd(Kcc - target * Kcx.t());
   
   //uj = cr_caching(g)(i);
   //block_rotix_reverse = indexing_rotated(uj).col(1);
   //par_rotix_reverse = parents_indexing_rotated(uj).col(1);
   //find_uj_in_cached = arma::find(cr_caching_uniques(g) == uj);
   //found_ix = find_uj_in_cached(0);
   arma::mat temp = Rcc_invchol(g)(found_ix).submat(block_rotix_reverse, block_rotix_reverse); // * Kcr_perms(u, 1);
   retrieve = temp.t() * temp;
   
   check2 = matsame(retrieve, target);
   if(!check2){
   Rcpp::Rcout << arma::accu(abs(retrieve-target)) << endl;
   Rcpp::Rcout << "err Rcc" << endl;
   Rcpp::Rcout << retrieve << endl
               << target << endl;
   throw 1;
   }
   }
   }
   }
   
   Rcpp::Rcout << "finished with no error" << endl;
   throw 1;
   */
  message("[init_caching] done.");
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
#ifdef _OPENMP
#pragma omp parallel for 
#endif
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


bool SpamTreeMVdevel::get_loglik_comps_w(SpamTreeMVDataDevel& data){
  // S=standard gibbs (cheapest), P=residual process, R=residual process using recursive functions
  return get_loglik_comps_w_std(data);
}


bool SpamTreeMVdevel::get_loglik_comps_w_std(SpamTreeMVDataDevel& data){
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_w_std] start. ");
  
  
  arma::vec timings = arma::zeros(n_actual_groups);
  arma::vec timings2 = arma::zeros(n_actual_groups);
  
  
  // update theta
  arma::vec cparams = data.theta;
  covpars.transform(cparams);
  
  message("[refresh_cache]");
  
  // this computes the cached matrices
  arma::field<arma::field<arma::mat> > Kcc_cached(n_actual_groups);
  arma::field<arma::field<arma::mat> > Kcx_cached(n_actual_groups);
  arma::field<arma::field<arma::mat> > H_cached(n_actual_groups);
  arma::field<arma::field<arma::mat> > Kxxi_invchol(n_actual_groups);
  arma::field<arma::field<arma::mat> > Kxxi_cached(n_actual_groups);
  arma::field<arma::field<arma::mat> > Rcc_invchol(n_actual_groups);
  arma::field<arma::field<arma::mat> > Rcci_cached(n_actual_groups);
  
  // start at root
  for(int g=0; g<n_actual_groups-1; g++){//exclude nonreference
    start = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "g: " << g << ", storing cache cc " << endl;
    
    Kcc_cached(g) = arma::field<arma::mat>(cc_caching_uniques(g).n_elem);
    for(int i=0; i<cc_caching_uniques(g).n_elem; i++){
      //arma::uvec ixv = arma::find(cc_caching(g) == cc_caching_uniques(g)(i));
      int u = cc_caching_uniques(g)(i);
      arma::uvec block_rotix = indexing_rotated(u).col(0);
      Kcc_cached(g)(i) = Covariancef(coords, qvblock_c, block_rotix, block_rotix, 
                 covpars, true);
    }
    
    end = std::chrono::steady_clock::now();
    timings(g) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //Kcx_cached(g) = arma::field<arma::mat>(cx_caching_uniques(g).n_elem);
    /*
#pragma omp parallel for
     for(int i=0; i<cx_caching_uniques(g).n_elem; i++){
     //arma::uvec ixv = arma::find(cc_caching(g) == cc_caching_uniques(g)(i));
     int u = cx_caching_uniques(g)(i);
     if(parents(u).n_elem > 0){
     arma::uvec block_rotix = indexing_rotated(u).col(0);
     arma::uvec par_rotix = parents_indexing_rotated(u).col(0);
     Kcx_cached(g)(i) = Covariancef(coords, qvblock_c, block_rotix, par_rotix, 
     covpars, false);
     }
     }*/
  }
  
  for(int g=0; g<n_actual_groups; g++){
    start = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "g: " << g << ", storing cache H, Rcc " << endl;
    H_cached(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
    Rcc_invchol(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
    Rcci_cached(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
    Kxxi_invchol(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
    Kxxi_cached(g) = arma::field<arma::mat>(cr_caching_uniques(g).n_elem);
    Kcx_cached(g) = arma::field<arma::mat>(cx_caching_uniques(g).n_elem);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<cr_caching_uniques(g).n_elem; i++){
      int u = cr_caching_uniques(g)(i);
      if(parents(u).n_elem > 0){
        
        if(res_is_ref(g) == 1){
          //Rcpp::Rcout << "number of parents " << parents(u).n_elem << endl;
          //Rcpp::Rcout << "parents of u are " << parents(u).t() << endl;
          
          arma::uvec block_rotix = indexing_rotated(u).col(0);
          arma::uvec par_rotix = parents_indexing_rotated(u).col(0);
          
          // we retrieve Kxxi for this block
          // knowing that it has been ordered according to its first child
          // which may not be this block!
          // block -> parent -> cached copy -> 1st child of cached copy
          // -> restore order -> order like block
          
          // go back one step and find last parent of u
          int lastpar = parents(u)(parents(u).n_elem - 1);
          //Rcpp::Rcout << "last parent thus is " << lastpar << endl;
          // lastpar has been cached using another prototype
          arma::uvec find_lastpar_main = store_find_lastpar_main(g)(i);// arma::find(u_by_block_groups(g-1) == lastpar, 1, "first");
          //Rcpp::Rcout << "which is indexed at " << find_lastpar_main << endl;
          int u_lastpar_on_cache = cr_caching(g-1)(find_lastpar_main(0));
          arma::uvec find_lastpar_on_cache = store_find_lastpar_on_cache(g)(i);//arma::find(cr_caching_uniques(g-1) == u_lastpar_on_cache, 1, "first");
          int found_lastpar = find_lastpar_on_cache(0);
          
          arma::mat Kxxi_of_lastpar;
          if(parents(u).n_elem == 1){ 
            // if the parent is at the root we need to order according
            arma::mat Kxx = Covariancef(coords, qvblock_c, par_rotix, par_rotix, covpars, true);
            Kxxi_of_lastpar = arma::inv(arma::trimatl(arma::chol(Kxx, "lower")));
          } else {
            Kxxi_of_lastpar = Kxxi_invchol(g-1)(found_lastpar);
          }
          // if not at the root, then this has already an order
          // the order guarantees that the matrix is the same
          //int firstchild_of_cached_lastpar = children(u_lastpar_on_cache)(0);
          //Rcpp::Rcout << "number of parents of firstchild " << parents(firstchild_of_cached_lastpar).n_elem << endl;
          //Rcpp::Rcout << children(u_lastpar_on_cache).subvec(0, 5) << endl;
          //arma::uvec childs_block_rotix = indexing_rotated(firstchild_of_cached_lastpar).col(0);
          //arma::uvec childs_par_rotix = parents_indexing_rotated(firstchild_of_cached_lastpar).col(0);
          
          //arma::uvec child1_of_cachelastpar_restore = parents_indexing_rotated(firstchild_of_cached_lastpar).col(1);
          //arma::mat KxxiC_restore = Kxxi_invchol(g)(found_lastpar);//.submat(child1_of_cachelastpar_restore, child1_of_cachelastpar_restore);
          //KxxiC_restore = KxxiC_restore.submat(par_rotix, par_rotix);
          //Rcpp::Rcout << "retrieving Kcx" << endl;// << arma::size(childs_block_rotix) 
          //  << " " << arma::size(childs_par_rotix) 
          //  << " " << arma::size(Kxxi_of_lastpar) << endl;
          //arma::mat cxtemp = Covariancef(coords, qvblock_c, block_rotix, par_rotix, //indexing(u), parents_indexing(u), 
          //                                covpars, false);
          
          
          arma::uvec find_u = store_find_u(g)(i);//arma::find(u_by_block_groups(g) == u, 1, "first");
          
          arma::uvec found_cx = store_found_cx(g)(i); //arma::find(cx_caching_uniques(g) == cx_caching(g)(find_u(0)), 1, "first");
          Kcx_cached(g)(i) = Covariancef(coords, qvblock_c, block_rotix, par_rotix, 
                     covpars, false);
          
          //arma::mat cx_retrieve = Kcx_cached(g)(i);//Kcx_cached(g)(found_cx(0));
          
          //double samecx = arma::accu(abs(cx_retrieve-cxtemp));
          //if(samecx > 1e-4){
          //  Rcpp::Rcout << cx_retrieve << endl
          //              << cxtemp << endl;
          //  throw 1;
          //}
          
          //arma::mat rcc_gotthis = cctemp - Kcx_cached(g)(found_cx(0)) * Kxxi_of_lastpar.t()*Kxxi_of_lastpar * Kcx_cached(g)(found_cx(0)).t();
          
          Kxxi_cached(g)(i) = Kxxi_of_lastpar.t()*Kxxi_of_lastpar;
          H_cached(g)(i) = Kcx_cached(g)(i)* //Kcx_cached(g)(found_cx(0))*
            Kxxi_cached(g)(i);
          
          
          //Rcpp::Rcout << "retrieving Kcc 1" << endl;
          //arma::mat cctemp = Covariancef(coords, qvblock_c, block_rotix, block_rotix, //indexing(u), parents_indexing(u), 
          //                                covpars, true);
          //arma::uvec find_cc = arma::find(u_by_block_groups(g) == u, 1, "first");
          arma::uvec found_cc = store_found_cc(g)(i);//arma::find(cc_caching_uniques(g) == cc_caching(g)(find_u(0)), 1, "first");
          arma::mat cc_retrieve = Kcc_cached(g)(found_cc(0));
          //double samecc = arma::accu(abs(cc_retrieve-cctemp));
          //if(samecc > 1e-4){
          //  Rcpp::Rcout << cc_retrieve << endl
          //              << cctemp << endl;
          //  throw 1;
          //}
          
          
          Rcc_invchol(g)(i) = arma::inv(arma::trimatl(arma::chol(arma::symmatu( // *** check here
            Kcc_cached(g)(found_cc(0)) - H_cached(g)(i) * Kcx_cached(g)(i).t()// Kcx_cached(g)(found_cx(0)).t()
          ), "lower")));
          Rcci_cached(g)(i) = Rcc_invchol(g)(i).t() * Rcc_invchol(g)(i);
          //if(children(u).n_elem > 0){
          Kxxi_invchol(g)(i) = arma::zeros(parents_indexing(u).n_elem + indexing(u).n_elem, 
                       parents_indexing(u).n_elem + indexing(u).n_elem);
          invchol_block_inplace_direct(Kxxi_invchol(g)(i), Kxxi_of_lastpar, 
                                       H_cached(g)(i), Rcc_invchol(g)(i));
          Kxxi_cached(g)(i) = Kxxi_invchol(g)(i).t() * Kxxi_invchol(g)(i);
          //}
        } else {
          //Rcpp::Rcout << "number of parents " << parents(u).n_elem << endl;
          //Rcpp::Rcout << "parents of u are " << parents(u).t() << endl;
          
          arma::uvec block_rotix = indexing_obs_rotated(u).col(0);
          arma::uvec par_rotix = parents_indexing_rotated(u).col(0);
          
          // we retrieve Kxxi for this block
          // knowing that it has been ordered according to its first child
          // which may not be this block!
          // block -> parent -> cached copy -> 1st child of cached copy
          // -> restore order -> order like block
          
          // go back one step and find last parent of u
          int lastpar = parents(u)(parents(u).n_elem - 1);
          //Rcpp::Rcout << "last parent thus is " << lastpar << endl;
          // lastpar has been cached using another prototype
          arma::uvec find_lastpar_main = store_find_lastpar_main(g)(i);// arma::find(u_by_block_groups(g-1) == lastpar, 1, "first");
          //Rcpp::Rcout << "which is indexed at " << find_lastpar_main << endl;
          int u_lastpar_on_cache = cr_caching(g-1)(find_lastpar_main(0));
          arma::uvec find_lastpar_on_cache = store_find_lastpar_on_cache(g)(i);//arma::find(cr_caching_uniques(g-1) == u_lastpar_on_cache, 1, "first");
          int found_lastpar = find_lastpar_on_cache(0);
          
          //arma::mat Kxxi_of_lastpar = Kxxi_invchol(g-1)(found_lastpar);
          
          // if not at the root, then this has already an order
          // the order guarantees that the matrix is the same
          //int firstchild_of_cached_lastpar = children(u_lastpar_on_cache)(0);
          //Rcpp::Rcout << "number of parents of firstchild " << parents(firstchild_of_cached_lastpar).n_elem << endl;
          //Rcpp::Rcout << children(u_lastpar_on_cache).subvec(0, 5) << endl;
          //arma::uvec childs_block_rotix = indexing_rotated(firstchild_of_cached_lastpar).col(0);
          //arma::uvec childs_par_rotix = parents_indexing_rotated(firstchild_of_cached_lastpar).col(0);
          
          //arma::uvec child1_of_cachelastpar_restore = parents_indexing_rotated(firstchild_of_cached_lastpar).col(1);
          //arma::mat KxxiC_restore = Kxxi_invchol(g)(found_lastpar);//.submat(child1_of_cachelastpar_restore, child1_of_cachelastpar_restore);
          //KxxiC_restore = KxxiC_restore.submat(par_rotix, par_rotix);
          //Rcpp::Rcout << "retrieving Kcx" << endl;// << arma::size(childs_block_rotix) 
          //  << " " << arma::size(childs_par_rotix) 
          //  << " " << arma::size(Kxxi_of_lastpar) << endl;
          //arma::mat cxtemp = Covariancef(coords, qvblock_c, block_rotix, par_rotix, //indexing(u), parents_indexing(u), 
          //                                covpars, false);
          
          
          arma::uvec find_u = store_find_u(g)(i);//arma::find(u_by_block_groups(g) == u, 1, "first");
          
          arma::uvec found_cx = store_found_cx(g)(i); //arma::find(cx_caching_uniques(g) == cx_caching(g)(find_u(0)), 1, "first");
          Kcx_cached(g)(i) = Covariancef(coords, qvblock_c, block_rotix, par_rotix, 
                     covpars, false);
          
          //arma::mat cx_retrieve = Kcx_cached(g)(i);//Kcx_cached(g)(found_cx(0));
          
          //double samecx = arma::accu(abs(cx_retrieve-cxtemp));
          //if(samecx > 1e-4){
          //  Rcpp::Rcout << cx_retrieve << endl
          //              << cxtemp << endl;
          //  throw 1;
          //}
          
          //arma::mat rcc_gotthis = cctemp - Kcx_cached(g)(found_cx(0)) * Kxxi_of_lastpar.t()*Kxxi_of_lastpar * Kcx_cached(g)(found_cx(0)).t();
          
          
          H_cached(g)(i) = Kcx_cached(g)(i)* //Kcx_cached(g)(found_cx(0))*
            Kxxi_cached(g-1)(found_lastpar);//Kxxi_of_lastpar.t()*Kxxi_of_lastpar;
          
        }
        
      } else {
        arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u),
                                    covpars, true);
        Rcc_invchol(g)(i) = arma::inv(arma::trimatl(arma::chol(arma::symmatu(Kcc, "lower"))));
        Rcci_cached(g)(i) = Rcc_invchol(g)(i).t() * Rcc_invchol(g)(i);
      } 
    }
    
    end = std::chrono::steady_clock::now();
    timings(g) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  message("[refresh_cache] done.");
  
  // cycle through the resolutions starting from the bottom
  
  int errtype = -1;
  
  arma::vec ll_y = arma::zeros(y.n_rows);
  //Rcpp::Rcout << "about to enter for loop " << endl;
  //Rcpp::Rcout << u_by_block_groups.n_elem << endl;
  for(int g=0; g<n_actual_groups; g++){
    //Rcpp::Rcout << g << endl;
    start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      
      if(parents(u).n_elem == 0){
        arma::vec w_x = w.rows(indexing(u));
        arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
        //data.Kcc(u) = Kcc;
        try{
          arma::mat Kxx_invchol = arma::inv(arma::trimatl(arma::chol(Kcc, "lower")));
          //data.Kxx_inv(u) = Kxx_invchol.t() * Kxx_invchol;
          data.Rcc_invchol(u) = Kxx_invchol; 
          data.w_cond_prec(u) = Kxx_invchol.t() * Kxx_invchol;
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
        if(res_is_ref(g) == 1){
          
          arma::vec w_x = w.rows(indexing(u));
          
          arma::uvec block_rotix_reverse = indexing_rotated(u).col(1);
          arma::uvec par_rotix_reverse = parents_indexing_rotated(u).col(1);
          
          int uj = cr_caching(g)(i);
          arma::uvec find_uj_in_cached = arma::find(cr_caching_uniques(g) == uj, 1, "first");
          //Rcpp::Rcout << "u: " << u << " same as: " << uj << " which is at " << find_uj_in_cached.t() << endl;
          int found_ix = find_uj_in_cached(0);
          
          data.w_cond_mean_K(u) = H_cached(g)(found_ix).submat(block_rotix_reverse, par_rotix_reverse);
          
          
          //int last_par = parents(u)(parents(u).n_elem - 1);
          //arma::mat LAi = Kxx_invchol(last_par);
          
          //Covariancef_inplace(data.Kxc(u), coords, qvblock_c, parents_indexing(u), indexing(u), covpars, false);
          arma::vec w_pars = w.rows(parents_indexing(u));
          
          //data.w_cond_mean_K(u) = data.Kxc(u).t() * data.Kxx_inv(last_par);
          w_x -= data.w_cond_mean_K(u) * w_pars;
          //Rcpp::Rcout << "done for branch " << endl;
          
          //arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);
          //data.Kcc(u) = Kcc;
          //try {
          //Rcpp::Rcout << "branch 1" << endl;
          //Rcpp::Rcout << arma::size(Kcc) << " " << arma::size(data.w_cond_mean_K(u)) <<
          // " " << arma::size(data.Kxc(u)) << endl;
          
          data.Rcc_invchol(u) = Rcc_invchol(g)(found_ix).submat(block_rotix_reverse, block_rotix_reverse); // * Kcr_perms(u, 1);
          //retrieve = temp.t() * temp;
          
          
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
          
          //invchol_block_inplace_direct(data.Kxx_invchol(u), data.Kxx_invchol(last_par), 
          //                             data.w_cond_mean_K(u), data.Rcc_invchol(u));
          
          //Rcpp::Rcout << "aand ";
          //data.Kxx_inv(u) = data.Kxx_invchol(u).t() * data.Kxx_invchol(u);
          //Rcpp::Rcout << "done with Kxx_inv" << endl;
          //}
          //Rcpp::Rcout << "branch 5" << endl;
          
          data.has_updated(u) = 1;
          //}
          
          //Rcpp::Rcout << "other comps ";
          data.w_cond_prec(u) = Rcci_cached(g)(found_ix).submat(block_rotix_reverse, block_rotix_reverse);//data.Rcc_invchol(u).t() * data.Rcc_invchol(u);
          
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          data.ccholprecdiag(u) = data.Rcc_invchol(u).diag();
          data.logdetCi_comps(u) = arma::accu(log(data.ccholprecdiag(u)));
          data.loglik_w_comps(u) = (indexing(u).n_elem+.0) 
            * hl2pi -.5 * data.wcore(u);
        } else {
          //Rcpp::Rcout << "no ref 1" << endl;
          
          //arma::vec w_x = w.rows(indexing_obs(u));
          
          // this is a non-reference THIN set. *all* locations conditionally independent here given parents.
          // all observed locations are in these sets.
          // we integrate out the latent effect and use the output likelihood instead.
          
          arma::uvec block_rotix_reverse = indexing_obs_rotated(u).col(1);
          arma::uvec par_rotix_reverse = parents_indexing_rotated(u).col(1);
          
          int uj = cr_caching(g)(i);
          arma::uvec find_uj_in_cached = arma::find(cr_caching_uniques(g) == uj, 1, "first");
          //Rcpp::Rcout << "u: " << u << " same as: " << uj << " which is at " << find_uj_in_cached.t() << endl;
          int found_ix = find_uj_in_cached(0);
          
          data.w_cond_mean_K(u) = H_cached(g)(found_ix).submat(block_rotix_reverse, par_rotix_reverse);
          
          //int last_par = parents(u)(parents(u).n_elem - 1);
          //arma::mat LAi = Kxx_invchol(last_par);
          
          //Covariancef_inplace(data.Kxc(u), coords, qvblock_c, parents_indexing(u), indexing(u), covpars, false);
          //arma::vec w_pars = w.rows(parents_indexing(u));
          
          //data.w_cond_mean_K(u) = data.Kxc(u).t() * data.Kxx_inv(last_par);
          //w_x -= data.w_cond_mean_K(u) * w_pars;
          //Rcpp::Rcout << "done for branch " << endl;
          
          uj = cx_caching(g)(i);
          find_uj_in_cached = arma::find(cx_caching_uniques(g) == uj, 1, "first");
          //Rcpp::Rcout << "u: " << u << " same as: " << uj << " which is at " << find_uj_in_cached.t() << endl;
          found_ix = find_uj_in_cached(0);
          
          
          data.Kxc(u) = arma::trans(Kcx_cached(g)(found_ix).submat(block_rotix_reverse, par_rotix_reverse));
          
          //data.Kxc(u) = Covariancef(coords, qvblock_c, parents_indexing(u), indexing(u), covpars, false);
          
          //start = std::chrono::steady_clock::now();
          //Rcpp::Rcout << arma::size(data.w_cond_mean_K(u)) << " " << arma::size(data.Kxc(u)) << endl;
          data.wcore(u) = 0;
          arma::uvec ones = arma::ones<arma::uvec>(1);
          for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
            if(na_1_blocks(u)(ix) == 1){ 
              // compute likelihood contribution of this observation
              arma::uvec uix = ones * indexing_obs(u)(ix);
              //arma::uvec ix_q = arma::regspace<arma::uvec>(ix*q, ix*q+q-1);
              int first_ix = ix;
              int last_ix = ix;
              //start = std::chrono::steady_clock::now();
              arma::mat Kcc = Covariancef(coords, qvblock_c, uix, uix, covpars, true);
              //end = std::chrono::steady_clock::now();
              //timings(5) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              
              //start = std::chrono::steady_clock::now();
              param_data.Ddiag(u)(ix) = arma::conv_to<double>::from(
                data.w_cond_mean_K(u).rows(first_ix, last_ix) * data.Kxc(u).cols(first_ix, last_ix) );
              double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv_long(indexing_obs(u)(ix));
              double ytilde =  
                arma::conv_to<double>::from(y(indexing_obs(u)(ix)) - XB.row(indexing_obs(u)(ix)));// - Zw.row(indexing_obs(u)(i)));
              ll_y(indexing_obs(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
              
            }
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    timings(g) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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
  
  arma::vec cparams = param_data.theta;
  covpars.transform(cparams);
  
  arma::vec timings = arma::zeros(8);
  
  int errtype = -1;
  //Rcpp::Rcout << res_is_ref << endl;
  //Rcpp::Rcout << "n groups " << n_actual_groups << endl;
  
  for(int g=n_actual_groups-1; g>=0; g--){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //Rcpp::Rcout << g << " " << u << " " << indexing(u).n_elem << endl;
      
      //Rcpp::Rcout << res_is_ref.t() << endl;
      //arma::mat Dtaui = arma::diagmat(tausq_inv_long.rows(indexing(u)));
      //arma::sp_mat Ztausq = spmat_by_diagmat(Zblock(u).t(), tausq_inv_long.rows(indexing(u)));
      if(res_is_ref(g) == 0){
        //if(blocks_predicting(i)==0){
        //w.rows(indexing_obs(u)) = param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        //}
      } else {
        //start = std::chrono::steady_clock::now();
        // reference set. full dependence
        arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
        // Sigi_p
        
        arma::mat Sigi_tot = param_data.w_cond_prec(u);
        arma::mat AK_uP_all;
        if(parents(u).n_elem > 0){
          //param_data.AK_uP_all(u) = 
          AK_uP_all = param_data.w_cond_mean_K(u).t() * param_data.w_cond_prec(u); 
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
        
        arma::sp_mat tsqD = spmat_by_diagmat(Ib(uc), tausq_inv_long(indexing_obs(uc)));
        
        Sigi_tot += Hthis.t() * tsqD * Hthis;
        Smu_tot += Hthis.t() * tsqD * (y.rows(indexing_obs(uc)) - XB.rows(indexing_obs(uc)) - Hother * w_opars_of_child);
        
        
        // ------------------------- 
        
        
        start = std::chrono::steady_clock::now();
        if(parents(u).n_elem>0){
          Smu_tot += AK_uP_all.t() * w.rows(parents_indexing(u));//param_data.w_cond_mean(u);
          // for updating the parents that have this block as child
        }
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        
        //if(children(u).n_elem > 0){
        Smu_tot += arma::sum(param_data.Smu_children(u), 1);
        //}
        
        arma::mat Sigi_chol;
        try {
          Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        } catch(...){
          errtype = 10;
          Sigi_chol = arma::eye(arma::size(Sigi_tot));
        }
        //end = std::chrono::steady_clock::now();
        //timings(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        
        arma::vec rnvec = bigrnorm.rows(indexing(u));//arma::randn(indexing(u).n_elem);//arma::vectorise(rand_norm_mat.rows(indexing(u)));
        
        w.rows(indexing(u)) = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        // message the parents
        if(parents(u).n_elem > 0){
          //start = std::chrono::steady_clock::now();
          
          arma::mat AK_uP_u_all = AK_uP_all * param_data.w_cond_mean_K(u);
          
          
          arma::vec w_par = w.rows(parents_indexing(u));
          
          for(int p=0; p<parents(u).n_elem; p++){
            // up is the parent
            int up = parents(u)(p);
            
            int c_ix = this_is_jth_child(u)(p);
            
            if(need_update){
              param_data.Sigi_children(up).slice(c_ix) = 
                AK_uP_u_all.submat(u_is_which_col_f(up)(c_ix)(0), 
                                       u_is_which_col_f(up)(c_ix)(0));
            }
            
            //start = std::chrono::steady_clock::now();
            param_data.Smu_children(up).col(c_ix) = 
              AK_uP_all.rows(u_is_which_col_f(up)(c_ix)(0)) * w.rows(indexing(u)) -  
              AK_uP_u_all.submat(u_is_which_col_f(up)(c_ix)(0), 
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
  
  arma::vec cparams = param_data.theta;
  covpars.transform(cparams);
  
  //arma::vec timings = arma::zeros(4);
  
  // cycle through the resolutions starting from the bottom
  //arma::uvec predicting_blocks = arma::find(block_ct_obs == 0);
  
  int g = n_actual_groups-1;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<u_by_block_groups(g).n_elem; i++){
    int u = u_by_block_groups(g)(i);
    
    //Rcpp::Rcout << res_is_ref.t() << endl;
    //arma::mat Dtaui = arma::diagmat(tausq_inv_long.rows(indexing(u)));
    //arma::sp_mat Ztausq = spmat_by_diagmat(Zblock(u).t(), tausq_inv_long.rows(indexing(u)));
    arma::vec w_par = w.rows(parents_indexing(u));
    arma::uvec ones = arma::ones<arma::uvec>(1);
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      arma::uvec uix = ones * indexing_obs(u)(ix);
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
        Rcpp::Rcout << Ktemp << endl;
        Ktemp(0,0) = 0;
        Rchol = arma::zeros(1,1);
      }
      //arma::vec rnvec = arma::randn(1);
      
      w.row(indexing_obs(u)(ix)) = param_data.w_cond_mean_K(u).rows(first_ix, last_ix) * w_par;// + Rchol * bigrnorm(indexing(u)(ix));
      
    }
    
    // meaning this block must be predicted
    //start = std::chrono::steady_clock::now();
    //start = std::chrono::steady_clock::now();
    
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
    
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv_long(indexing_obs(u)(ix));
        //Rcpp::Rcout << "llcontrib 2" << endl;
        double ytilde =  
          arma::conv_to<double>::from(y(indexing_obs(u)(ix)) - XB.row(indexing_obs(u)(ix)));// - Zw.row(indexing_obs(u)(i)));
        ll_y(indexing_obs(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
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