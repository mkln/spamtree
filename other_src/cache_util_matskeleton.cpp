#include <RcppArmadillo.h>
#include "cache_util_matskeleton.h"

using namespace std;

arma::umat lexisorter_rows(const arma::mat& x, double prec){
  arma::mat xr = arma::round(((x-x.min())/(x.max()-x.min()))/prec) * prec; // normalize to 0-1 and round to precision
  arma::vec rowsums = arma::sum(xr, 1);
  arma::vec sqrsums = arma::sum(xr % xr, 1);
  arma::uvec x_rowsortix = sortixnotie(rowsums, rowsums, sqrsums);
  arma::umat permrows = arma::zeros<arma::umat>(x.n_rows, x.n_rows);
  
  for(int i=0; i<permrows.n_rows; i++){
    //Rcpp::Rcout << "i: " << i << endl;
    //Rcpp::Rcout << arma::size(x_rowsortix) << " " << x_rowsortix.max() << endl;
    //Rcpp::Rcout <<  x_rowsortix(i) << endl;
    permrows(x_rowsortix(i),i) = 1;
  }
  return permrows.t();
}


arma::mat matskeleton(const arma::mat& x){
  if((x.n_rows + x.n_cols)*x.n_rows*x.n_cols > 0){
    arma::mat xdd = x;
    arma::umat permrows = lexisorter_rows(xdd);
    arma::umat permcols = lexisorter_rows(xdd.t());
    return permrows * x * permcols.t();
  }
  return arma::zeros(0,0);
}

arma::mat matskeleton(const arma::mat& x, const arma::umat& permrows, const arma::umat& permcols){
  if((x.n_rows + x.n_cols)*x.n_rows*x.n_cols > 0){
    return permrows * x * permcols.t();
  }
  return arma::zeros(0,0);
}

arma::mat restore_from_skeleton(const arma::mat& skeleton, const arma::umat& permrows, const arma::umat& permcols){
  return permrows.t() * skeleton * permcols;
}

bool matsame(const arma::mat& x, const arma::mat& y, double tol){
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

//[[Rcpp::export]]
arma::mat arma_matsort(const arma::mat& x, const arma::urowvec& ascending) {
  // Order the elements of x by sorting starting from the first column
  // we order by first column and resolve ties with other columns
  return x.rows(mat_sortix(x, ascending));
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


arma::umat block_rotation_group(const arma::mat& coords, const arma::field<arma::uvec>& indexing,
                                const arma::field<arma::uvec>& parents, const arma::vec& block_names){
  // based on the position of the block in the domain,
  // this function determines which sorting of the coordinate axes
  // makes the resulting covariance matrices be equal
  
  arma::umat rot_groups = arma::zeros<arma::umat>(block_names.n_elem, 3); // 2d space + mv
  
  for(int i=0; i<block_names.n_elem; i++){
    int u = block_names(i) - 1;
    //Rcpp::Rcout << "i " << i << " u " << u << endl;
    //Rcpp::Rcout << arma::size(coords) << endl;
    if((indexing(u).n_elem > 0) & (parents(u).n_elem > 0)){
      int last_par = parents(u)(parents(u).n_elem - 1);
      arma::mat coords_parent;
      arma::rowvec parent_centroid;
      
      arma::mat coords_block;
      arma::rowvec block_centroid;
      
      try {
        coords_parent = coords.rows(indexing(last_par));
        parent_centroid = arma::sum(coords_parent, 0) / (.0+coords_parent.n_elem);
        
        coords_block = coords.rows(indexing(u));
        block_centroid = arma::sum(coords_block, 0) / (.0+coords_block.n_elem);
        
      } catch (...) {
        Rcpp::Rcout << "got error " << endl;
        Rcpp::Rcout << indexing(last_par).t() << endl
                    << indexing(u).t() << endl;
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

