prebuild_precision <- function(coords, mv_id, ai1, ai2, phi_i, thetamv, Dmat,
                               cell_size=25, K=rep(2, ncol(coords)),
                               limited_tree = F,
                               num_threads = 4,
                               verbose=F, debug=F){
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  cat("Sampling a spatial multiresolution tree.\n")
  
  dd <- ncol(coords)
  na_which <- rep(1, nrow(coords))
  axis_cell_size <- rep(cell_size^(1/dd), dd) %>% round()
  
  cix <- coords %>% as.data.frame() %>% mutate(ix = 1:n())
  
  #load("debug.RData")
  cat("Partitioning into resolution layers.\n")
  ptime <- system.time(
    mgp_tree <- spamtree:::make_tree(cix, na_which, mv_id, axis_cell_size, K, Inf, F,)
  )
  
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking
  thresholds <- mgp_tree$thresholds
  res_is_ref <- mgp_tree$res_is_ref
  
  original_order <- order(coords_blocking$ix)
  coords_blocking %<>% arrange(res, block, ix)
  cx <- coords_blocking %>% dplyr::select(contains("Var")) %>% as.matrix()
  mv_id <- coords_blocking$sort_mv_id
  
  blocking <- coords_blocking$block
  block_info <- coords_blocking %>% mutate(color = res) %>%
    select(block, color) %>% unique()
  non_empty_blocks <- blocking#rep(1, nrow(parchi_map))
  cat("Building graph.\n")
  if(limited_tree){
    gtime <- system.time({
      parents_children <- spamtree:::make_edges_limited(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
    })
  } else {
    gtime <- system.time({
      parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
    })
  }
  
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  return(list(
    parchi_map=parchi_map,
    cx=cx,
    mv_id= mv_id,
    coords_blocking=coords_blocking,
    blocking=blocking,
    parents=parents,
    children=children,
    block_names=block_names,
    block_groups=block_groups,
    indexing=indexing,
    original_order=original_order,
    ai1=ai1, ai2=ai2, phi_i=phi_i, thetamv=thetamv, Dmat=Dmat
  ))
}

#' @export
precision_matrix <- function(coords, mv_id, ai1, ai2, phi_i, thetamv, Dmat,
                             cell_size=25, K=rep(2, ncol(coords)),
                             limited_tree = F,
                             num_threads = 4,
                             verbose=F, debug=F){

  prebuilt <- spamtree:::prebuild_precision(coords, mv_id, ai1, ai2, phi_i, thetamv, Dmat,
                                            cell_size, K, limited_tree,
                                            num_threads, verbose, debug)

  Ci_res <- with(prebuilt, spamtree:::spamtree_Cinv(cx, mv_id, blocking, parents, block_names,
                      indexing, ai1, ai2, phi_i, thetamv, Dmat, num_threads,
                      verbose, debug))
  
  Ci <- Ci_res$Ci
  IminusH <- Ci_res$IminusH
  Di <- Ci_res$Di
  HH <- Ci_res$H
  
  #coords_blocking %<>% arrange(res, block, ix)
  csx <- prebuilt$coords_blocking$ix
  
  return(list(
    original_order=prebuilt$original_order,
    csx=csx,
    precision_matrix = Ci,#[original_order, original_order][csx,csx],
    IminusH = IminusH,#[original_order, original_order][csx,csx],
    H = HH,#[original_order, original_order][csx,csx],
    Di = Di,#[original_order, original_order][csx,csx],
    coords_info = prebuilt$coords_blocking))
}