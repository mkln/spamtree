rspamtree <- function(coords, theta,
                     cell_size=25, K=rep(2, ncol(coords)),
                     num_threads = 4,
                     verbose=F, debug=F){
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  cat("Sampling a spatial multiresolution tree.\n")
  
  dd <- ncol(coords)
  na_which <- rep(1, nrow(coords))
  axis_cell_size <- rep(cell_size^(1/dd), dd)
  
  cix <- coords %>% as.data.frame() %>% mutate(ix = 1:n())
  
  
  cat("Partitioning into resolution layers.\n")
  ptime <- system.time(
    mgp_tree <- make_tree(cix, na_which, axis_cell_size, K)
  )
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking
  thresholds <- mgp_tree$thresholds
  
  blocking <- coords_blocking$block
  block_info <- coords_blocking %>% mutate(color = res) %>%
    select(block, color) %>% unique()
  non_empty_blocks <- rep(1, nrow(parchi_map))
  cat("Building graph.\n")
  gtime <- system.time({
    parents_children <- multires_graph(parchi_map %>% as.matrix(), non_empty_blocks)
  })
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  Dmat <- matrix(0)
  
  cx <- coords_blocking %>% dplyr::select(contains("Var")) %>% as.matrix()
  sampled <- spamtree_sample(cx, blocking,
    parents, block_names, indexing, 
    theta, Dmat,
    num_threads,
    verbose, debug)
  
  return(sampled[order(coords_blocking$ix)])
}