make_tree <- function(coords, na_which, sort_mv_id, 
                      axis_cell_size=c(5,5), K=c(2,2), 
                      max_res=Inf, last_not_reference=T,
                      cherrypick_same_margin = T,
                      cherrypick_group_locations = T){
  #Rcpp::sourceCpp("src/multires_utils.cpp")
 
  #coords <- rnorm(1000, 0, 2) %>% matrix(ncol=2) %>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  #colnames(coords) <- c("Var1", "Var2")
  
  coords_allres <- coords %>% as.data.frame() %>% cbind(sort_mv_id)
  coords_missing <- coords_allres[is.na(na_which),] 
  coords_availab <- coords_allres[!is.na(na_which),] %>% as.data.frame()
  
  nr <- nrow(coords_availab)
  dd <- ncol(coords_availab) - 2
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  
  coords_refset <- cx %>% mutate(block=NA, res=NA) %>% #, part=NA) %>%
    dplyr::filter(Var1 < minx)
  
  cell_size <- prod(axis_cell_size)
  n_u_cells <- nr/cell_size
  
  #LL <- floor( log(n_u_cells)/log(4)  )
  thresholds_list <- list()
  max_block_number <- 0
  res <- 1
  
  timings <- rep(0, 3)
  
  cat("Branching the tree")
  
  first_steps <- system.time({
    while((res <= max_res) & (nrow(cx) > 0)){
      
      cat(" ", res)
      # first, tessellate to find locations that are spread out. select one for each cell.
      # second, with the resulting subsample then actually split into blocks
      
      # generate threshold to make small cells from which we will sample 1 location
      # fixed threshold regularly on each axis
      #thresholds_knots <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=axis_cell_size[i] * 2^(res-1) + 1))
      # based on quantiles -- hopefully more uniform partitioning
      
      timings[1] <- timings[1] + 
        system.time({
          
          thresholds_knots <- 1:dd %>% lapply(function(i) 
            spamtree:::kthresholds(coords_availab[,i], axis_cell_size[i] * K[i]^(res-1)))
          
          grid_size <- thresholds_knots %>% lapply(function(x) length(x)+1) %>% unlist() %>% prod()
          
          if(grid_size < nrow(cx)){
            # if grid is too thin considering available coordinates, skip this and partition directly
            coords_knots <- cx[,-4] %>% as.matrix() %>%
              spamtree:::axis_parallel(thresholds_knots, 4) %>%
              #mutate(block = max_block_number + block) %>%
              group_by(block) %>% summarise(ix = ix[sample(1:n(), 1)], .groups="keep")
                                                 # ix[round(n()/2 +.5)])
            
            if(cherrypick_group_locations){
              coords_knots <- coords_knots %>% as.data.frame() %>% 
                left_join(cx, by=c("ix"="ix")) %>% dplyr::select(-ix, -sort_mv_id) %>%
                left_join(cx, by=c("Var1"="Var1", "Var2"="Var2")) %>%
                dplyr::select(contains("Var"), ix)
            } else {
              coords_knots <- coords_knots %>% as.data.frame() %>% 
                left_join(cx, by=c("ix"="ix")) %>% 
                dplyr::select(contains("Var"), ix)
            }
            
          } else {
            coords_knots <- cx 
          }
          
        })["elapsed"]
      
      #coords_knots$ix %>% intersect(coords_refset$ix)
      
      timings[2] <- timings[2] + 
        system.time({
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) spamtree:::kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res]] <- thresholds_res
          
          coords_res <- coords_knots[,-4] %>% as.matrix() %>% 
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            mutate(block = max_block_number + block,
                   res = res) %>%
            dplyr::select(contains("Var"), ix, block, res)#, part)
          coords_res %<>% left_join(coords_allres %>% dplyr::select("ix", "sort_mv_id"), by=c("ix"="ix"))
          max_block_number <- coords_res %>% pull(block) %>% max()
          
        })["elapsed"]
      
      timings[3] <- timings[3] + 
        system.time({
          # move selected locations to the reference set
          cx %<>% dplyr::filter(!(.$ix %in% coords_res$ix))
          coords_refset %<>% bind_rows(coords_res)
          
          # keep track of overall tessellation
          blockname = sprintf("block_res%02d", res)
          coords_keeptrack <- coords_availab %>% 
            dplyr::select(contains("Var"), ix) %>% 
            as.matrix() %>%
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            rename(!!sym(blockname) := block) %>%
            dplyr::select(ix, !!sym(blockname))
          
          coords_availab %<>% left_join(coords_keeptrack, by=c("ix"="ix"))
          
        })["elapsed"]
      res <- res + 1
      
    }
  })
  
  cat(".\n")
  
  res_is_ref <- rep(1, res-1)
  if(last_not_reference & (res < max_res)){
    res_is_ref[ length(res_is_ref) ] <- 0
  }
  
  cat("Finalizing with leaves.\n")
  #print(timings)
  #cat("Resolution part 1:", as.numeric(first_steps["elapsed"]), "\n")
  
  last_steps <- system.time({
    # now that we have the multiresolution partitioning, 
    # we create a table with the parent-child relationships
    
    relate <- coords_availab %>% dplyr::select(contains("Var"), ix, contains("block_res")) %>%
      dplyr::inner_join(coords_refset %>% dplyr::select(ix, res, block),#, part),
                        by=c("ix"="ix"))
    max_block_number <- 0
    res <- coords_refset$res %>% max()
    for(i in 1:res){
      blockname <- sprintf("block_res%02d", i)
      relate %<>% mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
      max_block_number <- max(relate %>% pull(!!sym(blockname)))
    }
    
    parchi_original <- relate[,sprintf("block_res%02d", 1:res)] %>% 
      dplyr::select(contains("block_res")) %>% unique()
    translator <- relate %>% 
      mutate(block_unnorm = relate %>% 
               apply(1, function(rr) rr[sprintf("block_res%02d", rr["res"])])) %>%
      dplyr::select(block_unnorm, block) %>% 
      unique() %>% 
      right_join(data.frame(block_unnorm = 1:max(.$block_unnorm)), by=c("block_unnorm"="block_unnorm"))
    translator[is.na(translator)] <- 0
    
    parchi_map <- spamtree:::number_revalue(parchi_original %>% as.matrix(), translator$block_unnorm, translator$block)
    colnames(parchi_map) <- colnames(parchi_original)
    parchi_map[parchi_map == 0] <- NA
    
    parchi_map <- parchi_map %>% as.data.frame() %>% 
      arrange(!!!syms(sprintf("block_res%02d", 1:(ncol(parchi_map)-1))))
    
    max_block_number <- coords_refset %>% pull(block) %>% max()
    
    # now manage leftouts
    if(nrow(cx) > 0){
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      column_select <- colnames(cx)[grepl("Var", colnames(cx))]
      cx_leftover <- cx[,c(column_select, "ix", "sort_mv_id")]
      max_res <- coords_refset$res %>% max()
      coords_refset_sub <- coords_refset %>% filter(res %in% c( max_res ))
      
      # CHERRY PICKING -- 
      # find same-margin nearest neighbor and assign to that block
      leftover_vars <- cx_leftover$sort_mv_id %>% unique()
      cx_leftover %<>% mutate(block=NA)
      
      if(!cherrypick_same_margin){
        #for(vv in leftover_vars){
        # row_of_this <- which(cx_leftover$sort_mv_id == vv)
        
        target_coords_same_margin <- coords_refset_sub %>% 
          #  filter(sort_mv_id == vv) %>% 
          dplyr::select(contains("Var"), ix, block)
        
        cx_leftover_same_margin <- cx_leftover #%>%
        #filter(sort_mv_id == vv)
        
        nn_of_leftover_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                    cx_leftover_same_margin[,1:dd], k=1, algorithm="kd_tree")
        
        block_of_leftover_same_margin <- target_coords_same_margin[nn_of_leftover_same_margin$nn.index, "block"]
        
        cx_leftover[#row_of_this
          , "block"] <- block_of_leftover_same_margin
        #}
        
      } else {
        for(vv in leftover_vars){
         row_of_this <- which(cx_leftover$sort_mv_id == vv)
        
        target_coords_same_margin <- coords_refset_sub %>% 
            filter(sort_mv_id == vv) %>% 
          dplyr::select(contains("Var"), ix, block)
        
        cx_leftover_same_margin <- cx_leftover %>%
        filter(sort_mv_id == vv)
        
        nn_of_leftover_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                    cx_leftover_same_margin[,1:dd], k=1, algorithm="kd_tree")
        
        block_of_leftover_same_margin <- target_coords_same_margin[nn_of_leftover_same_margin$nn.index, "block"]
        
        cx_leftover[row_of_this
          , "block"] <- block_of_leftover_same_margin
        }
        
      }
      
      block_of_leftover <- cx_leftover$block
      
      #target_coords <- coords_refset_sub %>% select(contains("Var"))
      #nn_of_leftover <- FNN::get.knnx(target_coords, cx_leftover, k=1, algorithm="kd_tree")
      #block_of_leftover <- coords_refset_sub[nn_of_leftover$nn.index, "block"]
      res_of_leftover <- max_res + 1
      
      blockname = sprintf("block_res%02d", res_of_leftover)
      parent_blockname = sprintf("block_res%02d", res_of_leftover-1)
      
      coords_res_leftover <- cx %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_leftover,
               block = as.numeric(factor(block_of_leftover)) + max_block_number,
               res = res_of_leftover)
      
      coords_all <- bind_rows(coords_refset, coords_res_leftover %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_leftover %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_leftover <- coords_res_leftover %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_leftover)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    } else {
      coords_all <- coords_refset
    }
    
    # now manage missing
    
    if(nrow(coords_missing) > 0){
      max_block_number <- coords_all %>% pull(block) %>% max()
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      column_select <- colnames(cx)[grepl("Var", colnames(coords_missing))]
      cx_missing <- coords_missing[,c(column_select, "ix", "sort_mv_id")]
      coords_refset_sub <- coords_refset %>% filter(res %in% c(max(res)))
      
      #target_coords <- coords_refset_sub %>% select(contains("Var"))
      #nn_of_missing <- FNN::get.knnx(target_coords, cx_missing, k=1, algorithm="kd_tree")
      #block_of_missing <- coords_refset_sub[nn_of_missing$nn.index, "block"]
      
      ## CHERRY PICKING
      
      missing_vars <- cx_missing$sort_mv_id %>% unique()
      cx_missing %<>% mutate(block=NA)
      
      if(!cherrypick_same_margin){
        for(vv in missing_vars){
          #row_of_this <- which(cx_missing$sort_mv_id == vv)
          
          target_coords_same_margin <- coords_refset_sub %>% 
            #filter(sort_mv_id == vv) %>% 
            dplyr::select(contains("Var"), ix, block)
          
          cx_missing_same_margin <- cx_missing# %>%
            #filter(sort_mv_id == vv)
          
          nn_of_missing_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                     cx_missing_same_margin[,1:dd], k=1, algorithm="kd_tree")
          
          block_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "block"]
          
          cx_missing[#row_of_this, 
                     "block"] <- block_of_missing_same_margin
        }
      } else {
        for(vv in missing_vars){
          row_of_this <- which(cx_missing$sort_mv_id == vv)
          
          target_coords_same_margin <- coords_refset_sub %>% 
            filter(sort_mv_id == vv) %>% 
            dplyr::select(contains("Var"), ix, block)
          
          cx_missing_same_margin <- cx_missing %>%
            filter(sort_mv_id == vv)
          
          nn_of_missing_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                     cx_missing_same_margin[,1:dd], k=1, algorithm="kd_tree")
          
          block_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "block"]
          
          cx_missing[row_of_this, "block"] <- block_of_missing_same_margin
        }
      }
      
      
      block_of_missing <- cx_missing$block
      
      # if there are leftovers then look resolution before those.
      max_res <- coords_all$res %>% max()
      res_of_missing <- max_res + 1 
      
      blockname = sprintf("block_res%02d", res_of_missing)
      parent_blockname = sprintf("block_res%02d", res_of_missing - 1 - 1*(nrow(cx)>0))
      
      coords_res_miss <- coords_missing %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_missing,
               block = as.numeric(factor(block_of_missing)) + max_block_number,
               res = res_of_missing)
      
      coords_all <- bind_rows(coords_all, coords_res_miss %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_miss %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_missing <- coords_res_miss %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_missing)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    }
  })
  #cat("Resolution part 2:", as.numeric(last_steps["elapsed"]), "\n")
  
  return(list(coords_blocking = coords_all,
              parchi_map = parchi_map %>% unique(),
              thresholds = thresholds_list,
              res_is_ref = res_is_ref))
}


make_tree_old1 <- function(coords, na_which, axis_cell_size=c(5,5), K=c(2,2), max_res=Inf, last_not_reference=T){
  #Rcpp::sourceCpp("src/multires_utils.cpp")
  
  #coords <- rnorm(1000, 0, 2) %>% matrix(ncol=2) %>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  #colnames(coords) <- c("Var1", "Var2")
  
  coords_allres <- coords %>% as.data.frame()
  coords_missing <- coords[is.na(na_which),]
  coords_availab <- coords[!is.na(na_which),] %>% as.data.frame()
  
  
  nr <- nrow(coords_availab)
  dd <- ncol(coords_availab) - 1
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  
  coords_refset <- cx %>% mutate(block=NA, res=NA) %>% #, part=NA) %>%
    dplyr::filter(Var1 < minx)
  
  cell_size <- prod(axis_cell_size)
  n_u_cells <- nr/cell_size
  
  #LL <- floor( log(n_u_cells)/log(4)  )
  thresholds_list <- list()
  max_block_number <- 0
  res <- 1
  
  timings <- rep(0, 3)
  
  first_steps <- system.time({
    cat("Creating resolution")
    while((res <= max_res) & (nrow(cx) > 0)){
      cat(" ", res,)
      # first, tessellate to find locations that are spread out. select one for each cell.
      # second, with the resulting subsample then actually split into blocks
      
      # generate threshold to make small cells from which we will sample 1 location
      # fixed threshold regularly on each axis
      #thresholds_knots <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=axis_cell_size[i] * 2^(res-1) + 1))
      # based on quantiles -- hopefully more uniform partitioning
      
      timings[1] <- timings[1] + 
        system.time({
          
          thresholds_knots <- 1:dd %>% lapply(function(i) 
            spamtree:::kthresholds(coords_availab[,i], axis_cell_size[i] * K[i]^(res-1)))
          
          grid_size <- thresholds_knots %>% lapply(function(x) length(x)+1) %>% unlist() %>% prod()
          
          if(grid_size < nrow(cx)){
            # if grid is too thin considering available coordinates, skip this and partition directly
            coords_knots <- cx %>% as.matrix() %>%
              spamtree:::axis_parallel(thresholds_knots, 4) %>%
              #mutate(block = max_block_number + block) %>%
              group_by(block) %>% summarise(ix = ix[round(n()/2)])
            coords_knots %<>% as.data.frame() %>% left_join(cx, by=c("ix"="ix")) %>% dplyr::select(contains("Var"), ix)
          } else {
            coords_knots <- cx 
          }
          
        })["elapsed"]
      
      
      timings[2] <- timings[2] + 
        system.time({
          
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) spamtree:::kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res]] <- thresholds_res
          
          coords_res <- coords_knots %>% as.matrix() %>% 
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            mutate(block = max_block_number + block,
                   res = res) %>%
            dplyr::select(contains("Var"), ix, block, res)#, part)
          max_block_number <- coords_res %>% pull(block) %>% max()
          
        })["elapsed"]
      timings[3] <- timings[3] + 
        system.time({
          
          # move selected locations to the reference set
          cx %<>% dplyr::filter(!(ix %in% coords_res$ix))
          coords_refset %<>% bind_rows(coords_res)
          
          # keep track of overall tessellation
          blockname = sprintf("block_res%02d", res)
          coords_keeptrack <- coords_availab %>% 
            dplyr::select(contains("Var"), ix) %>% 
            as.matrix() %>%
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            rename(!!sym(blockname) := block) %>%
            dplyr::select(ix, !!sym(blockname))
          
          coords_availab %<>% left_join(coords_keeptrack, by=c("ix"="ix"))
          
        })["elapsed"]
      res <- res + 1
    }
  })
  cat(".\n")
  
  res_is_ref <- rep(1, res-1)
  if(last_not_reference & (res < max_res)){
    res_is_ref[ length(res_is_ref) ] <- 0
  }
  
  cat("Finalizing resolutions layers \n")
  #print(timings)
  #cat("Resolution part 1:", as.numeric(first_steps["elapsed"]), "\n")
  
  last_steps <- system.time({
    # now that we have the multiresolution partitioning, 
    # we create a table with the parent-child relationships
    
    relate <- coords_availab %>% dplyr::select(contains("Var"), ix, contains("block_res")) %>%
      dplyr::inner_join(coords_refset %>% dplyr::select(ix, res, block),#, part),
                        by=c("ix"="ix"))
    max_block_number <- 0
    res <- coords_refset$res %>% max()
    for(i in 1:res){
      blockname <- sprintf("block_res%02d", i)
      relate %<>% mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
      max_block_number <- max(relate %>% pull(!!sym(blockname)))
    }
    
    parchi_original <- relate[,sprintf("block_res%02d", 1:res)] %>% 
      dplyr::select(contains("block_res")) %>% unique()
    translator <- relate %>% 
      mutate(block_unnorm = relate %>% 
               apply(1, function(rr) rr[sprintf("block_res%02d", rr["res"])])) %>%
      dplyr::select(block_unnorm, block) %>% 
      unique() %>% 
      right_join(data.frame(block_unnorm = 1:max(.$block_unnorm)), by=c("block_unnorm"="block_unnorm"))
    translator[is.na(translator)] <- 0
    
    parchi_map <- spamtree:::number_revalue(parchi_original %>% as.matrix(), translator$block_unnorm, translator$block)
    colnames(parchi_map) <- colnames(parchi_original)
    parchi_map[parchi_map == 0] <- NA
    
    parchi_map <- parchi_map %>% as.data.frame() %>% 
      arrange(!!!syms(sprintf("block_res%02d", 1:(ncol(parchi_map)-1))))
    
    max_block_number <- coords_refset %>% pull(block) %>% max()
    
    # now manage leftouts
    if(nrow(cx) > 0){
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      cx_leftover <- cx[,grepl("Var", colnames(cx))]
      max_res <- coords_refset$res %>% max()
      
      
      coords_refset_sub <- coords_refset %>% filter(res == max_res)
      
      target_coords <- coords_refset_sub %>% select(contains("Var"))
      nn_of_leftover <- FNN::get.knnx(target_coords, cx_leftover, k=1, algorithm="kd_tree")
      
      block_of_leftover <- coords_refset_sub[nn_of_leftover$nn.index, "block"]
      res_of_leftover <- max_res + 1
      
      blockname = sprintf("block_res%02d", res_of_leftover)
      parent_blockname = sprintf("block_res%02d", res_of_leftover-1)
      
      coords_res_leftover <- cx %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_leftover,
               block = as.numeric(factor(block_of_leftover)) + max_block_number,
               res = res_of_leftover)
      
      coords_all <- bind_rows(coords_refset, coords_res_leftover %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_leftover %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_leftover <- coords_res_leftover %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_leftover)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    } else {
      coords_all <- coords_refset
    }
    
    # now manage missing
    
    if(nrow(coords_missing) > 0){
      max_block_number <- coords_all %>% pull(block) %>% max()
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      cx_missing <- coords_missing[,grepl("Var", colnames(coords_missing))]
      
      coords_refset_sub <- coords_refset %>% filter(res == max(res))
      
      target_coords <- coords_refset_sub %>% select(contains("Var"))
      nn_of_missing <- FNN::get.knnx(target_coords, cx_missing, k=1, algorithm="kd_tree")
      
      block_of_missing <- coords_refset_sub[nn_of_missing$nn.index, "block"]
      # if there are leftovers then look resolution before those.
      max_res <- coords_all$res %>% max()
      res_of_missing <- max_res + 1 
      
      blockname = sprintf("block_res%02d", res_of_missing)
      parent_blockname = sprintf("block_res%02d", res_of_missing - 1 - 1*(nrow(cx)>0))
      
      coords_res_miss <- coords_missing %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_missing,
               block = as.numeric(factor(block_of_missing)) + max_block_number,
               res = res_of_missing)
      
      coords_all <- bind_rows(coords_all, coords_res_miss %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_miss %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_missing <- coords_res_miss %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_missing)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    }
  })
  #cat("Resolution part 2:", as.numeric(last_steps["elapsed"]), "\n")
  
  return(list(coords_blocking = coords_all,
              parchi_map = parchi_map %>% unique(),
              thresholds = thresholds_list,
              res_is_ref = res_is_ref))
}


make_tree_old <- function(coords, na_which, axis_cell_size=c(5,5), K=c(2,2), max_res=Inf){
  #Rcpp::sourceCpp("src/multires_utils.cpp")
  
  #coords <- rnorm(1000, 0, 2) %>% matrix(ncol=2) %>% apply(2, function(x) (x-min(x))/(max(x)-min(x)))
  #colnames(coords) <- c("Var1", "Var2")
  
  dd <- ncol(coords) - 1
  mv_group_ix <- coords %>% as.data.frame() %>% 
    group_by(!!!syms(paste0("Var", 1:dd))) %>% 
    group_indices()
  
  coords_allres <- coords %>% as.data.frame() %>% mutate(gix = mv_group_ix)
  
  coords_missing <- coords_allres[is.na(na_which),] %>% as.matrix()
  coords_availab <- coords_allres[!is.na(na_which),] %>% as.data.frame()
  
  
  nr <- nrow(coords_availab)
  
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  
  coords_refset <- cx %>% mutate(block=NA, res=NA) %>% #, part=NA) %>%
    dplyr::filter(Var1 < minx)
  
  cell_size <- prod(axis_cell_size)
  n_u_cells <- nr/cell_size
  
  #LL <- floor( log(n_u_cells)/log(4)  )
  thresholds_list <- list()
  max_block_number <- 0
  res <- 1
  
  timings <- rep(0, 3)
  
  first_steps <- system.time({
    while((res <= max_res) & (nrow(cx) > 0)){
      cat("Creating resolution ", res,"\n")
      # first, tessellate to find locations that are spread out. select one for each cell.
      # second, with the resulting subsample then actually split into blocks
      
      # generate threshold to make small cells from which we will sample 1 location
      # fixed threshold regularly on each axis
      #thresholds_knots <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=axis_cell_size[i] * 2^(res-1) + 1))
      # based on quantiles -- hopefully more uniform partitioning
      
      timings[1] <- timings[1] + 
        system.time({
          
          thresholds_knots <- 1:dd %>% lapply(function(i) 
            spamtree:::kthresholds(coords_availab[,i], axis_cell_size[i] * K[i]^(res-1)))
          
          grid_size <- thresholds_knots %>% lapply(function(x) length(x) + 2) %>% unlist() %>% prod()
          
          if(grid_size < nrow(cx)){
            # if grid is too thin considering available coordinates, skip this and partition directly
            coords_knots <- cx %>% spamtree:::axis_parallel(thresholds_knots, 4) %>%
              #mutate(block = max_block_number + block) %>%
              group_by(block) %>% #summarise(gix = gix[sample(1:n(), 1)])
              mutate(gfilt = ifelse(gix == gix[sample(1:n(), 1)], 1, 0)) %>%
              #mutate(gfilt = ifelse(gix %in% gix[sample(1:n(), ifelse(n() > 3, 1, n()), replace=F)], 1, 0)) %>%
              filter(gfilt == 1) %>% 
              dplyr::select(block, ix, gix)
            
            coords_knots %<>% as.data.frame() %>% 
              left_join(cx, by=c("ix"="ix", "gix"="gix")) %>% 
              dplyr::select(contains("Var"), ix, gix)
          } else {
            coords_knots <- cx 
          }
          
        })["elapsed"]
      timings[2] <- timings[2] + 
        system.time({
          
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) spamtree:::kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res]] <- thresholds_res
          
          coords_res <- coords_knots %>% as.matrix() %>% 
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            mutate(block = max_block_number + block,
                   res = res) %>%
            dplyr::select(contains("Var"), ix, gix, block, res)#, part)
          max_block_number <- coords_res %>% pull(block) %>% max()
          
        })["elapsed"]
      timings[3] <- timings[3] + 
        system.time({
          
          # move selected locations to the reference set
          cx %<>% dplyr::filter(!(ix %in% coords_res$ix))
          coords_refset %<>% bind_rows(coords_res)
          
          # keep track of overall tessellation
          blockname = sprintf("block_res%02d", res)
          coords_keeptrack <- coords_availab %>% 
            dplyr::select(contains("Var"), ix, gix) %>% 
            as.matrix() %>%
            spamtree:::axis_parallel(thresholds_res, 4) %>%
            rename(!!sym(blockname) := block) %>%
            dplyr::select(ix, gix, !!sym(blockname))
          
          coords_availab %<>% left_join(coords_keeptrack, by=c("ix"="ix", "gix"="gix"))
          
        })["elapsed"]
      res <- res + 1
    }
  })
  
  res_is_ref <- rep(1, res-1)
  
  cat("Finalizing resolutions layers \n")
  #print(timings)
  #cat("Resolution part 1:", as.numeric(first_steps["elapsed"]), "\n")
  
  last_steps <- system.time({
    # now that we have the multiresolution partitioning, 
    # we create a table with the parent-child relationships
    
    relate <- coords_availab %>% dplyr::select(contains("Var"), ix, gix, contains("block_res")) %>%
      dplyr::inner_join(coords_refset %>% dplyr::select(ix, gix, res, block),#, part),
                        by=c("ix"="ix", "gix"="gix"))
    max_block_number <- 0
    res <- coords_refset$res %>% max()
    for(i in 1:res){
      blockname <- sprintf("block_res%02d", i)
      relate %<>% mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
      max_block_number <- max(relate %>% pull(!!sym(blockname)))
    }
    
    parchi_original <- relate[,sprintf("block_res%02d", 1:res)] %>% 
      dplyr::select(contains("block_res")) %>% unique()
    translator <- relate %>% 
      mutate(block_unnorm = relate %>% 
               apply(1, function(rr) rr[sprintf("block_res%02d", rr["res"])])) %>%
      dplyr::select(block_unnorm, block) %>% 
      unique() %>% 
      right_join(data.frame(block_unnorm = 1:max(.$block_unnorm)), by=c("block_unnorm"="block_unnorm"))
    translator[is.na(translator)] <- 0
    
    parchi_map <- spamtree:::number_revalue(parchi_original %>% as.matrix(), translator$block_unnorm, translator$block)
    colnames(parchi_map) <- colnames(parchi_original)
    parchi_map[parchi_map == 0] <- NA
    
    parchi_map <- parchi_map %>% as.data.frame() %>% 
      arrange(!!!syms(sprintf("block_res%02d", 1:(ncol(parchi_map)-1))))
    
    max_block_number <- coords_refset %>% pull(block) %>% max()
    
    # now manage leftouts
    if(nrow(cx) > 0){
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      cx_leftover <- cx[,grepl("Var", colnames(cx))]
      max_res <- coords_refset$res %>% max()
      coords_refset_sub <- coords_refset %>% filter(res == max_res)
      
      target_coords <- coords_refset_sub %>% select(contains("Var"))
      nn_of_leftover <- FNN::get.knnx(target_coords, cx_leftover, k=1, algorithm="kd_tree")
      
      block_of_leftover <- coords_refset_sub[nn_of_leftover$nn.index, "block"]
      res_of_leftover <- max_res + 1
      
      blockname = sprintf("block_res%02d", res_of_leftover)
      parent_blockname = sprintf("block_res%02d", res_of_leftover-1)
      
      coords_res_leftover <- cx %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_leftover,
               block = as.numeric(factor(block_of_leftover)) + max_block_number,
               res = res_of_leftover)
      
      coords_all <- bind_rows(coords_refset, coords_res_leftover %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_leftover %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_leftover <- coords_res_leftover %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_leftover)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    } else {
      coords_all <- coords_refset
    }
    
    # now manage missing
    
    if(nrow(coords_missing) > 0){
      max_block_number <- coords_all %>% pull(block) %>% max()
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      cx_missing <- coords_missing[,grepl("Var", colnames(coords_missing))]
      
      coords_refset_sub <- coords_refset %>% filter(res == max(res))
      
      target_coords <- coords_refset_sub %>% select(contains("Var"))
      nn_of_missing <- FNN::get.knnx(target_coords, cx_missing, k=1, algorithm="kd_tree")
      
      block_of_missing <- coords_refset_sub[nn_of_missing$nn.index, "block"]
      # if there are leftovers then look resolution before those.
      max_res <- coords_all$res %>% max()
      res_of_missing <- max_res + 1 
      
      blockname = sprintf("block_res%02d", res_of_missing)
      parent_blockname = sprintf("block_res%02d", res_of_missing - 1 - 1*(nrow(cx)>0))
      
      coords_res_miss <- coords_missing %>% 
        as.data.frame() %>%
        mutate(parent_block = block_of_missing,
               block = as.numeric(factor(block_of_missing)) + max_block_number,
               res = res_of_missing)
      
      coords_all <- bind_rows(coords_all, coords_res_miss %>% #mutate(part=NA) %>% 
                                dplyr::select(-parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% left_join(coords_res_miss %>% dplyr::select(-res) %>% rename(!!sym(blockname) := block))
      })
      
      parchi_of_missing <- coords_res_miss %>% 
        rename(!!sym(parent_blockname) := parent_block, !!sym(blockname) := block) %>% 
        dplyr::select(contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% left_join(parchi_of_missing)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    }
  })
  #cat("Resolution part 2:", as.numeric(last_steps["elapsed"]), "\n")
  
  return(list(coords_blocking = coords_all,
              parchi_map = parchi_map %>% unique(),
              thresholds = thresholds_list,
              res_is_ref = res_is_ref))
}
