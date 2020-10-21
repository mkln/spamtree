make_tree_old <- function(coords, na_which, sort_mv_id, 
                      axis_cell_size=c(5,5), K=c(2,2), 
                      max_res=Inf, last_not_reference=T,
                      cherrypick_same_margin = T,
                      cherrypick_group_locations = T,
                      mvbias=0){
  # mvbias: 0 = treat all multivariate margins the same
  # >0 = prefer picking sparser margins for lower resolutions
  
  mv_id_weights <- table(na_which, sort_mv_id) %>% `[`(1,) %>% 
    magrittr::raise_to_power(-mvbias) %>% magrittr::divide_by(sum(.))
  mv_id_weights <- data.frame(mv_id_weight=mv_id_weights) %>% 
    tibble::rownames_to_column("sort_mv_id") %>% mutate(sort_mv_id=as.numeric(sort_mv_id))
  
  coords_allres <- coords %>% as.data.frame() %>% cbind(sort_mv_id)
  coords_missing <- coords_allres[is.na(na_which),] 
  coords_availab <- coords_allres[!is.na(na_which),] %>% as.data.frame()
  
  nr <- nrow(coords_availab)
  dd <- ncol(coords_availab) - 2
  
  joining_coords <- paste0("Var", 1:dd)
  names(joining_coords) <- joining_coords
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  colmvid <- dd+2#which(grepl("mv_id", colnames(cx)))

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
            coords_knots <- cx[,-colmvid] %>% as.matrix() %>%
              spamtree:::axis_parallel(thresholds_knots, 4) %>%
              cbind(data.frame(sort_mv_id = cx[,colmvid])) %>% left_join(mv_id_weights, by=c("sort_mv_id"="sort_mv_id")) %>%
              #mutate(block = max_block_number + block) %>%
              group_by(block) %>% summarise(ix = ix[sample(1:n(), 1, prob=mv_id_weight)], .groups="keep")
            
            if(cherrypick_group_locations){
              coords_knots <- coords_knots %>% as.data.frame() %>% 
                left_join(cx, by=c("ix"="ix")) %>% dplyr::select(-ix, -sort_mv_id) %>%
                left_join(cx, by=joining_coords) %>% #by=c("Var1"="Var1", "Var2"="Var2")) %>%
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
      
      timings[2] <- timings[2] + 
        system.time({
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) spamtree:::kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res]] <- thresholds_res
          
          coords_res <- coords_knots[,-colmvid] %>% as.matrix() %>% 
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
    
    parchi_original <- relate[,sprintf("block_res%02d", 1:res), drop=F] %>% 
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
    
    parchi_map <- parchi_map %>% as.data.frame()
    if(ncol(parchi_map)>1){
      parchi_map %<>% 
      arrange(!!!syms(sprintf("block_res%02d", 1:(ncol(parchi_map)-1))))
    }
    
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

make_tree <- function(coords, na_which, sort_mv_id, 
                      axis_cell_size=c(5,5), K=c(2,2), 
                      start_level=0, tree_depth = Inf,
                      last_not_reference=T,
                      cherrypick_same_margin = T,
                      cherrypick_group_locations = T,
                      mvbias=0){
  # mvbias: 0 = treat all multivariate margins the same
  # >0 = prefer picking sparser margins for lower resolutions

  max_res <- start_level + tree_depth 
  
  mv_id_weights <- table(na_which, sort_mv_id) %>% `[`(1,) %>% 
    magrittr::raise_to_power(-mvbias) %>% magrittr::divide_by(sum(.))
  mv_id_weights <- data.frame(mv_id_weight=mv_id_weights) %>% 
    tibble::rownames_to_column("sort_mv_id") %>% mutate(sort_mv_id=as.numeric(sort_mv_id))
  
  coords_allres <- coords %>% as.data.frame() %>% cbind(sort_mv_id)
  coords_missing <- coords_allres[is.na(na_which),] 
  coords_availab <- coords_allres[!is.na(na_which),] %>% as.data.frame()
  
  nr <- nrow(coords_availab)
  dd <- ncol(coords_availab) - 2
  
  joining_coords <- paste0("Var", 1:dd)
  names(joining_coords) <- joining_coords
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  colmvid <- dd+2#which(grepl("mv_id", colnames(cx)))
  
  coords_refset <- cx %>% mutate(block=NA, res=NA) %>% #, part=NA) %>%
    dplyr::filter(Var1 < minx)
  
  cell_size <- prod(axis_cell_size)
  n_u_cells <- nr/cell_size
  
  #LL <- floor( log(n_u_cells)/log(4)  )
  thresholds_list <- list()
  max_block_number <- 0
  res <- start_level+1
  
  timings <- rep(0, 3)
  
  cat("Branching the tree")
  
  res_ix <- 1
  
  first_steps <- system.time({
    while((res <= max_res) & (nrow(cx) > 0)){
      
      cat(" ", res, "(",res_ix,")")
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
            coords_knots <- cx[,-colmvid] %>% as.matrix() %>%
              spamtree:::axis_parallel(thresholds_knots, 4) %>%
              cbind(data.frame(sort_mv_id = cx[,colmvid])) %>% left_join(mv_id_weights, by=c("sort_mv_id"="sort_mv_id")) %>%
              #mutate(block = max_block_number + block) %>%
              group_by(block) %>% summarise(ix = ix[sample(1:n(), 1, prob=mv_id_weight)], .groups="keep")
            
            if(cherrypick_group_locations){
              coords_knots <- coords_knots %>% as.data.frame() %>% 
                left_join(cx, by=c("ix"="ix")) %>% dplyr::select(-ix, -sort_mv_id) %>%
                left_join(cx, by=joining_coords) %>% #by=c("Var1"="Var1", "Var2"="Var2")) %>%
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
      
      timings[2] <- timings[2] + 
        system.time({
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) spamtree:::kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res_ix]] <- thresholds_res
          
          coords_res <- coords_knots[,-colmvid] %>% as.matrix() %>% 
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
      res_ix <- res_ix + 1
    }
  })
  
  cat(".\n")
  
  res_is_ref <- rep(1, res_ix-1)
  if(last_not_reference & (res < max_res)){
    res_is_ref[ length(res_is_ref) ] <- 0
  }
  
  cat("Finalizing with leaves.\n")
  #print(timings)
  #cat("Resolution part 1:", as.numeric(first_steps["elapsed"]), "\n")
  
  last_steps <- system.time({
    # now that we have recursive domain partitioning, 
    # we create a table with the parent-child relationships
    
    relate <- coords_availab %>% dplyr::select(contains("Var"), ix, contains("block_res")) %>%
      dplyr::inner_join(coords_refset %>% dplyr::select(ix, res, block),#, part),
                        by=c("ix"="ix"))
    max_block_number <- 0
    res_from <- coords_refset$res %>% min()
    res_to <- coords_refset$res %>% max()
    for(i in res_from:res_to){
      blockname <- sprintf("block_res%02d", i)
      relate %<>% mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
      max_block_number <- max(relate %>% pull(!!sym(blockname)))
    }
    
    parchi_original <- relate[,sprintf("block_res%02d", res_from:res_to), drop=F] %>% 
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
    
    parchi_map <- parchi_map %>% as.data.frame()
    if(ncol(parchi_map)>1){
      parchi_map %<>% 
        arrange(!!!syms(colnames(parchi_map)))
    }
    
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
      coords_refset_sub <- coords_refset %>% dplyr::filter(res %in% c(max(res)))
      
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
            #dplyr::filter(res == max(res)) %>%
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
            dplyr::filter(sort_mv_id == vv) %>%
            dplyr::select(contains("Var"), ix, block)
          
          cx_missing_same_margin <- cx_missing %>%
            dplyr::filter(sort_mv_id == vv)
          
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


make_tree_devel2 <- function(coords, sort_mv_id, 
                      axis_cell_size=c(5,5), K=c(2,2), 
                      max_res=10){
  
  #coordx <- coords 
  #coords <- coords[sample(1:nrow(coords), 1000, replace=F),] %>% as.data.frame() %>% arrange(Var1, Var2)
  
  dd <- sum(grepl("Var", colnames(coords)))
  unique_coords <- coords[,1:dd] %>% 
    as.data.frame() %>% 
    unique()
  
  join_vars <- paste0("Var", 1:dd)
  names(join_vars) <- join_vars

  coordranges <- coords[,1:dd] %>% apply(2, range)
  #max_res <- 4
  coord_by_res <- list()
  res_meta <- matrix(0, ncol=dd, nrow=3)
  rownames(res_meta) <- c("cells_every_split", 
                          "cells_this_res", 
                          "knots_this_res")
  coords_knots <- unique_coords #data.frame(Var1=numeric(0), Var2=numeric(0))
  
  if(dd==3){ coords_knots %<>% mutate(Var3=numeric(0)) }
  nudge <- rep(0, dd)
  n_added_knots <- rep(0, max_res)
  knots_by_res <- list()
  last_thresholds <- list()
  thresholds <- list()
  
  cat("Adding knots...\n")
  for(res in max_res:1){
    xknots <- list()
    thresholds[[res]] <- list()
    for(r in 1:dd){
      res_meta["cells_every_split", r] <- K[r]
      res_meta["cells_this_res", r] <- K[r]^(res-1)
      res_meta["knots_this_res", r] <- K[r]^(res-1) * axis_cell_size[r]
      delta <- (coordranges[2,r]-coordranges[1,r])/res_meta["knots_this_res", r]
      
      xknots_base <- seq(coordranges[1,r], coordranges[2,r], length.out=res_meta["knots_this_res", r]+1)
      xknots[[r]] <- seq(coordranges[1,r] + delta/2, coordranges[2,r], delta)
      
      thresholds[[res]][[r]] <- seq(coordranges[1,r]-1e-6, coordranges[2,r]+1e-6, length.out=K[r]^(res-1)+1) %>%
        head(-1) %>% tail(-1)
      #spamtree:::kthresholds(xknots_base, K[r]^(res-1))
      
      if(res==max_res){
        last_thresholds[[r]] <- thresholds[[r]]
      }
    }
    coords_this_res <- expand.grid(xknots) %>% mutate(res=res)
    
    knots_by_res[[res]] <- coords_this_res
    
    n_added_knots[res] <- nrow(coords_this_res)
    cat("Level:",res,"Adding", n_added_knots[res], "\n")
  }
  
  coords_latent_knots <- bind_rows(knots_by_res)
  coords_latent_knots %>% filter(res<4) %>% 
    ggplot(aes(Var1, Var2, color=factor(res))) + geom_point(size=1) + theme(legend.position="none")
  
  
  expand_to_mv <- function(df, sort_mv_id){
    ixn <- 1:nrow(df)
    umv <- unique(sort_mv_id)
    mv_expand <- expand.grid(ixn, umv) %>% rename(ixn=Var1, sort_mv_id=Var2)
    df %<>% mutate(ixn = ixn) %>%
      full_join(mv_expand) %>% dplyr::select(-ixn)
    return(df)
  }
  
  coords_latent_knots %<>% expand_to_mv(sort_mv_id)
  
  coords_data <- bind_rows(coords %>% 
                             as.data.frame() %>%
                             dplyr::select(-ix) %>%
                             mutate(res=max_res+1) %>% cbind(sort_mv_id),
                           coords_latent_knots) 
  
  
  cat("Partitioning...\n")
  for(res in 1:max_res){
    resname <- "blockres_{res}" %>% glue::glue()
    cat("Level:", res)
    blocking <- coords_data %>% 
      dplyr::select(-res, -contains("blockres_"), -sort_mv_id) %>% 
      unique() %>%
      as.matrix() %>%
      spamtree:::axis_parallel(thresholds[[res]], 4) %>%
      dplyr::select(contains("Var"), block) %>%
      rename(!!sym(resname) := block) 
    #blocking %<>% expand_to_mv(sort_mv_id)
    
    coords_data %<>% left_join(blocking, by=join_vars)
    cat(".\n")
  }
  
  coords_data %<>% 
    mutate(!!sym(paste0("blockres_", max_res+1)) := !!sym(paste0("blockres_", max_res)))
  parchi_relations <- coords_data %>% 
    dplyr::select(res, contains("blockres_")) %>% 
    unique()
  
  if(F){
    valid_blocks <- parchi_relations %>% 
      filter(res == max_res+1) %>% 
      dplyr::pull(!!sym(paste0("blockres_", max_res+1))) %>%
      unique()
    coords_data %<>% filter(!!sym(paste0("blockres_", max_res+1)) %in% valid_blocks)
  }
  
  
  max_block_number <- 0
  for(i in 1:(max_res+1)){
    blockname <- sprintf("blockres_%d", i)
    coords_data %<>% 
      mutate(!!sym(paste0("blockres_", i)) := as.numeric(factor(as.character(!!sym(paste0("blockres_", i)))))) %>%
      mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
    max_block_number <- max(coords_data %>% pull(!!sym(blockname)))
  }
  
  coords_data %<>% mutate(block = coords_data %>% apply(1, function(rr) rr[sprintf("blockres_%d", rr["res"])]))
  trash_blocks <- coords_data %>% dplyr::select(contains("blockres_")) %>% as.matrix() %>% is_greater_than(coords_data$block)
  trash_blocks[trash_blocks==T] <- NA
  trash_blocks[trash_blocks==F] <- 1

  coords_data[,grepl("blockres_", colnames(coords_data))] <-
    coords_data[,grepl("blockres_", colnames(coords_data))] * trash_blocks
  
  for(i in 1:max_res){
    blockname <- sprintf("blockres_%d", i)
    nextres_block <- sprintf("blockres_%d", i+1)
    count_empty_children <- coords_data %>% group_by(!!sym(blockname)) %>% summarise(size=sum(!is.na(!!sym(nextres_block)))) %>% 
      filter(complete.cases(!!sym(blockname)))
    empty_children <- count_empty_children %>% filter(size==0) %>% `[`(,blockname) %>% as.matrix() %>% as.numeric()
    coords_data %<>% filter(!(!!sym(blockname) %in% empty_children))
    #coords_data %>% filter(blockres_6 == 435)
  }
  
  
  #coords_data <- coords_data %>% 
  #  filter(complete.cases(!!sym(paste0("blockres_",max_res+1))))
  parchi_before <- coords_data %>% 
    dplyr::select(contains("blockres_")) %>% 
    unique() %>% arrange(!!!syms(paste0("blockres_", 1:(max_res+1))))
  
  max_block_number <- 0
  for(i in 1:(max_res+1)){
    blockname <- sprintf("blockres_%d", i)
    coords_data %<>% 
      mutate(!!sym(paste0("blockres_", i)) := as.numeric(factor(as.character(!!sym(paste0("blockres_", i)))))) %>%
      mutate(
        block_unnorm = !!sym(blockname),
        !!sym(blockname) := !!sym(blockname) + max_block_number)
    max_block_number <- max(coords_data %>% pull(!!sym(blockname)), na.rm=T)
    
    
  }
  
  coords_data %<>% mutate(block = coords_data %>% apply(1, function(rr) rr[sprintf("blockres_%d", rr["res"])]))
  
  
  parchi_relations <- coords_data %>% 
    dplyr::select(contains("blockres_")) %>% 
    arrange(!!!syms(paste0("blockres_",1:max_res)))
  
  parchi_map <- parchi_relations %>% unique() %>% arrange(!!!syms(paste0("blockres_", 1:(max_res+1))))
  
  if(F){
    remapping <- coords_mapparchi %>% dplyr::select(block_unnorm, !!sym(blockname)) %>% 
      unique() %>% as.matrix()
    focus_coords <- coords_data %>% dplyr::select(!!sym(blockname)) %>%
      as.matrix()
    remapped <- focus_coords %>% spamtree:::number_revalue(remapping[,1], remapping[,2])
    coords_data %<>% mutate(!!sym(blockname) := ifelse(remapped==0, NA, remapped))
    
    
    coords_summary <- knots_by_res[[res <- 3]] %>% 
      rename(block = !!sym("blockres_{res}" %>% glue::glue())) %>%
      group_by(block) %>% 
      summarise(Var1=mean(Var1), Var2=mean(Var2),
                block=block[1])
    ggplot(coords_summary %>% arrange(Var1, Var2) %>% mutate(block=factor(as.numeric(factor(block)))), 
           aes(Var1, Var2, label=block, fill=factor(block))) + 
      geom_raster() + 
      geom_text(size=4) + 
      theme(legend.position="none")
  }
  
  incoords <- cbind(coords, sort_mv_id) %>% as.data.frame()
  coords_blocking <- coords_data %>% 
    left_join(incoords, by=c(join_vars, "sort_mv_id"="sort_mv_id")) %>% 
    #rename(block = !!sym(paste0("blockres_", max_res+1))) %>%
    dplyr::select(contains("Var"), sort_mv_id, ix, block,#!!sym(paste0("blockres_", max_res+1)), 
                  res) 
  
  
  res_is_ref <- c(rep(1, max_res), 0)
  
  return(list(coords_blocking = coords_blocking,
              parchi_map = parchi_map,
              thresholds = thresholds,
              res_is_ref = res_is_ref))
}

make_tree_devel <- function(coords, na_which, sort_mv_id, 
                            axis_cell_size=c(5,5), K=c(2,2), 
                            max_res=10, min_res=0){
  
  #coords_all <- coords
  #sort_mv_id_all <- sort_mv_id
  
  #coords_of_missing <- coords_all[is.na(na_which),]
  #mv_id_missing <- sort_mv_id_all[is.na(na_which)]
  
  #coords <- coords_all[!is.na(na_which),]
  #sort_mv_id <- sort_mv_id_all[!is.na(na_which)]
  
  dd <- sum(grepl("Var", colnames(coords)))
  unique_coords <- coords[,1:dd] %>% 
    as.data.frame() %>% 
    unique()
  
  join_vars <- paste0("Var", 1:dd)
  names(join_vars) <- join_vars
  
  coordranges <- coords[,1:dd] %>% apply(2, range)
  #max_res <- 4
  coord_by_res <- list()
  res_meta <- matrix(0, ncol=dd, nrow=3)
  rownames(res_meta) <- c("cells_every_split", 
                          "cells_this_res", 
                          "knots_this_res")
  coords_knots <- unique_coords #data.frame(Var1=numeric(0), Var2=numeric(0))
  
  if(dd==3){ coords_knots %<>% mutate(Var3=numeric(0)) }
  nudge <- rep(0, dd)
  n_added_knots <- rep(0, max_res)
  knots_by_res <- list()
  last_thresholds <- list()
  thresholds <- list()
  
  cat("Adding knots...\n")
  for(rr in (max_res-min_res):1){
    res <- rr+min_res
    xknots <- list()
    thresholds[[rr]] <- list()
    for(r in 1:dd){
      res_meta["cells_every_split", r] <- K[r]
      res_meta["cells_this_res", r] <- K[r]^(res-1)
      res_meta["knots_this_res", r] <- K[r]^(res-1) * axis_cell_size[r]
      delta <- (coordranges[2,r]-coordranges[1,r])/res_meta["knots_this_res", r]
      
      xknots_base <- seq(coordranges[1,r], coordranges[2,r], length.out=res_meta["knots_this_res", r]+1)
      xknots[[r]] <- seq(coordranges[1,r] + delta/2, coordranges[2,r], delta)
      
      thresholds[[rr]][[r]] <- seq(coordranges[1,r]-1e-6, coordranges[2,r]+1e-6, length.out=K[r]^(res-1)+1) %>%
        head(-1) %>% tail(-1)
      #spamtree:::kthresholds(xknots_base, K[r]^(res-1))
      
      if(res==max_res){
        last_thresholds[[r]] <- thresholds[[r]]
      }
    }
    coords_this_res <- expand.grid(xknots) %>% mutate(res=res)
    
    knots_by_res[[rr]] <- coords_this_res
    
    n_added_knots[rr] <- nrow(coords_this_res)
    cat("Level:",res,"Adding", n_added_knots[rr], "\n")
  }
  
  coords_latent_knots <- bind_rows(knots_by_res)
  #coords_latent_knots %>% filter(res<4) %>% 
  #  ggplot(aes(Var1, Var2, color=factor(res))) + geom_point(size=1) + theme(legend.position="none")
  
  
  expand_to_mv <- function(df, sort_mv_id){
    ixn <- 1:nrow(df)
    umv <- unique(sort_mv_id)
    mv_expand <- expand.grid(ixn, umv) %>% rename(ixn=Var1, sort_mv_id=Var2)
    df %<>% mutate(ixn = ixn) %>%
      full_join(mv_expand) %>% dplyr::select(-ixn)
    return(df)
  }
  
  coords_latent_knots %<>% expand_to_mv(sort_mv_id)
  
  if(T){
    coords_data <- bind_rows(coords %>% 
                               as.data.frame() %>%
                               dplyr::select(-ix) %>%
                               mutate(res=max_res+1) %>% cbind(sort_mv_id),
                             coords_latent_knots) 
    
    
    cat("Partitioning...\n")
    for(res in (min_res+1):max_res){
      rr <- res - min_res
      resname <- "blockres_{res}" %>% glue::glue()
      cat("Level:", res)
      blocking <- coords_data %>% 
        dplyr::select(-res, -contains("blockres_"), -sort_mv_id) %>% 
        unique() %>%
        as.matrix() %>%
        spamtree:::axis_parallel(thresholds[[rr]], 4) %>%
        dplyr::select(contains("Var"), block) %>%
        rename(!!sym(resname) := block) 
      #blocking %<>% expand_to_mv(sort_mv_id)
      
      coords_data %<>% left_join(blocking, by=join_vars)
      cat(".\n")
    }
    
    coords_data %<>% 
      mutate(!!sym(paste0("blockres_", max_res+1)) := !!sym(paste0("blockres_", max_res)))
    parchi_relations <- coords_data %>% 
      dplyr::select(res, contains("blockres_")) %>% 
      unique()
    
    if(F){
      valid_blocks <- parchi_relations %>% 
        filter(res == max_res+1) %>% 
        dplyr::pull(!!sym(paste0("blockres_", max_res+1))) %>%
        unique()
      coords_data %<>% filter(!!sym(paste0("blockres_", max_res+1)) %in% valid_blocks)
    }
    
    max_block_number <- 0
    #for(i in 1:(max_res+1)){
    for(i in (min_res+1):(max_res+1)){
      rr <- i - min_res
      blockname <- sprintf("blockres_%d", i)
      coords_data %<>% 
        mutate(!!sym(paste0("blockres_", i)) := as.numeric(factor(as.character(!!sym(paste0("blockres_", i)))))) %>%
        mutate(!!sym(blockname) := !!sym(blockname) + max_block_number)
      max_block_number <- max(coords_data %>% pull(!!sym(blockname)))
    }
    
    coords_data %<>% mutate(block = coords_data %>% apply(1, function(rr) rr[sprintf("blockres_%d", rr["res"])]))
    trash_blocks <- coords_data %>% dplyr::select(contains("blockres_")) %>% as.matrix() %>% 
      is_greater_than(coords_data$block)
    trash_blocks[trash_blocks==T] <- NA
    trash_blocks[trash_blocks==F] <- 1
    
    coords_data[,grepl("blockres_", colnames(coords_data))] <-
      coords_data[,grepl("blockres_", colnames(coords_data))] * trash_blocks
    
  }
  
  last_name <- "blockres_{max_res+1}" %>% glue::glue()
  #for(i in 1:max_res){
  for(i in max_res:(min_res+1)){
    rr <- i - min_res
    blockname <- sprintf("blockres_%d", i)
    #prevres_block <- sprintf("blockres_%d", i-1)
    count_empty_children <- coords_data %>% group_by(!!sym(blockname)) %>% 
      summarise(size=sum(!is.na(!!sym(last_name)))) %>% 
      filter(complete.cases(!!sym(blockname)))
    empty_children <- count_empty_children %>% filter(size==0) %>% `[`(,blockname) %>% as.matrix() %>% as.numeric()
    coords_data %<>% filter(!(!!sym(blockname) %in% empty_children))
    #coords_data %>% filter(blockres_6 == 435)
  }
  
  #coords_data <- coords_data %>% 
  #  filter(complete.cases(!!sym(paste0("blockres_",max_res+1))))
  parchi_before <- coords_data %>% 
    dplyr::select(contains("blockres_")) %>% 
    unique() %>% arrange(!!!syms(paste0("blockres_", (min_res+1):(max_res+1)))) %>% 
    filter(complete.cases(!!sym(paste0("blockres_", max_res+1))))
  
  
  max_block_number <- 0
  for(i in (min_res+1):(max_res+1)){
    rr <- i - min_res
    blockname <- sprintf("blockres_%d", i)
    coords_data %<>% 
      mutate(!!sym(paste0("blockres_", i)) := as.numeric(factor(as.character(!!sym(paste0("blockres_", i)))))) %>%
      mutate(
        block_unnorm = !!sym(blockname),
        !!sym(blockname) := !!sym(blockname) + max_block_number)
    max_block_number <- max(coords_data %>% pull(!!sym(blockname)), na.rm=T)
  }
  
  coords_data %<>% mutate(block = coords_data %>% apply(1, function(rr) rr[sprintf("blockres_%d", rr["res"])]))
  
  
  parchi_relations <- coords_data %>% 
    dplyr::select(contains("blockres_")) %>% 
    arrange(!!!syms(paste0("blockres_",(min_res+1):max_res)))
  
  parchi_map <- parchi_relations %>% 
    filter(complete.cases(!!sym(paste0("blockres_", max_res+1)))) %>%
    unique() %>% 
    arrange(!!!syms(paste0("blockres_", (min_res+1):(max_res+1)))) 
  
  if(F){
    remapping <- coords_mapparchi %>% dplyr::select(block_unnorm, !!sym(blockname)) %>% 
      unique() %>% as.matrix()
    focus_coords <- coords_data %>% dplyr::select(!!sym(blockname)) %>%
      as.matrix()
    remapped <- focus_coords %>% spamtree:::number_revalue(remapping[,1], remapping[,2])
    coords_data %<>% mutate(!!sym(blockname) := ifelse(remapped==0, NA, remapped))
    
    
    coords_summary <- knots_by_res[[res <- 3]] %>% 
      rename(block = !!sym("blockres_{res}" %>% glue::glue())) %>%
      group_by(block) %>% 
      summarise(Var1=mean(Var1), Var2=mean(Var2),
                block=block[1])
    ggplot(coords_summary %>% arrange(Var1, Var2) %>% mutate(block=factor(as.numeric(factor(block)))), 
           aes(Var1, Var2, label=block, fill=factor(block))) + 
      geom_raster() + 
      geom_text(size=4) + 
      theme(legend.position="none")
  }
  
  incoords <- cbind(coords, sort_mv_id) %>% as.data.frame()
  coords_blocking <- coords_data %>% 
    left_join(incoords, by=c(join_vars, "sort_mv_id"="sort_mv_id")) %>% 
    #rename(block = !!sym(paste0("blockres_", max_res+1))) %>%
    dplyr::select(contains("Var"), sort_mv_id, ix, block,#!!sym(paste0("blockres_", max_res+1)), 
                  res) 
  
  
  res_is_ref <- c(rep(1, length(unique(coords_blocking$res))-1), 0)
  
  colnames(parchi_map) <- paste0("blockres_", 1:ncol(parchi_map))
  coords_blocking %<>% mutate(res = res-min(res)+1)

  return(list(coords_blocking = coords_blocking,
              parchi_map = parchi_map,
              thresholds = thresholds,
              res_is_ref = res_is_ref))
}


manage_for_predictions <- function(simdata_missing,
                                   coords_blocking, parchi_map){
  # manage for predictions
  #coords_missing <- simdata_missing[,1:2]
  #mv_id_missing <- simdata_missing$mv_id
  dd <- 2
  coords_missing <- simdata_missing %>% dplyr::select(contains("Var"))
  maxres <- max(coords_blocking$res)
  pred_parchis <- parchi_map[,paste0("blockres_", (maxres-1):maxres)] %>% filter(complete.cases(.))
  from_name <- colnames(pred_parchis)[1]
  repl_name <- colnames(pred_parchis)[2]
  coords_lastknots <- coords_blocking %>% filter(res == max(res)-1)
  nn_of_missing <- FNN::get.knnx(coords_lastknots[,1:dd], 
                                 coords_missing, 
                                 k=1, algorithm="kd_tree")
  simdata_missing %<>% mutate(block = coords_lastknots[nn_of_missing$nn.index, "block"]) %>%
    rename(!!sym(from_name) := block) %>% left_join(pred_parchis) %>%
    rename(block = !!repl_name) %>%
    dplyr::select(-!!sym(from_name)) %>% 
    mutate(res = maxres)
  
  return(simdata_missing)
}


manage_for_predictions_new <- function(simdata_missing,
                                   coords_blocking, parchi_map){
  # manage for predictions
  #coords_missing <- simdata_missing[,1:2]
  #mv_id_missing <- simdata_missing$mv_id
  dd <- 2
  coords_missing <- simdata_missing %>% dplyr::select(contains("Var"))
  maxres <- max(coords_blocking$res)
  maxblock <- max(coords_blocking$block)
  pred_parchis <- parchi_map[,paste0("blockres_", (maxres-1):maxres)] %>% filter(complete.cases(.))
  from_name <- colnames(pred_parchis)[1]
  repl_name <- colnames(pred_parchis)[2]
  coords_lastknots <- coords_blocking %>% filter(res == max(res)-1)
  nn_of_missing <- FNN::get.knnx(coords_lastknots[,1:dd], 
                                 coords_missing, 
                                 k=1, algorithm="kd_tree")
  
  simdata_miss <- simdata_missing %>% mutate(block = coords_lastknots[nn_of_missing$nn.index, "block"]) %>%
    rename(!!sym(from_name) := block) %>% left_join(pred_parchis) %>%
    rename(blocktemp = !!repl_name) %>%
    mutate(res = maxres+1,
           blocktemp = maxblock + as.numeric(factor(blocktemp))) %>%
    rename(!!sym(repl_name) := blocktemp)
  
  simdata_parchi <- simdata_miss %>% dplyr::select(contains("blockres_")) %>% unique()
  
  simdata_miss %<>%
    rename(block = !!repl_name) %>% 
    dplyr::select(-!!sym(from_name)) 
  
  return(list(simdata_missing = simdata_miss %>% dplyr::rename(sort_mv_id = mv_id),
              pred_parchis = simdata_parchi))
}
