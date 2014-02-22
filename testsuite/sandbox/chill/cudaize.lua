
function table.contains_key(table, key)
  for k in pairs(table) do
    if k == key then
      return true
    end
  end
  return false
end

function valid_indices(stmt, indices)
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   for idx in pairs(indices) do
      if not table.contains_key(cur,idx) then
         return false
      end
   end
   return true
end

function next_clean_level(cur_idxs,level)
   for i=level+1,#cur_idxs do
      --print("Checking '"..cur_idxs[i].."'")
      if (# cur_idxs[i] > 0) then
         --print("Good enough"..(# cur_idxs[i]))
         return i
      end
   end
   return -1 --sentinal that there were no non-dummy indices left
end

function build_order(final_order, tile_idx_names, ctrl_idx_names, tile_idx_map, cur_level)
   order = {}
   for i,k in ipairs(final_order) do
      skip = false
      cur = final_order[i]
      --control loops below our current level should not be in the current order
      for j=cur_level+2,# ctrl_idx_names do
         if ctrl_idx_names[j] == final_order[i] then
            skip = true
         end
      end
      --possibly substitute tile indices ifn necessar
      if table.contains_key(tile_idx_map,final_order[i]) then
         approved_sub = false
         sub_string = tile_idx_map[final_order[i]]
         for j=cur_level+2,# tile_idx_names do
            if tile_idx_names[j] == sub_string then
               approved_sub = true
            end
         end
         if approved_sub then
            cur = sub_string
         end
      end
      if not skip then
         table.insert(order,cur)
      end
   end
   return order
end

function list_to_string(str_list)
   --Helpful debug output
   l = ""
   for i,str in ipairs(str_list) do
      if i > 1 then
         l = l .. ", " .. str
      else
         l = str
      end
   end
   return l
end

function find_cur_level(stmt,idx)
   --Search cur_indices for a idx at stmt
   cur = cur_indices(stmt)
   for i,cidx in ipairs(cur) do
      if cidx == idx then
         return i
      end
   end
   error("Unable to find "..idx.." in current list of indices")
end
function chk_cur_level(stmt,idx)
   --Search cur_indices for a idx at stmt
   cur = cur_indices(stmt)
   for i,cidx in ipairs(cur) do
      if cidx == idx then
         return i
      end
   end
   return -1
end

function find_offset(cur_order, tile, control)
   --print("Looking for "..tile.." and "..control.." in "..list_to_string(cur_order))
   idx1 = -1
   idx2 = -1
   for i,cur in ipairs(cur_order) do
      if(cur == tile) then
         idx1 = i
      end
      if(cur == control) then
         idx2 = i
      end
   end
   if(idx1 < 0) then
      error("Unable to file " .. tile .. " in current list of indices")
   end
   if(idx2 < 0) then
      error("Unable to file " .. control .. " in current list of indices")
   end
   --print("found at level " .. idx2 .. " and " .. idx1)
   if(idx2 < idx1) then
      return idx2-idx1+1
   else
      return idx2-idx1
   end
end

function tile_by_index(stmt,tile_indices, sizes, index_names, final_order, tile_method)

   --stmt = 0 --assume stmt 0
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   if not valid_indices(stmt,tile_indices) then
      error('One of the indices in the first parameter were not '..
            'found in the current set of indices.')
   end
   if not tile_method then tile_method = counted end
   tile_idx_names = {}
   for i,s in ipairs(tile_indices) do tile_idx_names[i]=s end --shallow copy
   ctrl_idx_names = {}
   tile_idx_map = {}
   for k,v in pairs(index_names) do
      valid = false
      if(string.sub(k,1,1) == "l") then
         if string.sub(k,-8) == "_control" then
            i = tonumber(string.sub(k,2,-9))
            if i and i >= 1 and i <= (# tile_indices) then
               ctrl_idx_names[i] = v
               print(string.format("Handling control %s for loop level %d",v,i))
               valid = true
            end
         elseif string.sub(k,-5) == "_tile" then
            i = tonumber(string.sub(k,2,-6))
            if i and i >= 1 and i <= (# tile_indices) then
               --print(string.format("tile %s -> %s",tile_indices[i], v))
               tile_idx_names[i] = v
               tile_idx_map[v] = tile_indices[i]
               --print(string.format("tile %s -> %s",tile_indices[i], v))
               valid = true
            end
         end
      end
      if not valid then error(string.format("%s is not a proper key for specifying "..
                                            "tile or control loop indices\n", k)) end
   end
   
   --filter out control indices (and do name substitution of unprocessed tile indices) for a given level
   cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, -1)
   permute(stmt, cur_order)
   
   for i,cur_idx in ipairs(tile_indices) do
      cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, i-1)
      --Find a offset between tile loop and control loop
      -- 0   = control loop one level above tile loop
      -- -1  = control loop two levels above tile loop
      -- > 0 = tile loop above control loop
      -- In the last case, we do two extra tile commands to get the control
      -- above the tile and then rely on the final permute to handle the
      -- rest
      level = find_cur_level(stmt,cur_idx)
      offset = find_offset(cur_order, tile_idx_names[i], ctrl_idx_names[i])
      if (offset <= 0) then
	 print(string.format("\n[offset<=0]tile(%d, %d, %d, %d,%s,%s,%s)",stmt, level, sizes[i], level+offset, tile_idx_names[i], ctrl_idx_names[i], tile_method)) 
         tile(stmt, level, sizes[i], level+offset, tile_idx_names[i], ctrl_idx_names[i], tile_method)
      else

         tile(stmt, level, sizes[i], level, tile_idx_names[i], ctrl_idx_names[i], tile_method);--regular level
         print(string.format("\n1\n"))

         --print_code(0)

         --flip tile and control loop
         tile(stmt, level+1, level+1);
         print(string.format("\n2\n"))

         --print_code(0)
         tile(stmt, level+1, level);
         print(string.format("\n3\n"))

	 --print_code(0)

         --print(string.format("\n[offset>0]tile(%d, %d, %d, %d,%s,%s,%s)",stmt, level, sizes[i], level, tile_idx_names[i], ctrl_idx_names[i], tile_method)) 

      end
      --Do permutation based on curOrder
	 print_code(0)

      cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, i-1)
      permute(stmt, cur_order);

         --print(string.format("\n4\n"))

   end
end

function normalize_index(stmt,index)
   --stmt = 0 --assume stmt 0cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   l = find_cur_level(stmt, index)
   tile(stmt, l, l)
   --print(string.format("\n[Normalize]tile(%d, %d, %d)",stmt, l,l)) 
end

function is_in_indices(stmt, idx)
	cur = cur_indices(stmt)
	for i=0,#cur,1 do
		if(cur[i]==idx) then
			return true
		end
	end
	return false

end


function copy_to_registers(start_loop, array_name)
  
   --print("starting copy to registers")
   level_tx = -1
   level_ty = -1
   stmt = 0 --assume stmt 0
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   hold_constant = {}
   -- [Malik] first we make sure that tx and ty are consecutive loops in the 2D thread setup, otherwise all levels for subsequent operations are messed up. Start logic.
   cur = cur_indices(stmt)
   table_Size = table.getn(cur)
   --print("Cur indices "..list_to_string(cur))
   --print("The table size is"..table.getn(cur))
   --table.foreach(cur, print)
   --print_code()

   if is_in_indices(stmt,"tx") then   level_tx = find_cur_level(stmt,"tx") end
   if is_in_indices(stmt,"ty") then level_ty = find_cur_level(stmt,"ty") end
   ty_lookup_idx = "" 
   org_level_ty = level_ty
   --if(cur[level_tx+1]~=nil and cur[level_tx+1]~="") then ty_lookup = ty_lookup+1 end
   if(cur[level_ty+1]~=nil and cur[level_ty+1]~="") then 
	ty_lookup_idx = cur[level_ty+1] 
	else
	ty_lookup_idx = cur[level_ty] 
   end
   if level_ty > 0 then
	--print("\ntile(%d,%d,%d)",stmt,level_ty,level_tx+1)
	tile(stmt,level_ty,level_tx+1) 
   end
   --print("\ntylookup is %d",ty_lookup)
--exit(0)
--
   cur = cur_indices(stmt)
   table_Size = table.getn(cur)
   --print("Cur indices "..list_to_string(cur))
   if is_in_indices(stmt,"tx") then   level_tx = find_cur_level(stmt,"tx") end
   if ty_lookup_idx then
	   if is_in_indices(stmt,ty_lookup_idx) then level_ty = find_cur_level(stmt,ty_lookup_idx) end
   end

   ty_lookup = 1
   idx_flag = -1
   -- find the level of the next valid index after ty+1
   if level_ty > 0 then
	   for num= level_ty+ty_lookup,table_Size do
		if(cur[num] ~= "") then
			idx_flag = find_cur_level(stmt,cur[num])
			break
			--print("\n I am checking all indexes after ty+1 %s",idx)
		end
   	end
  end
print("\n I am checking all indexes after ty+1 %s",idx_flag)
print_code(0)
how_many_levels = 1
for ch_lev = idx_flag+1,table_Size,1 do
	
	if(cur[ch_lev] ~= nil and cur[ch_lev] ~= "") then
		how_many_levels = how_many_levels+1
	end
end
print("\n How Many Levels",how_many_levels)

--exit(0)
   -- change this all to reflect the real logic which is to normalize all loops inside the thread loops. 
if(how_many_levels <2) then
   while( idx_flag >= 0) do
--
          for num = level_ty+ty_lookup,(table_Size) do
                if(cur[num] ~= "") then
                    idx=cur[num]
                    --print_code()
                    print("\n[COPYTOREG]tile(%d,%d,%d)",stmt,find_cur_level(stmt,idx),level_tx)
                    tile(stmt,find_cur_level(stmt,idx),find_cur_level(stmt,idx))
                    tile(stmt,find_cur_level(stmt,idx),level_tx)
                    --print("hehe "..cur[num])
		    cur = cur_indices(stmt)
		    print("Cur indices INSIDE"..list_to_string(cur))
		    table_Size = table.getn(cur)
		    print("\n Table Size is: %d",table_Size) 	
		    level_tx = find_cur_level(stmt,"tx")
		    print("\n level TX is: %d",level_tx)
		    level_ty = find_cur_level(stmt,ty_lookup_idx)
		    print("\n level TY is: %d",level_ty)
		    idx_flag = -1
		   -- find the level of the next valid index after ty+1
		    for num= level_ty+ty_lookup,table_Size do
			if(cur[num] ~= nil and cur[num] ~= "") then
				idx_flag = find_cur_level(stmt,cur[num])
				print("\n I am checking all indexes after ty+1 %s",cur[num])
				break
	
				end
   			     end
        	        end
		end
          end
   end
--end
--print_code()
--exit(0)
--]]
--print_code()
--exit(0)


--   level_tx = find_cur_level(stmt,"tx")
--   level_ty = find_cur_level(stmt,"ty")
--   print("\ntile(%d,%d,%d)",stmt,level_ty,level_tx+1)
--   tile(stmt,level_ty,level_tx+1)
   --idx_flag = -1
   -- find the level of the next valid index after ty+1
--[[
   for num= level_ty+1,table_Size do
	if(cur[num] ~= "") then
		idx_flag = find_cur_level(stmt,cur[num])
		break
		--print("\n I am checking all indexes after ty+1 %s",idx)
	end
   end
   -- change this all to reflect the real logic which is to normalize all loops inside the thread loops. 
   while(level_ty+1 < (table_Size-1) and idx_flag >= 0) do
          for num = level_ty+2,(table_Size) do
                if(cur[num] ~= "") then
                    idx=cur[num]
                    print_code()
                    print("\n[COPYTOREG]tile(%d,%d,%d)",stmt,find_cur_level(stmt,idx),level_tx)
                    tile(stmt,find_cur_level(stmt,idx),find_cur_level(stmt,idx))
                    tile(stmt,find_cur_level(stmt,idx),level_tx)
                    --print("hehe "..cur[num])
		    cur = cur_indices(stmt)
		    print("Cur indices "..list_to_string(cur))
		    table_Size = table.getn(cur)
		    print("\n Table Size is: %d",table_Size) 	
		    level_tx = find_cur_level(stmt,"tx")
		    print("\n level TX is: %d",level_tx)
		    level_ty = find_cur_level(stmt,"ty")
		    print("\n level TY is: %d",level_ty)
		    idx_flag = -1
		   -- find the level of the next valid index after ty+1
		    for num= level_ty+1,table_Size do
			if(cur[num] ~= "") then
				idx_flag = find_cur_level(stmt,cur[num])
			break
			--print("\n I am checking all indexes after ty+1 %s",idx)
			end
   		     end
                end
          end
   end
--]]
   --print_code()
   --print("\ntile(%d,%d,%d)",stmt,level_k,level_k)
   --tile(stmt,level_k,level_k)
   
   -- [Malik] end logic
   --print_code()
   start_level = find_cur_level(stmt, start_loop)
   --We should hold contant any block or tile loop
   block_idxs = block_indices()
   thread_idxs = thread_indices()
   --print("\nblock indices are\n")
   table.foreach(block_idxs, print)
   --print("\nthread indices are\n")
   table.foreach(thread_idxs, print)
   --print("\nStart Level: %d",start_level)
   --print("\n Now in Blocks")
   for i,idx in ipairs(block_idxs) do
	--print("\n Idx:%s : Level: %d",idx,find_cur_level(stmt,idx))
      if find_cur_level(stmt,idx) >= start_level then
         table.insert(hold_constant, idx)
	 --print("\nJust inserted %s in block_idxs",idx)
      end
   end
   --print("\n Now in Threads")
   for i,idx in ipairs(thread_idxs) do
	--print("\n Idx:%s : Level: %d",idx,find_cur_level(stmt,idx))
      if find_cur_level(stmt,idx) >= start_level then
         table.insert(hold_constant, idx)
	 --if idx=="ty" then 
	--	for t,tdx in ipairs(hold_constant) do
	--		if tdx == "tx"
		 --print("\nJust inserted %s in th_idxs",idx)
      end
   end
   --print("\nbefore datacopy pvt")
   old_num_stmts = num_statements()
   --print_code()
   print(string.format("\n[DataCopy]datacopy_privatized(%d, %s, %s, vector having privatized levels)",stmt, start_loop, array_name)) 
   --table.foreach(hold_constant, print)
   datacopy_privatized(stmt, start_loop, array_name, hold_constant)

   --print(hold_constant)
   new_num_stmts = num_statements()
   --print("\nthe num of statements:%d\n",new_num_stmt)
   --print_code()
   --exit(0)
   -- [Malik] normalize the copy loops created.
   cur = cur_indices(old_num_stmts)
   --print("Cur indices "..list_to_string(cur))
   for cidx,i in ipairs(cur) do
      if i ~= "tx" and i~="ty" and i~="bx" and i~="by" then
         --tile(old_num_stmts,find_cur_level(old_num_stmts,i),find_cur_level(old_num_stmts,i))
		 --print("\nTILE OF REG: tile(%d,%d,%d)",old_num_stmts,find_cur_level(old_num_stmts,i),find_cur_level(old_num_stmts,i))
      end
   end
   --print_code()
   --print("\nthe num of statements OLD+1 :",(old_num_stmts+1))  
--[[ 
   if( (old_num_stmts+1) <= new_num_stmts) then
   	cur = cur_indices(old_num_stmts+1)
   --print("Cur indices+1 "..list_to_string(cur))
   	for cidx,i in ipairs(cur) do
      		if i ~= "tx" and i~="ty" and i~="bx" and i~="by" then
         		tile(old_num_stmts+1,find_cur_level(old_num_stmts+1,i),find_cur_level(old_num_stmts+1,i))
		 	print("\nTILE OF REG: tile(%d,%d,%d)",old_num_stmts+1,find_cur_level(old_num_stmts+1,i),find_cur_level(old_num_stmts+1,i))
      		end
   	end
   end
--]]
   --Unroll to the last thread level
   --for stmt=old_num_stmts,new_num_stmts-1 do
     -- level = find_cur_level(stmt,thread_idxs[#thread_idxs])--get last thread level
      --if level < #cur_indices(stmt) then
        -- unroll(stmt,level+1,0)
	 --print(string.format("\n[Unroll]unroll(%d, %d, 0)",stmt, level+1)) 
         ----print_code()
      --end
   --end
end

function copy_to_shared(start_loop, array_name, alignment)
   stmt = 0 --assume stmt 0
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))

   start_level = find_cur_level(stmt, start_loop)
   
   old_num_stmts = num_statements()
   --Now, we give it indices for up to two dimentions for copy loop
   copy_loop_idxs = {"tmp1","tmp2"}
   datacopy(stmt, start_level, array_name, copy_loop_idxs, false, 0, 1, alignment,true)
   --print(string.format("\n[DataCopy]datacopy(%d, %d, %s, {\"tmp1\",\"tmp2\"},false,0,1,%d,true)",stmt, start_level, array_name, alignment)) 
   add_sync(stmt,start_loop)
   new_num_stmts = num_statements()
   --print_code()

   --This is fairly CUBLAS2 specific, not sure how well it generalizes,
   --but for a 2D copy, what we want to do is "normalize" the first loop
   --"tmp1" then get it's hard upper bound. We then want to tile it to
   --make the control loop of that tile "ty". We then tile "tmp2" with a
   --size of 1 and make it "tx".
   for stmt=old_num_stmts,new_num_stmts-1 do
      was_no_error, level = pcall(find_cur_level, stmt, "tmp2")
      if was_no_error then 
	 --print_code()	
	 --print("\n Copy to shared: [If was no error]\n")
         find_cur_level(stmt,"tmp2")
         tile(stmt, level, level)
	
         lower,upper = hard_loop_bounds(stmt, level)
         upper = upper + 1
         tx,ty = thread_dims()
         --print("2-loop cleanup: lower, upper: "..lower..", "..upper..", tx: "..tx)
         level = find_cur_level(stmt,"tmp1")
         if tx == upper and ty == 1 then
            --Don't need an extra tile level, just move this loop up
            second_level = find_cur_level(stmt,"tmp2")
            tile(stmt, second_level, 1, level, "tx", "tx", counted)
	    --print(string.format("\n[Tile0]tile(%d, %d, 1, %d,%s,%s,counted)",stmt, second_level, level, "tx", "tx")) 
         else
	    --print_code()
            if(ty == 1) then new_ctrl = "tmp3" else new_ctrl = "ty" end
	   --[[ Commenting out a block of Gabe's code in this control flow
	   -- level = find_cur_level(stmt,"tmp1")
	    tile(stmt, level, level)
	
	    lower,upper = hard_loop_bounds(stmt, level)
            upper = upper + 1
	    --print_code()
	    print("2-loop cleanup: lower, upper: "..lower..", "..upper..", tx: "..tx..", level: "..level)
	    if(math.ceil(upper/ty) > 1)then
            tile(stmt, level, math.ceil(upper/ty), level, "tmp", new_ctrl, counted)
print(string.format("\n[Tile1]tile(%d, %d, %f[%d,%d], %d,%s,%s,counted)",stmt, level,  math.ceil(upper/ty),upper,ty, level, "tmp", new_ctrl)) 
	    else
	    tile(stmt, level, math.ceil(upper/ty), level, "ty", new_ctrl, counted)
print(string.format("\n[Tile1]tile(%d, %d, %f[%d,%d], %d,%s,%s,counted)",stmt, level,  math.ceil(upper/ty),upper,ty, level, "tx", new_ctrl))
	    end
	     	 
	--print_code()    
	    -- [Malik] If here we have the loop upper bound > tx, then we should tile once more after the next tile, to carve out the correct tx. 
	    lower1,upper1 = hard_loop_bounds(stmt,level)
 	    level1 = level
	    stmt1 = stmt
	    -- [Malik] Do the tile after the second level tile with if condition. Just to keep the original order, the tile is being pushed to the end. 
	    
	    print("[Malik]-loop cleanup: lower1, upper1: "..lower1..", "..upper1..", tx: "..tx..", level:"..level1)
	    --print_code()
            --print_code()
	    --level = find_cur_level(stmt,"tmp")
	    --tile(stmt,level,level)
	    --print_code() 

	    --[Malik] if you are moving the loop above the level1, you need to update level1 with new position which would be level1+2 or second_level
	    if(level <= level1) then level1 = level1+2 end
	    print(string.format("\n[Tile2]tile(%d, %d, 1, %d,%s,%s,counted)",stmt, second_level, level, "tx", "tx")) 
	    print("\n----------------------------------")
            --print_code()
	    print("\n**********************************")
	    print("[Malik]-loop cleanup: lower1, upper1: "..lower1..", "..upper1..", tx: "..tx..", level:"..level1)
	    -- [Malik] If the upper bound > tx, we do another tile to carve out the correct tx from a bigger loop. Else just normalize the bounds. 
	    if( upper1 > ty) then
	    third_level = find_cur_level(stmt1,"tmp")
	    print("\n\n\n\t\t\t\tthirdlevel:"..third_level)
	    tile(stmt1, third_level, ty, third_level, "ty", "tmp", counted)
	    print(string.format("\n[Tile3]tile(%d, %d, %d,%d,%s,%s,counted)",stmt1, third_level, ty,third_level, "ty", "tmp"))
	    tile(stmt1,third_level+1,third_level+1)
            print(string.format("\n[Tile3]tile(%d, %d, %d)",stmt1, third_level+1, third_level+1))
	    tile(stmt1,third_level+1,third_level)
            print(string.format("\n[Tile3]tile(%d, %d, %d)",stmt1, third_level+1, third_level))
	    else
	    tile(stmt1,level1,level1)
            print(string.format("\n[Tile3ELSE]tile(%d, %d, %d)",stmt1,level1,level1))
	    end

	    print("\nStarting tmp2\n");--print_code();
	    second_level = find_cur_level(stmt,"tmp2")
	    lower,upper = hard_loop_bounds(stmt,second_level)
 	    level = second_level
	    print("[Malik]-loop cleanup@tmp2: lower, upper: "..lower..", "..upper..", tx: "..tx..", level:"..level)

	    if(math.ceil(upper/tx) > 1)then
	    tile(stmt, second_level,math.ceil(upper/tx), level, "tmp", "tx", counted)
print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,math.ceil(upper/tx),second_level, "tmp", "tx"))
	    else
            tile(stmt, second_level,math.ceil(upper/tx), level, "tx", "tx", counted)
print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,math.ceil(upper/tx),second_level, "tx", "tx"))
	    end
	    --print_code()
	    lower2,upper2 = hard_loop_bounds(stmt,level)
 	    level2 = level
	    stmt2 = stmt
	    print("[Malik]-loop cleanup@tmp2: lower2, upper2: "..lower2..", "..upper2..", tx: "..tx..", level:"..level2)
-- now for the second level.
	    if( upper2 > tx) then
	    forth_level = find_cur_level(stmt2,"tmp")
	    print("\n\n\n\t\t\t\tforthlevel:"..forth_level)
	    --print_code()
	    tile(stmt2, forth_level, 1, forth_level, "tx", "tmp", counted)
	    print(string.format("\n[Tile3B]tile(%d, %d, %d,%d,%s,%s,counted)",stmt2, forth_level, tx,forth_level, "ty", "tmp"))
	    --print_code()
	    --tile(stmt2,forth_level+1,forth_level+1)
            --print(string.format("\n[Tile3B]tile(%d, %d, %d)",stmt2, forth_level+1, forth_level+1))
	    --tile(stmt2,forth_level+1,forth_level)
            --print(string.format("\n[Tile3B]tile(%d, %d, %d)",stmt2, forth_level+1, forth_level))
	    else
	    new_level = find_cur_level(stmt2,"ty")
	    tile(stmt2,level2,1,new_level,"tx","tx",counted)
            print(string.format("\n[Tile3BELSE]tile(%d, %d, %d)",stmt2,level2,level2))
	    tmp_level = find_cur_level(stmt2,"tmp")
	    tile(stmt2,tmp_level,tmp_level)
	    end

	    --print_code()
	    print("\n----------------------------------")
	    --]]

	    --print("\nStarting tmp2\n");--print_code();
	    first_level = find_cur_level(stmt,"tmp1")
	    second_level = find_cur_level(stmt,"tmp2")
	    lower,upper = hard_loop_bounds(stmt,second_level)
 	    
	    --print("[Malik]-loop cleanup@tmp2: lower, upper: "..lower..", "..upper..", tx: "..tx..",first level:"..first_level..",second_level:"..second_level)

	    -- Move the fastest changing dimension loop to the outermost,identified by "tmp2" and to be identified as tx.
	    tile(stmt,second_level,1,first_level,"tx","tx",counted)
  	    --print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,1,first_level, "tx", "tx"))
	    
	    
	    first_level = find_cur_level(stmt,"tmp1")
	    lower_1,upper_1 = hard_loop_bounds(stmt,first_level)
	    tx_level = find_cur_level(stmt,"tx")
	    lower_tx,upper_tx = hard_loop_bounds(stmt,tx_level)

	   
	    if(math.ceil(upper_tx/tx) > 1)then
		tile(stmt,tx_level,tx,tx_level,"tx","tmp_tx",counted)
		--print(string.format("\n[Tile1]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, tx_level,tx,tx_level, "tx", "tmp1"))
		tile(stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tx"))
		--print(string.format("\n[Tile1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tx")))
		if (find_cur_level(stmt,"tx")>find_cur_level(stmt,"tmp_tx")) then
			tile(stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp_tx"))
			--print(string.format("\n[Tile1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp")))
		end
	    --else
		--tile(stmt, tx_level,1, tx_level, "tx", "tx", counted)
		--print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, tx_level,1,tx_level, "tx", "tx"))
	    end
	    --print_code()
		--]]

   	    --print("\nStarting tmp1\n")
	    -- Handle the other slower changing dimension, the original outermost loop, now identified by "tmp1", to be identified as "ty".
	    tile(stmt,find_cur_level(stmt,"tmp1"),find_cur_level(stmt,"tmp1"))	    
	    --print_code()	
	    
	    ty_level = find_cur_level(stmt,"tmp1")
	    lower_ty,upper_ty = hard_loop_bounds(stmt,ty_level)
	    
	    tx_level = find_cur_level(stmt,"tx")
	    lower_tx,upper_tx = hard_loop_bounds(stmt,tx_level)
		--print("[Malik]-loop cleanup@tmp1: lowerty, upperty: "..lower_ty..", "..upper_ty..", ty: "..ty..",ty level:"..ty_level..",tx_level:"..tx_level..", stmt: "..stmt)
	    if(math.ceil(upper_ty/ty) > 1)then
		--print("\n Inside upper_ty/ty > 1\n");
		tile(stmt,ty_level,ty,ty_level,"ty","tmp_ty",counted)
		--print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, ty_level,ty,ty_level, "ty", "tmp"))
		--print_code()
		tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"ty"))
		--print(string.format("\n[Tile2-1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"ty")))
-----------------------------------------------------------------------
----------------------------------------------------------------------
		cur_idxs = cur_indices(stmt)
		--print("\n cur indexes are "..list_to_string(cur_idxs))
		-- Putting ty before any tmp_tx		
		idx_flag = -1
		for num= 0,table.getn(cur_idxs) do
			if(cur[num] == "tmp_tx") then
				idx_flag = find_cur_level(stmt,cur[num])
			break
			--print("\n I am checking all indexes after ty+1 %s",idx)
			end
		end
   		--print("\n so i have found out the value of idx flag as %d",idx_flag) 
		if(idx_flag >=0 ) then	
			if (find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty")) then
				tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
			--print(string.format("\n[Tile2-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
			end
		end
		-- Now Putthing ty before any tmp_ty
		idx_flag = -1
		for num= 0,table.getn(cur_idxs) do
			if(cur[num] == "tmp_ty") then
				idx_flag = find_cur_level(stmt,cur[num])
			break
			--print("\n I am checking all indexes after ty+1 %s",idx)
			end
		end
   		--print("\n so i have found out the value of idx flag as %d",idx_flag) 
		if(idx_flag >=0 ) then	
			if ((find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty"))) then
				tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
				--print(string.format("\n[Tile2-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
			end
		end
	    else
		--cur_idxs = cur_indices(stmt)
		--print("\n Inside upper_ty/ty <= 1\n");
		tile(stmt, ty_level,1, ty_level, "ty", "ty", counted)
		--print(string.format("\n[Tile3]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, ty_level,1,ty_level, "ty", "ty"))
		tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1)
		--print(string.format("\n[Tile3-1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1))
				idx_flag = -1
		if(cur_idxs) then
		for num= 0,table.getn(cur_idxs) do
			if(cur[num] == "tmp_ty") then
				idx_flag = find_cur_level(stmt,cur[num])
			break
			--print("\n I am checking all indexes after ty+1 %s",idx)
			end
		end
		end
   		--print("\n so i have found out the value of idx flag as %d",idx_flag) 
		if(idx_flag >=0 ) then	
		if (find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty")) then
			tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
			--print(string.format("\n[Tile3-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
		end
		end
	    end
		
	    --print_code()
         end
      else
         --copy to shared only created on level, not two, so we use a different approach (MV & TMV)
 	 --print("\n Copy to shared: [If was error]\n")
         level = find_cur_level(stmt,"tmp1")
         tile(stmt, level, level)
	 --print(string.format("\n[Tile]tile(%d, %d, %d)",stmt, level, level)) 
         tx,ty = thread_dims()
         lower,upper = hard_loop_bounds(stmt, level)
         upper = upper+1 --upper bound given as <=, compare to dimentions tx which is <
         --print("upper "..upper.." tx "..tx)
         if upper == tx then
            rename_index(stmt, "tmp1", "tx")
         else
             --TODO: Don't know, maybe do some tileing etc
            --print_code()
	    --print("upper "..upper.." tx "..tx.." stmt: "..stmt.." level: "..level)
	    tile(stmt, level,tx,level, "tx", "tmp_tx", counted)
	    --print_code()
	    --print("stmt:"..stmt.." level+1: "..level+1)
            tile(stmt, level+1,1,level+1,"tx", "tx",counted)
	    tile(stmt,level+1,level)
	    if(ty > 1) then
		--print_code()
		--print("GOING IN")
		lower,upper = hard_loop_bounds(stmt, level+1)
		--upper=125
		--print("NOW FOR Y: upper "..upper.." ty "..ty.." stmt: "..stmt.." level: "..(level+1).." bound:"..math.ceil(upper/ty))
		tile(stmt, level+1,math.ceil(upper/ty),level+1, "tmp_ty", "ty", counted)
		--tile(stmt, level+2,math.ceil(upper/ty),level+2, "tmp_ty", "ty", counted)
	    end
	    --print_code()
	    --rename_index(stmt, "tmp1", "tx")
            --print("Warning: Need to implement some logic here to tile the single level shared copy loop to match thread dimensions")
         end
      end
      --Always add sync
      add_sync(stmt,start_loop)

   end
end

function unroll_to_depth(max_depth)
   cur = cur_indices(0)
   --print("Cur indices "..list_to_string(cur))
   thread_idxs = thread_indices()
   guard_idx = thread_idxs[#thread_idxs]

---- HERE FIND OUT THE LOOPS WHICH ARE COMMON BETWEEN STATEMENTS   
   common_loops = {}
   comm_loops_cnt = 0
   num_stmts = num_statements()
   for stmt=0,num_stmts-1 do
	cur_idxs = cur_indices(stmt)
	print("\nSTMT %d Current Indices: %s",stmt,list_to_string(cur_idxs))
      if(chk_cur_level(stmt,"tx")>0) then
	for ii=0,find_cur_level(stmt,"tx")-1 do
		if(cur_idxs[ii] ~= "bx" and cur_idxs[ii] ~= "by" and cur_idxs[ii] ~= nil and cur_idxs[ii] ~= "tx" and cur_idxs[ii] ~= "ty" and cur_idxs[ii] ~= "") then 
			for stmt1=stmt+1,num_stmts-1 do
				--print("\nstmt1 is "..stmt1)			
				cur_idxs1 = cur_indices(stmt1)
				--print("\nstmt1 cur idxs1 is "..list_to_string(cur_idxs1))			
				for iii=0,find_cur_level(stmt,"tx")-1 do
					if(cur_idxs1[iii] ~= "bx" and cur_idxs1[iii] ~= "by" and cur_idxs1[iii] ~= nil and cur_idxs1[iii] ~= "tx" and cur_idxs1[iii] ~= "ty" and cur_idxs1[iii] ~= "") then 	
						if(cur_idxs[ii] == cur_idxs1[iii]) then
							--print("\nfound idx:"..cur_idxs[ii])
							common_loops[comm_loops_cnt] = cur_idxs[ii]
							comm_loops_cnt = comm_loops_cnt + 1
						end
					end	
				end
			end	
		end
	end
      end
   end
----
--print("\n COMM LOOPS :TOTAL "..comm_loops_cnt..", and are "..list_to_string(common_loops).." this loop :"..common_loops[0])




   repeat
      old_num_stmts = num_statements()

      for stmt=0,old_num_stmts-1 do
         cur_idxs = cur_indices(stmt)
         if(#cur_idxs > 0) then 
	    gaurd_level = -1
	    if(chk_cur_level(stmt,guard_idx)>0) then
	            gaurd_level = find_cur_level(stmt,guard_idx)
	    end
	    if(gaurd_level>-1) then
	            level = next_clean_level(cur_idxs,gaurd_level)
        	    --print("looking at "..stmt)
        	    --print("comparing "..guard_level.." and "..level.." in "..list_to_string(cur_idxs))
        	    --need to handle max_depth
        	    num_unrolled = 0
		    level_unroll_comm = level
        	    level_arr = {}
        	    while level >= 0 do
        	       if num_unrolled == max_depth then break end
        	       print("Unrolling "..stmt.." at level "..(level).." index ".. cur_idxs[gaurd_level+1])
        	       --print_idx()
        	       --print_ri()
		       --print_code()
		       --tile(stmt,level,level)
        	       --unroll(stmt,level,0)
		       level_arr[num_unrolled] = level
        	       --print("finished unroll")
        	       --print_idx()
        	       --print_code()
        	       num_unrolled = num_unrolled + 1
		       guard_level = find_cur_level(stmt,guard_idx)
        	       level = next_clean_level(cur_idxs,level+1)
        	    end
			--print("How many levels for unroll commands"..table.getn(level_arr).." which is "..level_arr[0].." and "..level_arr[#level_arr])
			--if(table.getn(level_arr) ~= nil) then
		    if(level_unroll_comm >= 0)then
		    	for i = table.getn(level_arr),0,-1 do
			--	    	for i = table.getn(level_arr),0,-1 do
			--if num_unrolled == max_depth then break end
			--print_code()
				--print_code()
				--print("[[In stmt "..stmt.." Levels are: ",level_arr[i])
				--print(string.format("[tile]tile(%d, %d, %d)",stmt, level_arr[i],level_arr[i]))  
				print(string.format("[Unroll]unroll(%d, %d, 0)",stmt, level_arr[i]))			
				--tile(stmt,level_arr[i],1,level_arr[i],"s","s",counted)cur_idxs = cur_indices(stmt)
				unroll(stmt,level_arr[i],0)
				--print("finished unroll]]\n")
				--print_code()
				--io.write("Press <Enter> to continue...")
				--io.read()
	
	
			end
		    end
------		
		end    
--[[	    
	    num_unrolled_reverse = 0
            for i=find_cur_level(stmt,"tx")-1,1,-1 do	
               if num_unrolled_reverse == max_depth then break end
	       cur_idxs = cur_indices(stmt)	
	       --print("\n REVERSE UNROLL the indexes in reverse are",list_to_string(cur_idxs)cur_idxs = cur_indices(stmt))
	       if(cur_idxs[i] ~= "bx" and cur_idxs[i] ~= "by" and cur_idxs[i] ~= "") then
			-- to avoid trying to unroll the loops which have more than one statements inside them.
			common_loops_flag = -1
			for j=0,#common_loops-1,1 do
				if(cur_idxs[i]==common_loops[j])then
					common_loops_flag = 1
					--print("\nI HAVE FOUND THE LOOP TO OMIT")
				end
			end
			if common_loops_flag == 1 then break end
               		idx_in_reverse_unroll = cur_idxs[i]
               		--print("Unrolling "..stmt.." at level "..i.." idx "..cur_idxs[i])
			--print_code()
               		--print_idx()
			--print("Unrolling "..stmt)
               		unroll(stmt,i,0)
               		num_unrolled_reverse = num_unrolled_reverse + 1
                        --print_idx()
               		--print_code()
			--print("\nUNROLL DONE")
			--io.read()
	       end
            end
--]]
------
        end
      end
      new_num_stmts = num_statements()
      
   until old_num_stmts == new_num_stmts
   
end

function scalar_expand_product_expression(stmt_num, levels, segment, shared_memory,pad_factor, order)

scalar_expand_by_index(stmt_num,{"new_index"},segment,shared_memory, pad_factor, order)
scalar_expand_by_index(stmt_num,{"new_index"},"RHS",shared_memory, pad_factor, order)

end



function scalar_expand_by_index(stmt_num,levels,rhs,shared_memory,padding, accumulate_then_assign)
  parallel_levels = {}
 
  for i=1, #levels do
        parallel_levels[i] = find_cur_level(stmt_num,levels[i]) 
        lower,upper = hard_loop_bounds(stmt_num, parallel_levels[i])
  
        if(not (lower == 0)) then   
        tile(stmt_num, parallel_levels[i], parallel_levels[i]) 

        end
        parallel_levels[i] = find_cur_level(stmt_num,levels[i]) 

  end  
  
  scalar_expand(stmt_num,parallel_levels,rhs,shared_memory,padding,accumulate_then_assign)

end


function reduce_by_index(stmt_num,level,function_name,seq_levels, bound_level)

  sequential_levels = {}
  
  for i=1, #seq_levels do
        sequential_levels[i] = find_cur_level(stmt_num,seq_levels[i]) 
  
  end 


   cur = cur_indices(stmt_num)
   cur_levels ={}
  if(#level == 1) then
  current_level = find_cur_level(stmt_num,level[1])
   
   


   cur_levels[1] = current_level
  start = current_level
   current_level = current_level + 1 
  
  while (current_level <= #cur and # cur[current_level] <= 0 ) do
     cur_levels[current_level - start + 1] = current_level

     current_level = current_level +1

     if(current_level > #cur) then
       break
     end
  end 

  end

 if(not(#level == 1) )then
    
  for i=1, #level do
        cur_levels[i] = find_cur_level(stmt_num,level[i]) 
 
  end 



 end





  if(#bound_level == 1) then
  reduce(stmt_num, cur_levels,0,function_name, sequential_levels,   find_cur_level(stmt_num,bound_level[1]) )
  else
    reduce(stmt_num, cur_levels,0,function_name, sequential_levels )
  end

end


--flatten loop levels 1 and 2 with NNZ being uninterpreted omega function name
function coalesce_by_index(stmt_num,index_name,indices_to_flatten, inspector)

indices ={}
for i=1, #indices_to_flatten do
   indices[i] = find_cur_level(stmt_num,indices_to_flatten[i]) 
end

  coalesce(stmt_num,index_name, indices, inspector)

 return (num_statements() - 1 )   

end
--split flattened loop level to be a perfect multiple of warp size (32)
function split_with_alignment_by_index(stmt_num,index,amount)
  level = find_cur_level(stmt_num,index) 
  split_with_alignment(stmt_num, level, amount)
   
   return (num_statements() - 1 )   

end

--distribute remainder of splitted statement as it is not cudaized
function distribute_by_index(stmt_nums,index)

  level = find_cur_level(stmt_num,index)
  distribute(stmt_nums, level) 

end




function setup_for_segreduce(stmt_num, target_level,parallel_levels,segment_index,padding,shared_memory, tile_factor_for_second_reduction, second_block_index, stmts_to_reduce)



  current_level = find_cur_level(stmt_num,target_level)



   --1.Identify block level(first parallel dimension)
   --block_level = parallel_levels[0] 

   --2.If shared mem=true elide first block level and expand other parallel levels
   net_parallel_levels={}
   if(shared_memory) then
            
      for i=2, #parallel_levels do

         
         net_parallel_levels[i-1] = find_cur_level(stmt_num,parallel_levels[i]) 
         lower,upper = hard_loop_bounds(stmt_num, net_parallel_levels[i-1])
         if(not (lower == 0)) then   
          shift_to(stmt_num, net_parallel_levels[i-1], 0) 
         end
      end  
   end 


   if(shared_memory) and (padding > 0) then 
   --3.scalar expansion of product expression
   scalar_expand(stmt_num,net_parallel_levels,segment_index,1,padding)
   scalar_expand(stmt_num,net_parallel_levels,"RHS",1,padding)
   end
   --4.Peel last statement at target_level +1 and distribute other statements and fuse with scalar expanded, repeat peel-distribute-fuse cycle
   num_stmts_before_peel = num_statements()
   stmts_to_reduce[1] = num_stmts_before_peel
   peel(stmt_num, current_level + 1, -1)

   num_stmts_after_peel = num_statements()

   stmts={}
   for j=num_stmts_before_peel,num_stmts_after_peel - 1 do
       stmts[j-num_stmts_before_peel+1] = j 
   end   
     

   for i=current_level + 2,#cur do
    
       distribute(stmts,i)
    	
       peel(num_stmts_before_peel, i, -1)  
       fuse(stmts, i - 1 ) 


       num_stmts_after_peel = num_statements()
       stmts={}
       for j=num_stmts_before_peel,num_stmts_after_peel - 1 do
           stmts[j-num_stmts_before_peel + 1] = j 
       end   
      
       num_stmts_before_peel = num_stmts_after_peel  
         

   end
    --5.all parallel levels uptil target level appear in second level reduction
     peeled_statement = num_stmts_after_peel -1;
   
     kernels ={}

     kernels[1] = stmt_num

     kernels[2] = peeled_statement

          print_code(1)	

     distribute(kernels, find_cur_level(stmt_num, parallel_levels[1]))   


     second_phase_reduction_levels= {}
     for i=1,#parallel_levels  do
       
      
        second_phase_reduction_levels[i] = find_cur_level(stmt_num,parallel_levels[i] )
        if parallel_levels[i] == target_level then
         break
       end
     end     
    --6.scalar expand at target level and then distribute the scalar expanded statements from peeled statement
    num_stmts_before_expand = num_statements()
    
    scalar_expand(num_stmts_before_expand - 1,second_phase_reduction_levels,"_P1")
    scalar_expand(num_stmts_before_expand - 1,second_phase_reduction_levels,"RHS")
    num_stmts_after_expand = num_statements()
    
    stmts_to_dist={}
    for j=num_stmts_before_expand -1 ,num_stmts_after_expand - 1 do
           stmts_to_dist[j-num_stmts_before_expand+2] = j 
    end   
     
    distribute(stmts_to_dist ,find_cur_level(stmt_num,parallel_levels[1]))

    stmts_to_fuse ={}
    stmts_to_fuse[1] = stmt_num
    for j=num_stmts_before_expand  ,num_stmts_after_expand - 1 do
           stmts_to_fuse[j-num_stmts_before_expand + 2] = j 
    end  

   
   if not (find_cur_level(stmt_num,parallel_levels[1]) == 1 )then
    distribute(stmts_to_dist ,1)  
  
   end
   --7. fuse scalar expanded statemets with first stateement and fuse till block level
   for j=1,find_cur_level(stmt_num,parallel_levels[1]) do
       fuse(stmts_to_fuse, j)
      
   end 
                            

      tile_loop = {}        
      tile_loop[1] = find_cur_level(peeled_statement,parallel_levels[1])
         
     second_phase_indices = cur_indices(peeled_statement) 
new_indices = {}
  for i=1,#second_phase_indices do
     if i < tile_loop[1] then
      new_indices[i] = second_phase_indices[i]
     end
     if i >= tile_loop[1] then
      new_indices[i+1] = second_phase_indices[i]     
     end

  end        
   new_indices[tile_loop[1]] = second_block_index
  


   tile_by_index(peeled_statement,{parallel_levels[1]},{tile_factor_for_second_reduction},{l1_control=second_block_index},new_indices)CU=1


  shift_to(peeled_statement ,find_cur_level(peeled_statement,parallel_levels[1]),0)

  for i=1, #second_phase_reduction_levels do
         second_phase_reduction_levels[i] = second_phase_reduction_levels[i] + 1
  end 
  num_stmts_before_expand = num_statements()
  scalar_expand(peeled_statement, second_phase_reduction_levels,"_P_DATA1",1)
  scalar_expand(peeled_statement,second_phase_reduction_levels,"RHS",1)
  num_stmts_after_expand = num_statements()

  stmts_to_dist = {}
  stmts_to_dist[1] = peeled_statement
  for j=num_stmts_before_expand,num_stmts_after_expand -1  do
      stmts_to_dist[j-num_stmts_before_expand + 2] = j 
  end
 
  distribute(stmts_to_dist,current_level+1)
  
  return peeled_statement
end
