#Decision Tree Binary Classifier from scratch
#Original source https://github.com/AntonPrice/Titanic_miniproject/blob/master/Development%20Notebooks/Titanic%20Part%207%20-%20Learning%20Concepts%20-%20Decision%20Trees.ipynb



def Split_data(X_in, Y_in, col=0, n_outputs=2):
    outputs_X = []
    outputs_Y = []
    threshholds = np.zeros(n_outputs)
    

    if len(X_in.shape) == 1 :
        X_max = max(X_in)
        X_min = min(X_in)
        for i in range(n_outputs):
            #For simplicity leaving it as evenly segmenting over selected variable range
            #This should suffice due to normalizing my inputs
            threshholds[i] = ((X_max - X_min) / n_outputs)*(i + 1) + X_min
            if i == (n_outputs - 1):
                threshholds[i] = X_max
            
        for j in range(n_outputs):
            if j == 0:
                output_mat_X = X_in[np.where(X_in <= threshholds[j])]
                output_mat_Y = Y_in[np.where(X_in <= threshholds[j])]                       
            else:  
                output_mat_X = X_in[np.where(np.logical_and(threshholds[j - 1]<  X_in, X_in <= threshholds[j] ))]
                #output_mat_X = X_in[np.where(threshholds[j - 1]<=  X_in && X_in < threshholds[j])]
                output_mat_Y = Y_in[np.where(np.logical_and(threshholds[j - 1]<  X_in, X_in <= threshholds[j] ))]                
                                    
            outputs_X.append(output_mat_X)
            outputs_Y.append(output_mat_Y)
            
    else:
        X_max = max(X_in[:, col])
        X_min = min(X_in[:, col])
        for i in range(n_outputs):
            threshholds[i] = ((X_max - X_min) / n_outputs)*(i + 1) + X_min
            
        if i == (n_outputs - 1):
                threshholds[i] = X_max
        
        for j in range(n_outputs):
            if j == 0:
                output_mat_X = X_in[np.where(X_in[:, col] <= threshholds[j])]
                output_mat_Y = Y_in[np.where(X_in[:, col] <= threshholds[j])]                       
            else:  
                output_mat_X = X_in[np.where(np.logical_and(threshholds[j - 1]<  X_in[:, col] , X_in[:, col]  <= threshholds[j] ))]
                output_mat_Y = Y_in[np.where(np.logical_and(threshholds[j - 1]<  X_in[:, col] , X_in[:, col] <= threshholds[j] ))]                
                                    
            outputs_X.append(output_mat_X)
            outputs_Y.append(output_mat_Y)
    
    
    return outputs_X, outputs_Y, threshholds

	
def calc_gini(Y_inputs):
    num_datasets = len(Y_inputs)
    denom = np.zeros(len(Y_inputs))
    num_ones = np.zeros(len(Y_inputs))
    num_zeros = np.zeros(len(Y_inputs))
    predictions = np.zeros(len(Y_inputs))
    
    gini = 0.0
    
    num_obs = 0
    
    for i in range(num_datasets):
        Y_data = Y_inputs[i]
        denom[i] = len(Y_data)
        num_ones[i] = sum(Y_data)
        num_zeros[i] = denom[i] - num_ones[i]
        num_obs += denom[i]
          
    for j in range(num_datasets):    
        leaf_prop = denom[j] / num_obs
        leaf_gini = ((num_ones[j]/ denom[j]) * (num_ones[j]/ denom[j])) + ((num_zeros[j] / denom[j]) *(num_zeros[j] / denom[j]))
        predictions[j] = (num_ones[j]/ denom[j]) 

        
        gini += leaf_gini*leaf_prop
        
    gini = 1 - gini
    return gini, predictions
	
	
def find_split(X_input, Y_input, max_iters=2, max_children = 2):
    X_splits = []
    Y_splits = []
    results_cache = {}
    gini = 0.5
    error = 0.5
    no_split = 0
    
    sum_Y = sum(Y_input)
    
    if sum_Y == 0 :
        results_cache['gini'] =  0
        results_cache['predictions'] =  0
        results_cache['split_col'] = 9
        results_cache['num_childs'] = 0
        results_cache['thresholds'] = 0
        return X_splits, Y_splits, results_cache
    
    if int(sum_Y) == len(Y_input) :
        results_cache['gini'] =  0
        results_cache['predictions'] =  1
        results_cache['split_col'] = 9
        results_cache['num_childs'] = 0
        results_cache['thresholds'] = 0
        return X_splits, Y_splits, results_cache
    
    num_vars = X_input.shape[1] - 1
    
    for i in range(max_iters):
        test_col = randint(0, num_vars)
        if max_children == 2:
            num_childs = 2
        else :
            num_childs = randint(2, max_children)
        
        #Generate split
        testX, testY, thresh = Split_data(X_input, Y_input, test_col, num_childs)
        
        #Generate another split if results in null branch
        error = 0.5
        while error > 0 :
            n = 0
            for j in range(len(testY)) : 
                if len(testY[j]) == 0:
                    error += 1
                    
            if error == 0.5:
                error = 0
                break
            else :
                test_col = randint(0, num_vars)
                testX, testY, thresh = Split_data(X_input, Y_input, test_col, 2)
                num_childs = 2
                error = 0.5
                n += 1
            
            #Add non-convergence criteria
            if n >= 15 :
                error = 0
                print('Error did not split')
                no_split = 1
                break
        
        #Calculate Gini
        if no_split == 0 :
          test_gin, preds = calc_gini(testY)
        else :
            #Account for non-convergence
            test_gin = 0.5
            preds = 0.5
            X_splits.append(X_input)
            Y_splits.append(Y_input)
        
        if i == 0 :
            X_splits = testX
            Y_splits = testY
            results_cache['gini'] =  test_gin
            results_cache['predictions'] =  preds
            results_cache['split_col'] = test_col
            results_cache['num_childs'] = num_childs
            results_cache['thresholds'] = thresh
        elif test_gin < results_cache['gini'] :
            X_splits = testX
            Y_splits = testY
            results_cache['gini'] =  test_gin
            results_cache['predictions'] =  preds
            results_cache['split_col'] = test_col
            results_cache['num_childs'] = num_childs
            results_cache['thresholds'] = thresh
    
    return X_splits, Y_splits, results_cache
	
	
def Generate_Tree(X_input, Y_input, model, cur_depth = 0 , child_num = -1, max_depth = 10, max_childs = 3, max_iters = 10):
    mod = {}
    
    if cur_depth >= max_depth :
        return model, cur_depth
    
    X_split, Y_split, cache = find_split(X_input,Y_input, max_iters, max_childs)
    
    col = cache['split_col']
    childs = cache['num_childs']
    preds = cache['predictions']
    thresh = cache['thresholds']
    
    if childs == 0 :
        #mod['prediction'] = preds
        #model.append(mod)
        #mod = {}
        return model, cur_depth
    
    for n in range(childs) :
        X_new = X_split[n]
        Y_new =Y_split[n]
        index = len(model)
        mod['index'] = index
        mod['Layer'] = cur_depth
        mod['split_col'] = col
        mod['prev_parent'] = child_num
        #mod['hist'] = hist
        
        
        if n == 0:
            mod['threshmin'] = -99.99
            mod['threshmax'] = thresh[n]
        else :
            mod['threshmin'] = thresh[n - 1]
            mod['threshmax'] = thresh[n]
            
        mod['prediction'] = preds[n]
        
        model.append(mod)
        mod = {}
        depth = cur_depth + 1
        Generate_Tree(X_new,Y_new, model, depth,index, max_depth, max_childs, max_iters)
        
    
    return model, cur_depth
	
	
def Gen_Predictions(X_input, model):
    num_obs = X_input.shape[0]
    num_nodes = len(model)

    #Initialize vectors for data extract
    indices = np.zeros(num_nodes)
    prev_index = np.zeros(num_nodes)
    prediction = np.zeros(num_nodes)
    col = np.zeros(num_nodes)
    threshmax = np.zeros(num_nodes)
    threshmin = np.zeros(num_nodes)
    Layer = np.zeros(num_nodes)
    
    #Initialize Y_hat
    Y_hat = np.zeros(num_obs)
    
    #testing
    for y in range(len(Y_hat)) :
        Y_hat[y] = 9.9

    #Extract model data into useable form
    for n in range(num_nodes) :
        mod = model[n]
        indices[n] = mod['index']
        prev_index[n] = mod['prev_parent']
        prediction[n] = mod['prediction']
        col[n] = mod['split_col']
        threshmax[n] = mod['threshmax']
        threshmin[n] = mod['threshmin']
        Layer[n] = mod['Layer'] 

    #Get meta vars for parsing
    num_layers = int(max(Layer) + 1)
    
    #Iterate predictions
    for m in range(num_obs):
        X_row = X_input[m, :]
        #Initialize first layer as layer -1
        prev_ind = -1
        
        for l in range(num_layers) :
            #Get indices of layer by comparing to previous
            prev_i = [i for i,x in enumerate(prev_index) if x == prev_ind]
            
            for idx in prev_i :
                column = int(col[idx])
                max_t = threshmax[idx]
                min_t = threshmin[idx]
                
                if X_row[column] > min_t and X_row[column] <= max_t:
                    Y_hat[m] = prediction[idx]
                    prev_ind = indices[idx]
         
    
    return Y_hat
	
	
	
