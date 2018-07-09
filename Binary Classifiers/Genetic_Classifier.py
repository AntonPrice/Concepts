#Genetic Classifier from scratch
#Explained in - https://github.com/AntonPrice/Titanic_miniproject/blob/master/Development%20Notebooks/Titanic%20Part%2011%20-%20Genetic%20Programming.ipynb



#Genetic Algorithm Class
class Genetic_Binary_Classifier(object):
    def __init__(self, func_list, func_mask, eval_func = 1):
        self.func_list = func_list
        self.func_mask = func_mask
        self.best_models = None
        self.eval_func = eval_func
        
    def Log_Loss(self, Y_in, Y_hat_in) :
        epsilon = 1e-9
        m = Y_in.shape[0]
    
        Y_log = Y_in
        Y_hat_log = Y_hat_in
    
        #Y_log[np.where(Y_log == 0)] = -1
        #Y_hat_log[np.where(Y_hat_log == 0)] = -1
    
        loss = -(1.0/m )*np.sum((1 - Y_log)*np.log(1 - Y_hat_log + epsilon) + Y_log*np.log(Y_hat_log + epsilon))
    
        return loss
    
    #Calculate our accuracy figures
    def Get_acc_loss(self, Y_true, Y_hat) :
        #Compute inverse vectors for vectorization
        Y_true_inv = 1 - Y_true
        Y_hat_inv = 1 - Y_hat
        
        if hasattr(np.dot(Y_true,Y_hat.T), "__len__") == True :
            return 100
        if hasattr(np.dot(Y_true_inv,Y_hat_inv.T), "__len__") == True :  
            return 100
             
        acc = 100 - float((np.dot(Y_true,Y_hat.T) + np.dot(Y_true_inv,Y_hat_inv.T))/float(Y_true.size)*100)
    
        return acc

    #Sigmoid Activation Function - with overflow exception
    def sigmoid_overf(self, z):
        if hasattr(z, "__len__") == True :
            z[np.where(z < -7e2)] = -4e-2
        
        else :
            if z <= -7e2:
                z = -4e-2

        s = 1/(1+np.exp(-z))  
        return s

    #Redefined predictions due to tuple out of range errors
    def normalize_predictions_new(self, y_hat):
        #y_out = y_hat.reshape((y_hat.shape[0],))
        y_out = np.around(y_hat)
        
        return y_out
    
    #Generate first prediction level
    def Generate_Predictions(self, childstr, X):
        return self.sigmoid_overf((eval(childstr)))
    
    def Evaluate_Child(self, childstr, X, Y):
        y_hat = self.Generate_Predictions(childstr, X)
        #y_true = normalize_predictions_new(y_hat)
        return self.Log_Loss(Y, y_hat)
    
    def Evaluate_Child_v2(self, childstr, X, Y):
        y_hat = self.Generate_Predictions(childstr, X)
        y_true = self.normalize_predictions_new(y_hat)
        return self.Log_Loss(Y, y_true)
    
    def Evaluate_Child_v3(self, childstr, X, Y):
        y_hat = self.Generate_Predictions(childstr, X)
        y_true = self.normalize_predictions_new(y_hat)
        return self.Get_acc_loss(Y, y_true)

    #First Generation
    def Generate_First_Gen(self, X_len, func_list, func_map):
        select_func = randint(0, len(func_map) - 1)
        mask= func_map[select_func]
        funct = func_list[select_func]

        column_selector_init = randint(0, X_len)
        if column_selector_init == 0 :
            rand_num1 = np.abs(gauss(1, 1))
            rand_num2 = np.abs(gauss(1, 1))

            if mask == 0:
                funcstr = funct + str(rand_num1) + ')'
            elif mask == 2:
                rand_num = np.abs(gauss(1, 1))
                funcstr = funct + str(rand_num2) + ','+ str(rand_num1) + '+1e-9)'
            elif mask == 1:  
                rand_num = np.abs(gauss(1, 1))
                funcstr = funct + str(rand_num2) + ','+ str(rand_num1) + ')'
            else :
                funcstr = 'ERROR'

        else :
            column_selector = column_selector_init - 1
  
            if mask == 0:
                funcstr = funct + 'X[:,' + str(column_selector) + '])'
            elif mask == 2:
                rand_num = np.abs(gauss(1, 1))
                funcstr = funct + 'X[:,' + str(column_selector) + '],'+ str(rand_num) + '+1e-9)'
            elif mask == 1:  
                rand_num = np.abs(gauss(1, 1))
                funcstr = funct + 'X[:,' + str(column_selector) + '],'+ str(rand_num) + ')'
            else :
                funcstr = 'ERROR'

        return funcstr
    
    #Gen Subsequent Generation
    def Generate_Next_Generation(self, X_len, parent1, parent2, mut_prob, func_list, func_mask):
        select_func = randint(0, len(func_mask) - 1)
        mask= func_mask[select_func]
        funct = func_list[select_func]
        
        mutate_rand = uniform(0.0, 1.0)
        
        if mutate_rand <= mut_prob :
            #Mutated outcome
            Mutation = self.Generate_First_Gen(X_len, func_list, func_mask)
            #Generate original non-mutated outcome
            Orig_output = self.Generate_Next_Generation(X_len, parent1, parent2, mut_prob, func_list, func_mask)
            #Generate final output
            funcstr = self.Generate_Next_Generation(X_len, Orig_output, Mutation, mut_prob, func_list, func_mask)
        
        else :
            #No Mutate outcome
            if mask == 0:
                #Single Input - choose random parent
                parent_rand = uniform(0.0, 1.0)
                if parent_rand <= 0.5 :
                    funcstr = funct + parent1 + ')'
                else :
                    funcstr = funct + parent2 + ')'
                    
            elif mask == 2:
                #Divide
                rand_num = np.abs(gauss(1, 1))
                funcstr = funct +  parent1 + ' , ' + parent2 + '+1e-9)'
            elif mask == 1:  
                #Dual Input
                rand_num = np.abs(gauss(1, 1))
                funcstr =funct + parent1 + ' , ' + parent2  + ')'
            else :
                funcstr = 'ERROR'
        
        select_func = randint(0, len(func_mask) - 1)
        mask= func_mask[select_func]
        funct = func_list[select_func]
        
        mutate_rand = uniform(0.0, 1.0)
        
        if mutate_rand <= mut_prob :
            #Mutated outcome
            Mutation = self.Generate_First_Gen(X_len, func_list, func_mask)
            #Generate original non-mutated outcome
            Orig_output = self.Generate_Next_Generation(X_len, parent1, parent2, mut_prob, func_list, func_mask)
            #Generate final output
            funcstr = self.Generate_Next_Generation(X_len, Orig_output, Mutation, mut_prob, func_list, func_mask)
        return funcstr
    
    def Parent_selector(self, X, Y, gen_in, num_parents):
        #Initialize parameters
        num_in = len(gen_in)
        gen_out = []
        loss_out = []
        
        
        for i in range(num_in):
            #Get child and loss of child
            child_str = gen_in[i]
            if self.eval_func == 2 :
                child_loss = self.Evaluate_Child_v2(child_str, X, Y)
            elif self.eval_func == 3 :
                child_loss = self.Evaluate_Child_v3(child_str, X, Y)
            else:
                child_loss = self.Evaluate_Child(child_str, X, Y)
            
            if len(gen_out) <= num_parents:
                #If not full generation out then append child
                gen_out.append(child_str)
                loss_out.append(child_loss)
            else :
                #Otherwise replace worst child if this child is better
                max_loss_out = max(loss_out)
                max_loss_out_ind = loss_out.index(max(loss_out))
                
                if child_loss < max_loss_out :
                    del loss_out[max_loss_out_ind]
                    del gen_out[max_loss_out_ind]
                    
                    gen_out.append(child_str)
                    loss_out.append(child_loss)
        
        return gen_out, loss_out
    
    def Best_Children_Selector(self, new_gen, new_losses, prev_gen, prev_losses) :
        #Set best as previous best
        best_childs = prev_gen
        best_losses = prev_losses
        
        #Initialize variables
        len_new = len(new_gen)
        
        #Iterate through new generation and replace worst of best if better
        for i in range(len_new):
            max_best_loss = max(best_losses)
            max_best_loss_ind = best_losses.index(max(best_losses))
            
            loss_this = new_losses[i]
    
            if loss_this < max_best_loss :
                del best_childs[max_best_loss_ind]
                del best_losses[max_best_loss_ind]
                
                best_childs.append(new_gen[i])
                best_losses.append(loss_this)
            
        return best_childs, best_losses
    
    def Train_Genetic_Alg(self, X, Y, mut_prob = 0.1, num_gens = 10, num_childs = 30, 
                          num_parents = 10, Verbose = 0, initial_gen = None, Output = 0) :
        
        #Define population containers
        generation_par = []
        hof_pars = []
        hof_losses = []
        hall_of_fame = {}
        
        func_list = self.func_list
        func_mask = self.func_mask
        #Initialize parameters
        X_len = X.shape[1]
        
        #Generate first generation
        if self.best_models != None :
            generation_par = best_models
        elif initial_gen == None :
            for i in range(num_parents):
                generation_par.append(self.Generate_First_Gen(X_len, func_list, func_mask))       
        else :
            generation_par = initial_gen
        
        
        #Iterate through generations
        for j in range(num_gens) :
            #Kill previous generation's children
            generation_out = []
            
            #Produce k children in generation j
            for k in range(num_childs):
                num_in = len(generation_par)
                
                #Pick random parents
                par1 = generation_par[randint(0, num_in - 1)]
                par2 = generation_par[randint(0, num_in - 1)]
                
                #Generate child
                childk = self.Generate_Next_Generation(X_len, par1, par2, mut_prob, func_list, func_mask)
                
                #Append to childs generation
                generation_out.append(childk)
            
            #Clear previous generations parents
            generation_par = []
            
            #Generate new parents
            generation_par, par_losses = self.Parent_selector(X, Y, generation_out, num_parents) 
            
            #Generate Hall of Fame
            if len(hof_pars) == 0:
                hof_pars, hof_losses = self.Parent_selector(X, Y, generation_out, 5)
            else :
                hof_pars, hof_losses = self.Best_Children_Selector(generation_par, par_losses, hof_pars, hof_losses) 
                
            #Show training progress
            if Verbose == 1:
                best_in_gen = min(par_losses)
                gen_avg = np.mean(par_losses)
                best_ever = min(hof_losses)
                print("Generation %s, BestGen=%s AverageGen=%s, BestEver=%s" % 
                      (j, best_in_gen, gen_avg, best_ever))
            
            
        
        
        if Verbose == 1 :
            for p in range(len(hof_pars)):
                max_len = 0
                test_len = len(hof_pars[p])
                if p == 0 :
                    max_len = test_len
                if max_len < test_len:
                    max_len = test_len

            print("Largest Model = ", max_len)
            
        self.best_models = hof_pars
        
        if Output == 1:
            return hof_pars, hof_losses
        else:
            return None
    
        #Stack all models in output from best genetic algorithms
    def Stack_models(self, hof_models, X):
        num_mods = len(hof_models)
        
        for i in range(num_mods):
            if i == 0 :
                pred = self.Generate_Predictions(hof_models[i], X)        
                stack_hat = self.normalize_predictions_new(pred)
            else :
                pred = self.Generate_Predictions(hof_models[i], X) 
                hats = self.normalize_predictions_new(pred)
                stack_hat = np.vstack((stack_hat, hats))
        
        return stack_hat
    
    #Create simple voting mechanism from stack
    def pred_from_stack(self, full_hat):
        num_mods = full_hat.shape[0]
        cutoff = num_mods / 2.0
        
        #Sum all votes
        out_hat = np.sum(full_hat, axis=0)
        
        #Allocate majority votes the correct label
        out_hat[np.where(out_hat <= cutoff)] = 0
        out_hat[np.where(out_hat > cutoff)] = 1
        
        return out_hat
    
    #Generate Predictions from ensemble
    def Gen_Final_Preds(self, X):
        full_stack = self.Stack_models(self.best_models, X)
        out_preds = self.pred_from_stack(full_stack)
        return out_preds
        
    