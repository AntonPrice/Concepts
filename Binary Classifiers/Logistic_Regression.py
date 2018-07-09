#Logistic Regression functions developed from scratch
#Original notebook - https://github.com/AntonPrice/Titanic_miniproject/blob/master/Development%20Notebooks/Titanic%20Part%202%20-%20Building%20a%20Basic%20Model.ipynb



#Sigmoid Activation Function
def sigmoid(z):
    s = 1/(1+np.exp(-z))  
    return s

#Random Initializer
def initializer(dim):   
    w = np.random.uniform(-1,0,(dim, 1))
    b = np.random.uniform()
    return w, b

#One full pass
def Logistic_Reg(X, Y, w, b, regularizer = None , ld = 0.5):
    m = X.shape[0]

    #Forward Prop
    Z = np.nansum(w*X.T, axis = 0 ) + b
    A = sigmoid(Z) 
    
    #Logistic Cross-Entropy Loss
    if regularizer == "L2" :
        #L2 Regularization
        reg = (ld / (2*m)) * np.asscalar(np.matmul(w.T, w))
        cost = -(1/m)*np.nansum(Y*np.log(A) +(1 - Y)*np.log(1 - A)) + reg
    if regularizer == "L1" :
        #L1 Regularization
        reg = (ld / (2*m)) * np.sum(np.absolute(w))
        cost = -(1/m)*np.nansum(Y*np.log(A) +(1 - Y)*np.log(1 - A)) + reg
    else :
        #defaults to no regularization
        cost = -(1/m)*np.nansum(Y*np.log(A) +(1 - Y)*np.log(1 - A))
    
    
    
    #Gradients
    dZ = np.reshape(A - Y, (m, 1))
    dw = np.matmul(X.T, dZ) / m
    db = (1/m)*np.sum(dZ)

    #cache and output
    cache = {"dw": dw,
             "db": db}
    return cost, cache

#ADAM optimization algorithm for quickly training our logistic regression parameters
def adam_optmize(w, b, dw, db, v, s, t = 2, learning_rate = 0.1, Beta1 = 0.9, Beta2 = 0.999,  epsilon = 1e-8) :
    #Compute moving averages
    v["dw"] = v["dw"]*Beta1 + (1-Beta1)*dw
    v["db"] = v["db"]*Beta1 + (1-Beta1)*db
    
    #Compute bias-corrected first moment
    v_cor_dw = v["dw"] / (1 - (Beta1**t))
    v_cor_db = v["db"] / (1 - (Beta1**t))
    
    #Moving averages for squared grads
    s["dw"] = s["dw"]*Beta2 + (1-Beta2)*(np.square(dw))
    s["db"] = s["db"]*Beta2 + (1-Beta2)*(db**2)
    
    #Compute bias-corrected second moment 
    s_cor_dw = s["dw"] / (1 - (Beta2**t))
    s_cor_db = s["db"] / (1 - (Beta2**t))
    
    #Update parameters
    w = w - (learning_rate * (v_cor_dw / (np.sqrt(s_cor_dw) + epsilon)))
    b = b - (learning_rate * (v_cor_db / (np.sqrt(s_cor_db) + epsilon)))
    
    return w, b, v, s


#Optimization function to compute our weights matrix
def Train_model(w, b, X, Y, num_iterations, learning_rate = 0.5, regularizer = None, optimizer = None, decay=None, conv = None):
    costs = []
    k = 1.5
    
    
    #If using ADAM then initialize adam parameters
    if optimizer == "adam" :
        v = {}
        s = {}
        v["dw"] = np.zeros((w.shape))
        v["db"] = 0
        s["dw"] = np.zeros((w.shape))
        s["db"] = 0
        
    
    #iterate over n passes
    for i in range(num_iterations):
        #Propogate our network per iteration
        cost, grads = Logistic_Reg(X, Y, w, b, regularizer = regularizer)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        if decay == "power" :
            learning_rate = 0.95** i
        elif decay == "inv_sqrt"  and i > 1:
            learning_rate = k / np.sqrt(i)
        elif decay == "pow_inv" and i > 1 :
            learning_rate = k ** (1/i)
        
        if optimizer == "adam" :
            w, b, v, s = adam_optmize(w, b, dw, db, v, s, learning_rate = learning_rate)
        else :
            #Gradient Descent as default
            w = w - learning_rate*dw
            b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        j = len(costs)
        
        if conv == True and  j >= 3 and  costs[j - 1] < cost and costs[j - 1] < cost :
            print("Converged after iteration %d", i)
            break
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#Prediction function for generating outputs - effectively forward prop
#Would use softmax for multi-class however this is binary so can use simple logic
def predict(w, b, X):
    Y_pred = np.zeros((X.shape[0]))
    
    Z = np.nansum(w*X.T, axis = 0 ) + b
    A = sigmoid(Z) 
    
    for i in range(A.shape[0]):
        if A[i] > 0.5 :
            Y_pred[i] = 1
        else :
            Y_pred[i] = 0
    
    return Y_pred
