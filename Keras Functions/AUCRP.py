#Recall at fixed precision keras loss function 
#Original notebook - https://github.com/AntonPrice/Titanic_miniproject/blob/master/Development%20Notebooks/Titanic%20Part%209%20-%20ROC%20and%20Non-Differentiable%20Optimization.ipynb

lams = {}

#To optimize aucpr simply integrate over r@p across a range using simpson's rule integration
def aucpr_loss(target, output, k = 5):
    
    zeros = tf.zeros_like(output)
    ones = tf.ones_like(output)
    
    #Define all Lplus and Lminus matrices
    cond_z = tf.equal(zeros, target)
    cont_one = tf.not_equal(zeros, target)
    
    ind_z = tf.where(cond_z)
    ind_one = tf.where(cont_one)
    
    one_labels = tf.gather_nd(target, ind_one)
    one_logits = tf.gather_nd(output, ind_one)
    
    z_labels = tf.gather_nd(target, ind_z)
    z_logits = tf.gather_nd(output, ind_z)
    
    #Calculate basic loss functions
    L_plus = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_labels, logits=one_logits))
    L_minus = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=z_labels, logits=z_logits))
        
    Y_plus = tf.reduce_sum(one_labels)
    
    #dynamic assign of lambda variables
    for j in range(k):
        lams['lam_tf'+str(k)] =  tf.Variable(0.01, tf.float32, name='lam_tf'+str(k))
    
    
    fin_loss = 0.0
    
    #Simpson's rule integrate
    for i in range(k):
        if i == 0 :
            alph_t1 = float(0.5)
        else :
            alph_t1 = float(0.5 + ((1 - 0.5)*(i))/k)
        
        alph_t = float(0.5 + ((0.95 - 0.5)*(i + 1)/k))
        #alph_t = 0.95
    
        #Calculate p@r losses
        loss_one_t = tf.multiply(tf.add(1.0, lams['lam_tf'+str(k)]), L_plus) 
        loss_two_t = tf.multiply(tf.multiply(tf.divide(alph_t, tf.subtract(1.0, alph_t)), lams['lam_tf'+str(k)]), L_minus)
        loss_three_t = tf.multiply(-1.0, tf.multiply(lams['lam_tf'+str(k)], Y_plus))
        
        #Calculate t losses (note had to add absoloute function here to increase convergence)
        fin_loss_t = tf.abs(tf.add(tf.add(loss_one_t, loss_two_t), loss_three_t))
        
        #Update lambda_k
        lams['lam_tf'+str(k)] = upd_lambda_rp(lams['lam_tf'+str(k)],  L_plus, L_minus, Y_plus, alph_t)
        
        #Update and sum
        if i == 0:
            loss_in = fin_loss_t*alph_t
            #loss_in = 0.0
        else :
            loss_in = (alph_t - alph_t1)*fin_loss_t
            #loss_in = abs(fin_loss_t)
        
        fin_loss += loss_in
    
    return fin_loss