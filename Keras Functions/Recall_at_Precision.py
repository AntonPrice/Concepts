#Recall at fixed precision keras loss function 
#Original notebook - https://github.com/AntonPrice/Titanic_miniproject/blob/master/Development%20Notebooks/Titanic%20Part%209%20-%20ROC%20and%20Non-Differentiable%20Optimization.ipynb


#Took tensorflow function imports from tensorflow source code
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

#Optimize recall at fixed precision
def r_at_p_loss(target, output, alph_tf = 0.95):
    

    lam_tf = tf.Variable(0.01, tf.float32, name='lam_tf')
    #alph_tf = tf.placeholder(tf.float32, name='alph_tf')
    
    #lam_tf = K.variable(lam_tf, name='lam_tf')
    
    zeros = tf.zeros_like(output)
    ones = tf.ones_like(output)
    
    cond_z = tf.equal(zeros, target)
    cont_one = tf.not_equal(zeros, target)
    
    ind_z = tf.where(cond_z)
    ind_one = tf.where(cont_one)
    
    one_labels = tf.gather_nd(target, ind_one)
    one_logits = tf.gather_nd(output, ind_one)
    
    z_labels = tf.gather_nd(target, ind_z)
    z_logits = tf.gather_nd(output, ind_z)
    
    L_plus = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_labels, logits=one_logits))
    L_minus = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=z_labels, logits=z_logits))
    
    Y_plus = tf.reduce_sum(one_labels)
    

    loss_one = tf.multiply(tf.add(1.0, lam_tf), L_plus) 
    loss_two = tf.multiply(tf.multiply(tf.divide(alph_tf, tf.subtract(1.0, alph_tf)), lam_tf), L_minus)
    loss_three = tf.multiply(-1.0, tf.multiply(lam_tf, Y_plus))
    
    fin_loss = tf.add(tf.add(loss_one, loss_two), loss_three)
    
    lam_tf = upd_lambda_rp(lam_tf,  L_plus, L_minus, Y_plus, alph_tf)
    
    return fin_loss

#Define function to optimize lambda by gradient descent (or is it ascent?)
def upd_lambda_rp(lam,  L_plus, L_minus, Y_plus, alph_tf):
    lr = 0.01
    
    #Differential of loss function by lambda
    dl =  tf.subtract(tf.add(L_plus, tf.divide(alph_tf, tf.subtract(1.0, alph_tf))), Y_plus)
    new_lam = tf.add(lam, tf.multiply(lr, dl))
    
    return new_lam


