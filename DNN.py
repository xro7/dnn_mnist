class DNN():
    def __init__(self,X,y,layers):
		X = tf.placeholder(dtype=tf.float32,shape=(None,)+X_train.shape[1:])
		y = tf.placeholder(dtype=tf.float32,shape=(None,)+y_train.shape[1:])
		self.layers = layers
		self.activations = [self.X]
            
    def forward(self):       
        for i,layer in enumerate(self.layers):
            layer.set_input(self.activations[i])
            self.activations.append(layer.forward())
        return self.activations
    
    def cost(self):
        #return tf.reduce_sum(tf.square(self.activations[-1]-self.y))
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.activations[-1],labels=self.y))
    
    def squared_cost(self):
        return tf.reduce_mean(tf.squared_difference(prediction, Y))
class DenseLayer():
    
    number = 0
    
    def __init__(self,units,activation_function=tf.nn.relu,batch_norm=False,keep_prob=1.0):
        self.units = units
        self.keep_prob = keep_prob
        self.activation_function = activation_function
        self.batch_norm = batch_norm
        self.variable_scope_name = 'Dense-'+str(DenseLayer.number)
        DenseLayer.number+=1
        
    def set_input(self,x):
        with tf.variable_scope(self.variable_scope_name):         
            self.x = x
            if(len(x.shape)==4):
                shape = self.x.get_shape().as_list()        
                dim = np.prod(shape[1:])
                self.x = tf.reshape(tensor=self.x,shape=[-1,dim])
            self.init_W((self.x.get_shape().as_list()[1],self.units))
            self.init_b(self.units)
            if(self.batch_norm):
                self.epsilon = 1e-3
                self.scale = tf.get_variable('scale', initializer=tf.ones(shape=[self.units]))
                self.beta =  tf.get_variable('beta', initializer=tf.zeros(shape=[self.units]))
        
    def init_W(self,shape):
        #another way to do this with get variable
        #self.w= tf.Variable(tf.multiply(tf.random_normal(shape),0.01),dtype=tf.float32)
        self.w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('weight',self.w)
    
    def init_b(self,shape):
        #self.b = tf.Variable(tf.fill([shape],0.1),dtype=tf.float32)
        init = tf.constant(0.1,shape=[shape])
        self.b = tf.get_variable('bias', initializer=init)
        tf.summary.histogram('bias',self.b)
        
    def forward(self):
        if self.x == None:
            print('no input defined')
        else:
            #batch norm not working properly
            #or maybe it is working properly but it needs bigger batch size. like 64. This makes sense because you make estimates of
            #mean and variance for every z calculation. This means that the bigger the batch size the more accurate the estimate
            if(self.batch_norm):
                self.z = tf.matmul(self.x,self.w)
                self.batch_mean, self.batch_var = tf.nn.moments(self.z,[0])
                self.z  = tf.nn.batch_normalization(self.z,self.batch_mean,self.batch_var,self.beta,self.scale,self.epsilon)
            else:
                self.z = tf.nn.xw_plus_b(self.x,self.w,self.b)
            if self.activation_function == None:
                self.activation = self.z
            else:
                self.activation  = self.activation_function(self.z)
            self.activation = tf.nn.dropout(self.activation,self.keep_prob)
            tf.summary.histogram('activations',self.activation)
            return self.activation
        
class ConvLayer():

    number = 0
    def __init__(self,kernel_size,number_of_kernels,padding='SAME',activation_function=tf.nn.relu,batch_norm = False,keep_prob=1.0):
        self.kernel_size = kernel_size
        self.number_of_kernels = number_of_kernels
        self.padding = padding
        self.activation_function = activation_function
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm
        self.variable_scope_name = 'Conv-'+str(ConvLayer.number)
        ConvLayer.number+=1
        
    def set_input(self,x):
        with tf.variable_scope(self.variable_scope_name):         
            self.x = x
            if(isinstance(self.kernel_size,tuple)):
                self.init_Kernel(self.kernel_size+(x.get_shape().as_list()[-1],self.number_of_kernels))
            else:
                self.init_Kernel((self.kernel_size,self.kernel_size,x.get_shape().as_list()[-1],self.number_of_kernels)) 
            self.init_b(self.number_of_kernels)
            if(self.batch_norm):
                self.epsilon = 1e-3
                self.scale = tf.get_variable('scale', initializer=tf.ones(shape=[self.number_of_kernels]))
                self.beta =  tf.get_variable('beta', initializer=tf.zeros(shape=[self.number_of_kernels]))
        
    def init_Kernel(self,shape):
        self.kernel=tf.get_variable('kernel',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('kernel',self.kernel)
        
    def init_b(self,shape):
        self.b = tf.get_variable('bias',shape=[shape],initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('bias',self.b)
        
    def forward(self):
        if self.x == None:
            print('no input defined')
        else:
            if self.batch_norm:
                
                self.z = tf.nn.conv2d(self.x , self.kernel, [1, 1, 1, 1], padding=self.padding)
                self.batch_mean, self.batch_var = tf.nn.moments(self.z,[0,1,2])
                self.z  = tf.nn.batch_normalization(self.z,self.batch_mean,self.batch_var,self.beta,self.scale,self.epsilon)
                
            else:
                self.z = tf.nn.conv2d(self.x , self.kernel, [1, 1, 1, 1], padding=self.padding)
                self.z = tf.nn.bias_add(self.z, self.b)  
                    
            if self.activation_function == None:
                self.activation = self.z
            else:
                self.activation  = self.activation_function(self.z)
            #dropout
            self.activation = tf.nn.dropout(self.activation,self.keep_prob)
            tf.summary.histogram('activations',self.activation)
        return self.activation
        
class PoolingLayer():
    
    number = 0
    
    def __init__(self,kernel_size,stride,padding='SAME',pooling='MAX'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = pooling
        self.padding = padding
        self.variable_scope_name = 'Pool-'+str(PoolingLayer.number)
        PoolingLayer.number+=1
        
    def set_input(self,x):
        with tf.variable_scope(self.variable_scope_name):         
            self.x = x
        
    def forward(self):
        if self.x == None:
            print('no input defined')
        else:
            if(isinstance(self.kernel_size,tuple)):
                size = list(self.kernel_size)
            else:
                size = [self.kernel_size,self.kernel_size]   
            if (self.pooling == 'MAX'):
                self.activation = tf.nn.max_pool(self.x,[1]+size+[1],[1,self.stride,self.stride,1],padding=self.padding)
            elif (self.pooling == 'AVG'):
                self.activation = tf.nn.avg_pool(self.x,[1]+size+[1],[1,self.stride,self.stride,1],padding=self.padding)
        return self.activation
    
class EmbeddingLayer():
    
    number = 0
    
    def __init__(self,vocabulary_size,embedding_diamension,pretrained_word2vec=True,as_sequences=None):
        self.embedding_diamension = embedding_diamension
        self.vocabulary_size = vocabulary_size
        self.pretrained_word2vec = pretrained_word2vec
        self.variable_scope_name = 'Embedding-'+str(PoolingLayer.number)
        self.init_Embeddings((self.vocabulary_size,self.embedding_diamension))
        self.as_sequences = as_sequences
        EmbeddingLayer.number+=1
        
    def set_input(self,x):
        with tf.variable_scope(self.variable_scope_name):         
            self.x = x
            if self.as_sequences:
                self.sequence_length = x.shape[1]
        
    def init_Embeddings(self,shape):
        with tf.variable_scope(self.variable_scope_name):  
            if(self.pretrained_word2vec):             
                self.W = tf.Variable(tf.constant(0.0, shape=[self.vocabulary_size, self.embedding_diamension]),trainable=True, name="embedding_weights")
                self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocabulary_size, self.embedding_diamension])
                self.embedding_init = self.W.assign(self.embedding_placeholder)
            else:
                self.W = tf.get_variable("embedding_weights", shape=[self.vocabulary_size, self.embedding_diamension],initializer=tf.contrib.layers.xavier_initializer())
    def forward(self):
        # TODO: implement as sequences to provide data for lstm
        if self.x == None:
            print('no input defined')
        else:
            #this is called activation to follow the pattern of other layers
            self.activation = tf.nn.embedding_lookup(self.W, self.x)
            if self.as_sequences == 'static':
                self.activation = tf.transpose(self.activation, [1,0,2])
                self.activation = tf.reshape(self.activation , [-1,self.embedding_diamension])
                self.activation  = tf.split(self.activation ,self.sequence_length,0)
            elif self.as_sequences == 'dynamic':
                pass
            else:
                self.activation = tf.expand_dims(self.activation, -1)        #need 4 diamensions to apply convolution
            return self.activation
class RnnLayer():
    number = 0
    def __init__(self,units,activation_function = None,cell_type = 'LSTM',keep_prob=1.0,rnn_type='static'):
        self.units = units
        self.activation_function = activation_function
        self.keep_prob = keep_prob
        self.cell_type = cell_type
        self.rnn_type = rnn_type
        self.variable_scope_name = 'Rnn-'+str(RnnLayer.number)
        RnnLayer.number+=1
        
    def set_input(self,x):
        with tf.variable_scope(self.variable_scope_name):         
            self.x = x
        
    def forward(self):
        if self.x == None:
            print('no input defined')
        else:
            if self.cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(self.units,activation=self.activation_function)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
            elif self.cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(self.units,activation=self.activation_function)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
                
            #need to fix this
            #self.activation = tf.nn.dynamic_rnn(cell,self.x,sequence_length=self.sequence_length)
            if (self.rnn_type == 'static'):
                self.output, self.states = tf.nn.static_rnn(cell,self.x,dtype=tf.float32)
                self.activation = self.output[-1]
            elif (self.rnn_type == 'dynamic'):
                self.output, self.states = tf.nn.dynamic_rnn(cell,self.x,dtype=tf.float32)
                self.activation = self.output[:,-1,:]
        return self.activation
    