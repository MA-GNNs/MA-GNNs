# !/usr/local/bin/bash
'''
Created on Oct 10, 2019
Tensorflow Implementation of Disentangled Graph Collaborative Filtering (DGCF) model in:
Wang Xiang et al. Disentangled Graph Collaborative Filtering. In SIGIR 2020.
Note that: This implementation is based on the codes of NGCF.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np

epsilon = 1e-11


class BHGAT(object):
    def __init__(self, data_config, pretrain_data, args):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_nodes = self.n_users + self.n_items
        self.norm_adj = data_config['norm_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.norm_adj.tocoo().shape

        self.lr = args.lr
        self.emb_dim = args.embed_size   #64
        self.n_factors = args.n_factors  #4
        self.n_iterations = args.n_iterations   #图中的路由迭代次数2
        self.n_layers = args.n_layers  #高阶连通性的层数
        self.pick_level = args.pick_scale
        self.cor_flag = args.cor_flag

        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        #胶囊的参数
        self.inputs_CapsLayer_size = self.n_factors
        self.outputs_CapsLayer_size=1
        self.iter_routing = args.iter_routing

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tfv1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tfv1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tfv1.placeholder(tf.int32, shape=(None,))

        # additional placeholders for the distance correlation
        self.cor_users = tfv1.placeholder(tf.int32, shape=(None,))
        self.cor_items = tfv1.placeholder(tf.int32, shape=(None,))

        # assign different values with different factors (channels).
        # self.A_values = tfv1.placeholder(tf.float32, shape=[self.n_factors, len(self.all_h_list)], name='A_values')

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        # create models
        # 最重要的部分！！！！！！！！！！！！！
        """
        *********************************************************
        Create Models Parameters! The most important part!
        """
        # self.ua_embeddings, self.ia_embeddings, self.f_weight, self.ua_embeddings_t, self.ia_embeddings_t = self._create_star_routing_embed_with_P(
        # pick_=self.is_pick)
        self.all_f_embeddings, self.f_weight, self.all_f_embeddings_t = self._create_star_routing_embed_with_P(
            pick_=self.is_pick)
        self.ua_embeddings, self.ia_embeddings = self._capsule_2NE_aspect_fusion()

        self.ua_embeddings_t, self.ia_embeddings_t = self.ua_embeddings, self.ia_embeddings
        # self.ua_embeddings, self.ia_embeddings = tf.split(self.all_f_embeddings, [self.n_users, self.n_items], 0)
        # self.ua_embeddings_t, self.ia_embeddings_t = tf.split(self.all_f_embeddings, [self.n_users, self.n_items], 0)
        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)

        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # pre是用来计算正则化的
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        self.cor_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.cor_users)
        self.cor_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.cor_items)

        # Inference for the testing phase.
        self.batch_ratings = tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t, transpose_a=False,
                                       transpose_b=True)

        # Generate Predictions & Optimize via BPR loss.
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)

        # whether user distance correlation
        if args.corDecay < 1e-9:
            self.cor_loss = tf.constant(0.0)
        else:
            self.cor_loss = args.corDecay * self.create_cor_loss(self.cor_u_g_embeddings, self.cor_i_g_embeddings)

            # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.loss = self.mf_loss + self.emb_loss + self.cor_loss
        self.opt = tfv1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),trainable=True,
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),trainable=True,
                                                        name='item_embedding')
            all_weights['A_features'] = tf.Variable(initializer([len(self.all_h_list), self.emb_dim]),trainable=True,
                                                    name='A_features')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            all_weights['A_features'] = tf.Variable(initial_value=self.pretrain_data['A_fea'], trainable=True,
                                                    name='A_features', dtype=tf.float32)
            print('using pretrained initialization')

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.emb_dim // self.n_factors, self.emb_dim // self.n_factors]), name='W_gc_%d' % k)

            all_weights['W_a_%d' % k] = tf.Variable(
                initializer([3 * (self.emb_dim // self.n_factors), 1]), name='W_a_%d' % k)

        return all_weights

    def _create_star_routing_embed_with_P(self, pick_=False):
        '''
        pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
        pick_ : False, do nothing.
        '''
        p_test = False
        p_train = False


        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        all_embeddings_t = [ego_embeddings]
        A_features = self.weights['A_features']

        output_factors_distribution = []

        factor_num = [self.n_factors, self.n_factors, self.n_factors, self.n_factors]
        iter_num = [self.n_iterations, self.n_iterations, self.n_iterations, self.n_iterations]
        for k in range(0, self.n_layers):
            # prepare the output embedding list
            # .... layer_embeddings stores a (n_factors)-len list of outputs derived from the last routing iterations.
            n_factors_l = factor_num[k]
            n_iterations_l = iter_num[k]
            layer_embeddings = []
            layer_embeddings_t = []

            # split the input embedding table  切片得到胶囊
            # .... ego_layer_embeddings is a (n_factors)-leng list of embeddings [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = tf.split(ego_embeddings, n_factors_l, 1)
            ego_layer_embeddings_t = tf.split(ego_embeddings, n_factors_l, 1)

            # perform routing mechanism   先更新嵌入再更新图
            for t in range(0, n_iterations_l):
                iter_embeddings = []
                iter_embeddings_t = []
                A_iter_features = []

                # split the adjacency values & get three lists of [n_users+n_items, n_users+n_items] sparse tensors
                # .... A_factors is a (n_factors)-len list, each of which is an adjacency matrix
                # .... D_col_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. columns
                # .... D_row_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. rows
                if t == n_iterations_l - 1:
                    p_test = pick_
                    p_train = False

                A_factors = A_factors_t = tf.split(A_features, n_factors_l, 1)
                for i in range(0, n_factors_l):

                    # 采用注意力机制更新嵌入
                    factor_embeddings = self._attentional_propagation_one_intent(ego_layer_embeddings[i], A_factors[i],
                                                                                 k)
                    factor_embeddings_t = self._attentional_propagation_one_intent(ego_layer_embeddings_t[i],
                                                                                   A_factors_t[i], k)

                    # 以上是 拉普拉斯矩阵*E（嵌入矩阵）  得到更新后的矩阵表示
                    iter_embeddings.append(factor_embeddings)
                    iter_embeddings_t.append(factor_embeddings_t)

                    if t == n_iterations_l - 1:
                        layer_embeddings = iter_embeddings
                        layer_embeddings_t = iter_embeddings_t

                    # 这一部分是更新图
                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings

                    head_factor_embedings = tf.nn.embedding_lookup(factor_embeddings, self.all_h_list)  # 行
                    tail_factor_embedings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)  # 列

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_factor_embedings = tf.math.l2_normalize(head_factor_embedings, axis=1)
                    tail_factor_embedings = tf.math.l2_normalize(tail_factor_embedings, axis=1)

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [all_h_list,1]
                    A_factor_values = tf.multiply(head_factor_embedings, tf.tanh(tail_factor_embedings))

                    # tail_factor_embedings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)  # 列
                    A_factor_feature = tf.multiply(A_factor_values, head_factor_embedings)
                    # update the attentive weights
                    A_iter_features.append(A_factor_feature)

                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                A_iter_features = tf.concat(A_iter_features, 1)
                # add all layer-wise attentive weights up.
                A_features += A_iter_features

                if t == n_iterations_l - 1:
                    # layer_embeddings = iter_embeddings
                    output_factors_distribution.append(A_factors)

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = tf.concat(layer_embeddings, 1)
            side_embeddings_t = tf.concat(layer_embeddings_t, 1)

            ego_embeddings = side_embeddings
            ego_embeddings_t = side_embeddings_t
            # concatenate outputs of all layers
            all_embeddings_t += [ego_embeddings_t]
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        all_embeddings_t = tf.stack(all_embeddings_t, 1)
        all_embeddings_t = tf.reduce_mean(all_embeddings_t, axis=1, keep_dims=False)
        return all_embeddings, output_factors_distribution, all_embeddings_t


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        #         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        #         mf_loss = tf.negative(tf.reduce_mean(maxi))

        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))  # 即实现sigmoid函数

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = tf.constant(0.0, tf.float32)

        if self.cor_flag == 0:
            return cor_loss

        ui_embeddings = tf.concat([cor_u_embeddings, cor_i_embeddings], axis=0)
        ui_factor_embeddings = tf.split(ui_embeddings, self.n_factors, 1)  # 维度为1 切四份

        for i in range(0, self.n_factors - 1):  # 分四个胶囊进行计算
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i + 1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        return cor_loss

    def model_save(self, path, dataset, ses, savename='best_model'):
        save_pretrain_path = '%spretrain/%s/%s' % (path, dataset, savename)
        np.savez(save_pretrain_path, user_embed=np.array(self.weights['user_embedding'].eval(session=ses)),
                 item_embed=np.array(self.weights['item_embedding'].eval(session=ses)))

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples.
                (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)

            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = tf.dtypes.cast(tf.shape(D1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def _convert_A_values_to_A_factors_with_P(self, f_num, A_factor_values, pick=True):

        A_factors = []
        D_col_factors = []  # 列
        D_row_factors = []  # 行
        # get the indices of adjacency matrix.
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()  # 得到邻接矩阵的下标
        D_indices = np.mat(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))]).transpose()

        # apply factor-aware softmax function over the values of adjacency matrix
        # .... A_factor_values is [n_factors, all_h_list]
        if pick:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)
            min_A = tf.reduce_min(A_factor_scores, 0)
            index = A_factor_scores > (min_A + 0.0000001)
            index = tf.cast(index, tf.float32) * (
                    self.pick_level - 1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / tf.reduce_sum(A_factor_scores, 0)
        else:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)

        for i in range(0, f_num):
            # in the i-th factor, couple the adjacency values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            A_i_scores = A_factor_scores[i]
            A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, self.A_in_shape)

            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1 / tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=1))
            D_i_row_scores = 1 / tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=0))

            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, self.A_in_shape)
            D_i_row_tensor = tf.SparseTensor(D_indices, D_i_row_scores, self.A_in_shape)

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors

    def _attentional_propagation_one_intent(self, ego_embedding_one_intent, A_factors_one_intent, k_num):
        N = len(self.all_h_list)
        # 下面是邻接矩阵的压缩存储得到的  行和列
        head_factor_embedings = tf.nn.embedding_lookup(ego_embedding_one_intent, self.all_h_list)  # 行
        tail_factor_embedings = tf.nn.embedding_lookup(ego_embedding_one_intent, self.all_t_list)  # 列

        att_input = tf.reshape(
            tf.concat([head_factor_embedings, tail_factor_embedings, A_factors_one_intent], axis=1), [N, -1])
        attention = tf.squeeze(tf.nn.leaky_relu(tf.matmul(att_input, self.weights['W_a_%d' % k_num])), [1])

        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        att_tensor = tf.SparseTensor(A_indices, attention, self.A_in_shape)
        att_weight = tf.sparse.softmax(att_tensor)

        factor_embedding_one_intent = tf.sparse.sparse_dense_matmul(att_weight, ego_embedding_one_intent)

        norm_factor_embedding = tf.nn.l2_normalize(factor_embedding_one_intent, axis=1)

        """
        att_input_1 = tf.reshape(
            tf.concat([head_factor_embedings, tail_factor_embedings, A_factors_one_intent], axis=1), [N, -1])
        att_input_2 = tf.reshape(tf.concat([head_factor_embedings, tail_factor_embedings], axis=1), [N, -1])

        attention_1 = tf.squeeze(tf.nn.leaky_relu(tf.matmul(att_input_1, self.weights['W_a_%d' % k_num])), [1])
        attention_2 = tf.squeeze(tf.nn.leaky_relu(tf.matmul(att_input_2, self.weights['W_b_%d' % k_num])), [1])

        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()

        att_tensor_1 = tf.SparseTensor(A_indices, attention_1, self.A_in_shape)
        att_weight_1 = tf.sparse_softmax(att_tensor_1)
        factor_embedding_one_intent_1 = tf.sparse.sparse_dense_matmul(att_weight_1, ego_embedding_one_intent)

        att_tensor_2 = tf.SparseTensor(A_indices, attention_2, self.A_in_shape)
        att_weight_2 = tf.sparse_softmax(att_tensor_2)
        factor_embedding_one_intent_2 = tf.sparse.sparse_dense_matmul(att_weight_2, A_factors_one_intent)

        norm_factor_embedding = tf.nn.l2_normalize(factor_embedding_one_intent_1, axis=1)+tf.nn.l2_normalize(factor_embedding_one_intent_2, axis=1)
        """

        return norm_factor_embedding

    def _capsule_2NE_aspect_fusion(self, ):
        all_f_embeddings = tf.reshape(self.all_f_embeddings,
                                      shape=(self.all_f_embeddings.shape[0].value, self.n_factors, -1))
        input = tf.expand_dims(all_f_embeddings, -1)

        with tf.compat.v1.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1],
            b_IJ = tf.constant(np.ones([self.n_users + self.n_items, input.shape[1].value, self.outputs_CapsLayer_size, 1],
                                        dtype=np.float32))
            #b_IJ = tf.divide(b_IJ,4)
            self.capsules = self.routing(input, b_IJ, batch_size=self.n_users + self.n_items, iter_routing=self.iter_routing,
                               num_caps_i=self.inputs_CapsLayer_size, num_caps_j=self.outputs_CapsLayer_size, len_v_j=self.emb_dim)
            #self.caps = tf.squeeze(capsules, axis=1)

        #self.node_fea = tf.concat([self.all_f_embeddings, self.capsules], axis=-1)
        u_g_embeddings, i_g_embeddings = tf.split(self.capsules, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def routing(self, input, b_IJ, batch_size, iter_routing, num_caps_i, num_caps_j, len_v_j):
        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(input, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                c_IJ = tf.nn.softmax(b_IJ, axis=1)* num_caps_i  #axis=1 # original code

                if r_iter == iter_routing - 1:
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    s_J = tf.multiply(c_IJ, input)
                    self.c_IJ = c_IJ
                    # then concat them, resulting in [batch_size, 1, num_caps_j, len_v_j, 1]
                    s_J = tf.reshape(s_J, shape=(batch_size, len_v_j))

                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    #s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                    v_J = self.squash(s_J)
                    # line 7:
                    #v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1])
                    b_IJ = tf.matmul(u_hat_stopped, v_J, transpose_a=True)

        return(s_J)

    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)
