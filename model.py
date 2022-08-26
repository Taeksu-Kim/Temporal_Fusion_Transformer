import torch
import torch.nn as nn

from utils.loss import QuantileLossCalculator, tensorflow_quantile_loss

concat = torch.concat

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False, activation=None):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.activation = activation

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        if self.activation != None:
          y = nn.Sigmoid()(y)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class static_combine_and_mask(nn.Module):
    def __init__(self, config):
        super(static_combine_and_mask, self).__init__()
        self.config = config
        self.num_static = len(self.config['static_input_loc'])
        self.mlp_grn = gated_residual_network(self.config, self.config.hidden_layer_size, output_size=self.num_static, use_time_distributed=False, additional_input_size=None)
        self.emb_grn = gated_residual_network(self.config, self.config.hidden_layer_size, output_size=None, use_time_distributed=False)

    def forward(self, embedding):
        """Applies variable selection network to static inputs.
        Args:
          embedding: Transformed static inputs
        Returns:
          Tensor output for variable selection network
        """
        
        # Add temporal features
        flatten = torch.flatten(embedding, start_dim=1, end_dim=- 1)
        
        # Nonlinear transformation with gated residual network.
        mlp_outputs = self.mlp_grn(flatten, additional_context=None)

        sparse_weights = torch.nn.Softmax(dim=-1)(mlp_outputs) # softmax_dim check 필요
        sparse_weights = torch.unsqueeze(sparse_weights, dim=-1)

        trans_emb_list = []
        for i in range(self.num_static):
            e = self.emb_grn(embedding[:, i:i + 1, :])
            trans_emb_list.append(e)

        transformed_embedding = concat(trans_emb_list, dim=1)

        combined = sparse_weights * transformed_embedding

        static_vec = torch.sum(combined, dim=1)

        return static_vec, sparse_weights

def Add(inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output

class add_and_norm(nn.Module):
    def __init__(self, norm_size):
        super(add_and_norm, self).__init__()
        self.layer_norm = nn.LayerNorm(norm_size)

    def forward(self, x_list):
        """Applies skip connection followed by layer normalisation.
        Args:
          x_list: List of inputs to sum for skip connection
        Returns:
          Tensor output from layer.
        """
        tmp = Add(x_list)
        tmp = self.layer_norm(tmp)
        
        return tmp

class gated_residual_network(nn.Module):
    def __init__(self, config, input_size, output_size=None, use_time_distributed=True, additional_input_size=None):
        super(gated_residual_network, self).__init__()
        
        self.config = config

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = self.config['hidden_layer_size']

        self.linear_skip = False
        if self.output_size is None:
            self.linear_skip = True
            self.output_size = self.hidden_layer_size

        self.additional_input_size = additional_input_size
        self.use_time_distributed = use_time_distributed

        self.dropout_rate = self.config['dropout_rate']
        
        # linear layers
        self.skip_linear = nn.Linear(self.input_size, self.output_size)
        if  self.use_time_distributed == True:
            self.skip_linear = TimeDistributed(self.skip_linear)
        
        self.pre_act_linear =  nn.Linear(self.input_size, self.hidden_layer_size)
        if  self.use_time_distributed == True:
            self.pre_act_linear = TimeDistributed(self.pre_act_linear)

        self.additional_input_size = additional_input_size
        if self.additional_input_size != None:
            self.add_linear = nn.Linear(self.additional_input_size, self.hidden_layer_size, bias=False)
            if  self.use_time_distributed == True:
                self.add_linear = TimeDistributed(self.add_linear)

        self.post_act_linear = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        if  self.use_time_distributed == True:
             self.post_act_linear = TimeDistributed(self.post_act_linear)

        self.apply_gating_layer = apply_gating_layer(self.config, self.output_size, self.use_time_distributed, None)

        # add_and_norm
        self.add_and_norm = add_and_norm([self.output_size])


    def forward(self, x, additional_context=None, return_gate=False):
        """Applies the gated residual network (GRN) as defined in paper.
        Args:
          x: Network inputs
          hidden_layer_size: Internal state size
          output_size: Size of output layer
          dropout_rate: Dropout rate if dropout is applied
          use_time_distributed: Whether to apply network across time dimension
          additional_context: Additional context vector to use if relevant
          return_gate: Whether to return GLU gate for diagnostic purposes
        Returns:
          Tuple of tensors for: (GRN output, GLU gate)
        """

        # Setup skip connection
        if self.linear_skip == True:
            skip = x
        else:
            skip = self.skip_linear(x)
        
        # Apply feedforward network
        hidden = self.pre_act_linear(x)

        if additional_context is not None:
              hidden = hidden + self.add_linear(additional_context)
              
        hidden = nn.ELU()(hidden)

        hidden = self.post_act_linear(hidden)

        gating_layer, gate = self.apply_gating_layer(hidden)

        # print(skip.shape, gating_layer.shape)

        if return_gate:
            return self.add_and_norm([skip, gating_layer]), gate
        else:
            return self.add_and_norm([skip, gating_layer])

class apply_gating_layer(nn.Module):
    def __init__(self, config, output_size, use_time_distributed=True, activation=None):
        super(apply_gating_layer, self).__init__()

        self.config = config
        self.output_size = output_size
        self.dropout_rate = self.config['dropout_rate']
        self.use_time_distributed = use_time_distributed

        if use_time_distributed == True:
            self.activation_linear = TimeDistributed(nn.Linear(self.config['hidden_layer_size'], self.output_size))
            self.gated_linear = TimeDistributed(nn.Linear(self.config['hidden_layer_size'], self.output_size), activation='sigmoid')
        else:
            self.activation_linear = nn.Linear(self.config['hidden_layer_size'], self.output_size)
            self.gated_linear = nn.Linear(self.config['hidden_layer_size'], self.output_size)

    def forward(self, x):
        """Applies a Gated Linear Unit (GLU) to an input.
        Args:
          x: Input to gating layer
          hidden_layer_size: Dimension of GLU
          dropout_rate: Dropout rate to apply if any
          use_time_distributed: Whether to apply across time
          activation: Activation function to apply to the linear feature transform if
            necessary
        Returns:
          Tuple of tensors for: (GLU output, gate)
        """

        if self.dropout_rate is not None:
            x = nn.Dropout(self.dropout_rate)(x)

        if self.use_time_distributed:
            activation_layer = self.activation_linear(x)
            gated_layer = self.gated_linear(x)

        else:
            activation_layer = self.activation_linear(x)
            gated_layer = self.gated_linear(x)
            gated_layer = nn.Sigmoid()(gated_layer)

        return activation_layer * gated_layer, gated_layer

class get_tft_embedding(nn.Module):
    def __init__(self, config):
        super(get_tft_embedding, self).__init__()
        self.config = config
          
        self.num_categorical_variables = len(self.config['category_counts'])
        self.num_regular_variables = self.config['input_size'] - self.num_categorical_variables

        self.embedding_sizes = [self.config['hidden_layer_size'] for i, size in enumerate(self.config['category_counts'])]

        self.embeddings =  nn.ModuleList([])
        for i in range(self.num_categorical_variables):
          embedding = torch.nn.Embedding(self.config['category_counts'][i], self.embedding_sizes[i])
          self.embeddings.append(embedding)

        self.static_linear = nn.Linear(1, self.config['hidden_layer_size'])
        self.obs_liner = TimeDistributed(nn.Linear(1, self.config['hidden_layer_size']))
        self.unknown_liner = TimeDistributed(nn.Linear(1, self.config['hidden_layer_size']))
        self.known_liner = TimeDistributed(nn.Linear(1, self.config['hidden_layer_size']))
       
    def forward(self, all_inputs):
        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :self.num_regular_variables], \
              all_inputs[:, :, self.num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[Ellipsis, i].long())
            for i in range(self.num_categorical_variables)
        ]

        # Static inputs
        if self.config['static_input_loc']:
          
          static_linear_result = []
          for i in range(self.num_regular_variables):
            if i in self.config['static_input_loc']:
              static_linear_result.append(self.static_linear(regular_inputs[:, 0, i:i + 1].float()))
   
          static_emb_result = []
          for i in range(self.num_categorical_variables):
            if i + self.num_regular_variables in self.config['static_input_loc']:
              static_emb_result.append(embedded_inputs[i][:, 0, :])
        
          static_inputs = static_linear_result + static_emb_result
          static_inputs = torch.stack(static_inputs, axis=1)

        else:
          static_inputs = None

        # Targets        
        obs_inputs = []
        for i in self.config['input_obs_loc']:
          obs_inputs.append(self.obs_liner(regular_inputs[Ellipsis, i:i + 1].float()))

        obs_inputs = torch.stack(obs_inputs, axis=-1)

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(self.num_categorical_variables):
          if i not in self.config['known_categorical_inputs'] \
            and  i + self.num_regular_variables  not in self.config['input_obs_loc']:
            e = self.embeddings[i](categorical_inputs[:, :, i])
            wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
          if i not in self.config['known_regular_inputs'] \
              and i not in self.config['input_obs_loc']:
            e = self.unknown_liner(regular_inputs[Ellipsis, i:i + 1].float())
            unknown_inputs.append(e)
        
        if unknown_inputs + wired_embeddings:
          unknown_inputs = torch.stack(
              unknown_inputs + wired_embeddings, axis=-1)
        else:
          unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            self.known_liner(regular_inputs[Ellipsis, i:i + 1].float())
            for i in self.config['known_regular_inputs']
            if i not in self.config['static_input_loc']
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self.config['known_categorical_inputs']
            if i + self.num_regular_variables not in self.config['static_input_loc']
        ]

        known_combined_layer = torch.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

class lstm_combine_and_mask(nn.Module):
    def __init__(self, config, time_steps, num_inputs):
        super(lstm_combine_and_mask, self).__init__()
        self.config = config

        self.time_steps = time_steps
        self.embedding_dim = self.config['hidden_layer_size']
        self.num_inputs = num_inputs

        self.mlp_grn = gated_residual_network(self.config, self.embedding_dim * self.num_inputs, output_size=self.num_inputs, use_time_distributed=True, additional_input_size=self.config['hidden_layer_size'])


        self.trans_emb_grn_list =  nn.ModuleList([])
        for i in range(self.num_inputs):
            trans_emb_grn = gated_residual_network(self.config, self.config['hidden_layer_size'], use_time_distributed=True)
            self.trans_emb_grn_list.append(trans_emb_grn)

    def forward(self, embedding, static_context_variable_selection):
        """Apply temporal variable selection networks.
        Args:
          embedding: Transformed inputs.
        Returns:
          Processed tensor outputs.
        """

        # Add temporal features

        flatten = embedding.view([-1, self.time_steps, self.embedding_dim * self.num_inputs])

        expanded_static_context = torch.unsqueeze(static_context_variable_selection, axis=1)

        # Variable selection weights
        mlp_outputs, static_gate = self.mlp_grn(flatten,
                                                additional_context=expanded_static_context,
                                                return_gate=True)

        sparse_weights = torch.nn.Softmax(dim=-1)(mlp_outputs)
        sparse_weights = torch.unsqueeze(sparse_weights, axis=2)

        # Non-linear Processing & weight application
        trans_emb_list = []
        for i in range(self.num_inputs):            
            grn_output = self.trans_emb_grn_list[i](embedding[Ellipsis, i])
            trans_emb_list.append(grn_output)

        transformed_embedding = torch.stack(trans_emb_list, axis=-1)

        combined = sparse_weights * transformed_embedding
        temporal_ctx = torch.sum(combined, dim=-1)

        return temporal_ctx, sparse_weights, static_gate

class InterpretableMultiHeadAttention(nn.Module):
  """Defines interpretable multi-head attention layer.
  Attributes:
    n_head: Number of heads
    d_k: Key/query dimensionality per head
    d_v: Value dimensionality
    dropout: Dropout rate to apply
    qs_layers: List of queries across heads
    ks_layers: List of keys across heads
    vs_layers: List of values across heads
    attention: Scaled dot product attention layer
    w_o: Output weight matrix to project internal state to the original TFT
      state size
  """

  def __init__(self, n_head, d_model, dropout, hidden_layer_size):
    """Initialises layer.
    Args:
      n_head: Number of heads
      d_model: TFT state dimensionality
      dropout: Dropout discard rate
    """
    super().__init__()
    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout
    self.hidden_layer_size = hidden_layer_size

    self.qs_layers = nn.ModuleList([])
    self.ks_layers = nn.ModuleList([])
    self.vs_layers = nn.ModuleList([])

    # Use same value layer to facilitate interp
    vs_layer = nn.Linear(self.hidden_layer_size, d_v, bias=False)

    for _ in range(n_head):
      self.qs_layers.append(nn.Linear(self.hidden_layer_size, d_k, bias=False))
      self.ks_layers.append(nn.Linear(self.hidden_layer_size, d_k, bias=False))
      self.vs_layers.append(vs_layer)  # use same vs_layer

    self.attention = ScaledDotProductAttention()
    self.w_o = nn.Linear(self.hidden_layer_size, d_model, bias=False)

  def forward(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.
    Using T to denote the number of time steps fed into the transformer.
    Args:
      q: Query tensor of shape=(?, T, d_model)
      k: Key of shape=(?, T, d_model)
      v: Values of shape=(?, T, d_model)
      mask: Masking if required with shape=(?, T, T)
    Returns:
      Tuple of (layer outputs, attention weights)
    """
    n_head = self.n_head

    heads = []
    attns = []

    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask)
      head_dropout = nn.Dropout(self.dropout)(head)
      heads.append(head_dropout)
      attns.append(attn)
    head = torch.stack(heads) if n_head > 1 else heads[0]
    attn = torch.stack(attns)

    outputs = torch.mean(head, axis=0) if n_head > 1 else head
    outputs = self.w_o(outputs)
    outputs = nn.Dropout(self.dropout)(outputs)  # output dropout

    return outputs, attn

class ScaledDotProductAttention(nn.Module):
  """Defines scaled dot product attention layer.
  Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
  """

  def __init__(self, attn_dropout=0.0):
    super().__init__()
    self.dropout = nn.Dropout(attn_dropout)
    self.activation = torch.nn.Softmax(dim=-1)

  def forward(self, q, k, v, mask):
    """Applies scaled dot product attention.
    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value
    Returns:
      Tuple of (layer outputs, attention weights)
    """
    temper = torch.sqrt(torch.tensor(k.size()[-1]).float())
    
    attn = torch.div(torch.matmul(q,k.transpose(1,2)), temper)
    
    if mask is not None:
      mmask = (1 - mask) * (-1e+9)
      attn = attn + mmask.to(attn.device)

    attn = self.activation(attn)
    attn = self.dropout(attn)

    output = torch.matmul(attn, v)

    return output, attn

# Attention Components.
def get_decoder_mask(self_attn_inputs):
  """Returns causal mask to apply for self-attention layer.
  Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
  """
  len_s = self_attn_inputs.shape[1]
  bs = self_attn_inputs.shape[0]
  torch.eye(len_s).expand([bs,-1,-1])
  mask = torch.cumsum(torch.eye(len_s).expand([bs,-1,-1]), dim=1)
  return mask

class Temporal_Fusion_Transformer(nn.Module):
    def __init__(self, config):
        super(Temporal_Fusion_Transformer, self).__init__()
        self.config = config
        
        #Sanity checks
        for i in self.config['known_regular_inputs']:
            if i in self.config['input_obs_loc']:
                raise ValueError('Observation cannot be known a priori!')
        for i in self.config['input_obs_loc']:
            if i in self.config['static_input_loc']:
                raise ValueError('Observation cannot be static!')

        self.get_tft_embedding = get_tft_embedding(self.config)
        self.static_combine_and_mask = static_combine_and_mask(self.config)

        self.static_variable_selection_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=False)
        self.static_enrichment_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=False)
        self.static_state_h_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=False)
        self.static_state_c_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=False)

        # for lstm_combine_and_mask
        self.regular_inputs_len = self.config['input_size'] - len(self.config['category_counts'])
        self.categorical_inputs_len = len(self.config['category_counts'])
        self.static_inputs_len = len([1 for i in range(self.regular_inputs_len) if i in self.config['static_input_loc']] + \
                            [1 for i in range(self.categorical_inputs_len)  if i + self.regular_inputs_len in self.config['static_input_loc']])
        self.known_combined_inputs_len = len([1 for i in self.config['known_regular_inputs'] if i not in self.config['static_input_loc']] + \
                                    [1 for i in self.config['known_categorical_inputs'] if i + self.regular_inputs_len not in self.config['static_input_loc']])
        self.unknown_combined_inputs_len = len([1 for i in range(self.categorical_inputs_len) if i not in self.config['known_categorical_inputs'] and  i + num_regular_variables  not in self.config['input_obs_loc']] + \
                                          [1 for i in range(self.regular_inputs_len) if i not in self.config['known_regular_inputs'] and i not in self.config['input_obs_loc']])
        self.historical_lstm_combine_and_mask = lstm_combine_and_mask(self.config, 
                                                                      time_steps=self.config['num_encoder_steps'], 
                                                                      num_inputs= self.static_inputs_len + self.known_combined_inputs_len + self.unknown_combined_inputs_len)
        self.future_lstm_combine_and_mask = lstm_combine_and_mask(self.config, 
                                                                  time_steps=self.config['total_time_steps'] - self.config['num_encoder_steps'], 
                                                                  num_inputs=self.known_combined_inputs_len)

        self.historical_lstm = torch.nn.LSTM(input_size=self.config.hidden_layer_size, hidden_size=self.config.hidden_layer_size, num_layers=self.config.lstm_num_layers, batch_first=True)
        self.future_lstm = torch.nn.LSTM(input_size=self.config.hidden_layer_size, hidden_size=self.config.hidden_layer_size, num_layers=self.config.lstm_num_layers, batch_first=True)

        self.lstm_apply_gating_layer = apply_gating_layer(self.config, self.config.hidden_layer_size, activation=None)
        self.lstm_add_and_norm = add_and_norm([self.config.hidden_layer_size])

        self.static_enrichment_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=True, additional_input_size=self.config.hidden_layer_size)

        # for decoder
        self.self_attn_layer = InterpretableMultiHeadAttention(self.config.num_heads, self.config.hidden_layer_size, dropout=self.config.dropout_rate, hidden_layer_size=self.config.hidden_layer_size)
        self.decoder_apply_gating_layer = apply_gating_layer(self.config, self.config.hidden_layer_size, activation=None)
        self.decoder_add_and_norm = add_and_norm([self.config.hidden_layer_size])
        self.decoder_grn = gated_residual_network(self.config, self.config.hidden_layer_size, use_time_distributed=True)

        self.final_apply_gating_layer = apply_gating_layer(self.config, self.config.hidden_layer_size, activation=None)
        self.final_add_and_norm = add_and_norm([self.config.hidden_layer_size])

        self.output_linear = TimeDistributed(nn.Linear(self.config.hidden_layer_size, self.config.output_size * len(self.config.quantiles)))

    def forward(self, inputs, labels=None):
        
        #Sanity checks
        if inputs.size()[-1] != self.config['input_size']:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    inputs.size()[-1], self.config['input_size']))

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embedding(inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
          historical_inputs = concat([unknown_inputs[:, :self.config.num_encoder_steps, :],
                                      known_combined_layer[:, :self.config.num_encoder_steps, :],
                                      obs_inputs[:, :self.config.num_encoder_steps, :]], axis=-1)
        else:
          historical_inputs = concat([known_combined_layer[:, :self.config.num_encoder_steps, :],
              obs_inputs[:, :self.config.num_encoder_steps, :]], axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, self.config.num_encoder_steps:, :]

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)

        static_context_variable_selection = self.static_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_enrichment_grn(static_encoder)
        static_context_state_h = self.static_state_h_grn(static_encoder)
        static_context_state_c = self.static_state_c_grn(static_encoder)
        
        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)

        historical_lstm_result = self.historical_lstm(historical_features,(static_context_state_h.expand(self.config.lstm_num_layers, -1, -1), static_context_state_c.expand(self.config.lstm_num_layers, -1, -1)))
        history_lstm, state_h, state_c = historical_lstm_result[0], historical_lstm_result[1][0], historical_lstm_result[1][1]
        future_lstm = self.future_lstm(future_features, (state_h, state_c))[0]

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection
        input_embeddings = concat([historical_features, future_features], axis=1)

        lstm_layer, _ = self.lstm_apply_gating_layer(lstm_layer)
        temporal_feature_layer = self.lstm_add_and_norm([lstm_layer, input_embeddings])

        # Static enrichment layers
        expanded_static_context = torch.unsqueeze(static_context_enrichment, dim=1)
        
        enriched, _ = self.static_enrichment_grn(temporal_feature_layer,
                                             additional_context=expanded_static_context,
                                             return_gate=True)        

        # Decoder self attention
        mask = get_decoder_mask(enriched)

        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask=mask)

        x, _ = self.decoder_apply_gating_layer(x)
        x = self.decoder_add_and_norm([x, enriched])

        # Nonlinear processing on outputs
        decoder = self.decoder_grn(x)
        
        # Final skip connection
        decoder, _ = self.final_apply_gating_layer(decoder)
        transformer_layer = self.final_add_and_norm([decoder, temporal_feature_layer])

        outputs = self.output_linear(transformer_layer[Ellipsis, self.config.num_encoder_steps:, :])

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }
        loss = None
        if labels != None:
            quantile_loss = QuantileLossCalculator(self.config).quantile_loss
            loss = quantile_loss(labels.expand(-1,-1,len(self.config.quantiles)), outputs)


        return_dict = {
            'outputs' : outputs,
            'loss' : loss,
            'attention_components': attention_components,
        }

        return return_dict