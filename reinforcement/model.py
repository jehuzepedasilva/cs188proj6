import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        # LOOK AT CODE FROM PROJECT 5 cs188proj5/machinelearning original/models.py, should be kinda similar
        # self.parameters = nn.Parameter()
        self.learning_rate = 0.5
        self.numTrainingGames = 4000
    
        self.hidden_size = 300
        self.batch_size = 512

        self.W1 = nn.Parameter(self.state_size, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b2 = nn.Parameter(1, self.hidden_size)
        self.W3 = nn.Parameter(self.hidden_size, self.num_actions)
        self.b3 = nn.Parameter(1, action_dim)
        self.set_weights([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
        # self.helpers = Helpers(self.learning_rate, nn.SquareLoss, self)


    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        return nn.SquareLoss(self.run(states), Q_target)
    

    def forward(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"

        # x = nn.AddBias(nn.Linear(states, self.W1), self.b1) 
        # x = nn.ReLU(x) 
        # return nn.AddBias(nn.Linear(x, self.W2), self.b2)

        x1 = nn.ReLU(nn.AddBias(nn.Linear(states, self.W1), self.b1))
        x2 = nn.ReLU(nn.AddBias(nn.Linear(x1, self.W2), self.b2))  
        return nn.AddBias(nn.Linear(x2, self.W3), self.b3)

    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        # q_values = self.run(states)
        #self.helpers.loss(states, Q_target)
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)
        self.W1.update(gradients[0], -self.learning_rate) 
        self.b1.update(gradients[1], -self.learning_rate) 
        self.W2.update(gradients[2], -self.learning_rate) 
        self.b2.update(gradients[3], -self.learning_rate)
        self.W3.update(gradients[4], -self.learning_rate) 
        self.b3.update(gradients[5], -self.learning_rate)
        

# class Helpers():
#     def __init__(self, learning_rate, loss_function, model):
#         self.learning_rate = learning_rate
#         self.loss_function = loss_function
#         self.model = model
        
#     def get_gradients(self, loss, weight_result=False):
#         if weight_result:
#             return  nn.gradients(loss, [self.model.W1, self.model.b1, self.model.W2, self.model.b2, self.model.result])
#         return nn.gradients(loss, [self.model.W1, self.model.b1, self.model.W2, self.model.b2])
        
#     def update_weights(self, x, label, weight_result=False):
#         loss = self.model.get_loss(x, label)
#         gradients = self.get_gradients(loss, weight_result)
#         self.model.W1.update(gradients[0], -self.learning_rate)
#         self.model.b1.update(gradients[1], -self.learning_rate)
#         self.model.W2.update(gradients[2], -self.learning_rate)
#         self.model.b2.update(gradients[3], -self.learning_rate)
#         if weight_result: 
#             self.model.result.update(gradients[4], -self.learning_rate)
#         return nn.as_scalar(loss)
        
#     def loss(self, x, y):
#         y_pred = self.model.run(x)
#         return self.loss_function(y_pred, y)
            
