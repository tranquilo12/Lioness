class RUSBoost(object):
    """docstring for RUSBoost"""

    def __init__(self, base_classifier, N, x_train, y_train, x_test, step_size, loss):
        super(RUSBoost, self).__init__()
        self.base_classifier = tree.DecisionTreeClassifier()
        self.N = N
        self.loss = loss
        self.w_update = []
        self.clf = []
        for i in range(self.N):
            self.clf.append(self.base_classifier)
        self.step_size = step_size
        self.x = x_train
        self.y = y_train
        self.x_test = x_test
        self.weight_list = self.make_default_weights(self.x)
        
    def make_default_weights(self, df):
        l = []
        for i in range(df.shape[0]):
            l.append(1.0 / len(df))
        w = pd.Series(l, index=df.index)
        return(w)
    
    def classify(self):
        pred = {}
        for k in range(self.N):
            pred[k] = self.clf[k].predict(self.x_test)
        return(pred)
    
    def undersampling(self): 
        df = self.y
        one_list = df[df['YClassH28'] == 1].index.tolist()
        zero_list = df[df['YClassH28'] == 0].index.tolist()
        neg_list = df[df['YClassH28'] == -1].index.tolist()
        
        least = [len(one_list), len(zero_list), len(neg_list)]
        least = min(least)
        
        new_num_of_ones = abs(len(one_list) - least)
        new_num_of_zeros = abs(len(zero_list) - least)
        new_num_of_neg = abs(len(neg_list) - least)
        
        if new_num_of_ones == 0:
            while (len(zero_list) - least) >= 0: 
                k = random.choice(range(len(zero_list)))
                zero_list.pop(k)
            while (len(neg_list) - least) >= 0: 
                k = random.choice(range(len(neg_list)))
                neg_list.pop(k)
        
        if new_num_of_zeros == 0:
            while (len(one_list) - least) >= 0: 
                k = random.choice(range(len(one_list)))
                one_list.pop(k)
            while (len(neg_list) - least) >= 0: 
                k = random.choice(range(len(neg_list)))
                neg_list.pop(k)
        
        if new_num_of_neg == 0:
            while (len(one_list) - least) >= 0: 
                k = random.choice(range(len(one_list)))
                one_list.pop(k)
            while (len(zero_list) - least) >= 0: 
                k = random.choice(range(len(neg_list)))
                neg_list.pop(k)
        
        all_list = one_list + zero_list + neg_list
        return(sorted(all_list, key=lambda x: df.all().index))

    def learner(self):
        k = 0
        print('\n')
        for k in range(self.N):
            print(' Iteration %d: ... ' % k)
            
            # returns indices of all undersampled list
            sampled = self.undersampling()
            
            # get all x_sampled and y_sampled through the indices
            x_sampled = self.x.loc[sampled]
            y_sampled = self.y.loc[sampled]
            weight_list = self.weight_list[sampled]
            
            
            # fit using the decisiontree classifier, with an empty weights list first
            # train using 'self' model, so it remains consistant with ebery object instance
            self.clf[k].fit(X=x_sampled, 
                            y=y_sampled,
                            sample_weight=np.array(weight_list[:]), 
                            X_idx_sorted=None)
            
            # take each row, from x_sampled and try and predict it
            # x_sampled is used, as it correctly has the indexed rows that are changed
            
            i = 0
            for index, row in x_sampled.iterrows():
                # if it turns out to be a good prediction, then continue, 
                # redict with the self model... 
                if self.y['YClassH28'][index] == self.clf[k].predict(row):
                    continue
                # else, adjust loss factor, by dividing it by the number of samples
                # again. modify the object's instance of weights
                else:
                    self.loss = self.loss / (self.weight_list[index])
                
            # again, try iterating with the same list, rewarding the weight_list proportional to the
            # loss factor
            i = 0
            for index, row in x_sampled.iterrows():
                # if the loss factor has not changed, continue
                if self.loss == 0.1 or self.loss == 0:
                    continue
                # if the loss factor has changed, see if the row predicts well, if it does
                elif self.y['YClassH28'][index] == self.clf[k].predict(row):
                    # increase the weights, by the direct percentage representation of the loss factor
                    self.weight_list[index] = self.weight_list[index] * (self.loss / (1.0 - self.loss))
            
            sum_weight = 0
            for index, row in self.weight_list.iteritems():
                sum_weight += row

            for index, row in self.weight_list.iteritems():
                self.weight_list[index] = row / sum_weight