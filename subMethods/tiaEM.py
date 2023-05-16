import numpy as np

class tiaEM():

    def __init__(self, answers, answers_p, batch_size=16, alpha_prior=1.0, beta_prior=1.0):
        # self.data_train = data_train
        self.answers = answers
        self.answers_p = answers_p
        self.batch_size = batch_size
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.n_train = answers.shape[0]
        self.num_classes = np.max(answers) + 1
        self.num_annotators = answers.shape[1]
        self.num_annotators_p = answers_p.shape[1]

        # initialize annotators as reliable (almost perfect)
        self.alpha = 0.5 * np.ones(self.num_annotators)
        self.beta = 0.5 * np.ones(self.num_annotators)
        self.alpha_p = 0.5 * np.ones(self.num_annotators_p)
        self.beta_p = 0.5 * np.ones(self.num_annotators_p)
        # self.votes = np.zeros(self.num_classes)

        # initialize estimated ground truth with majority voting
        self.ground_truth_est = np.zeros((self.n_train, 2))
        self.ground_truth_train = np.zeros((self.n_train, 1))
        for i in range(self.n_train):
            votes = np.zeros(self.num_classes)
            for r in range(self.num_annotators):
                if answers[i, r] != -1:
                    votes[self.answers[i, r]] += 1
            for r in range(self.num_annotators_p):
                if answers_p[i, r] != -1:
                    votes[self.answers_p[i, r]] += 1
            self.ground_truth_est[i, np.argmax(votes)] = np.max(votes)/np.sum(votes)
            self.ground_truth_est[i, 1-np.argmax(votes)] = 1-self.ground_truth_est[i, np.argmax(votes)]
        self.pi = np.sum(self.ground_truth_est, axis=0) / self.n_train

        # #恶意工人答案对初始化的影响
        # for i in range(self.n_train):#任务个数
        #     for r in range(self.num_annotators_p):
        #         if answers_p[i, r] != -1:
        #             self.votes[self.answers_p[i, r]] += 1
        # self.ground_truth_est[i, np.argmax(self.votes)] = 1


    def e_step(self):
        # print "E-step"
        # print "M-step"
        # hist = self.model.fit(self.data_train, self.ground_truth_train)
        # # print "loss:", hist.history["loss"][-1]
        # ground_truth_est1 = self.model.predict_proba(self.data_train)
        a_res  = np.zeros(self.n_train)
        b_res = np.zeros(self.n_train)
        ap_res = np.zeros(self.n_train)
        bp_res = np.zeros(self.n_train)
        mu_res = np.zeros((self.n_train,1))

        for i in range(self.n_train):
            a = 1.0
            b = 1.0

            a_p = 1.0
            b_p = 1.0
            for r in range(self.num_annotators):
                if self.answers[i, r] != -1:
                    if self.answers[i, r] == 1:
                        a *= self.alpha[r]
                        b *= (1 - self.beta[r])
                    elif self.answers[i, r] == 0:
                        a *= (1 - self.alpha[r])
                        b *= self.beta[r]
                    else:
                        raise Exception()

            for r in range(self.num_annotators_p):
                if self.answers_p[i, r] != -1:
                    if self.answers_p[i, r] == 1:
                        a_p *= self.alpha_p[r]
                        b_p *= (1 - self.beta_p[r])
                    elif self.answers_p[i, r] == 0:
                        a_p *= (1 - self.alpha_p[r])
                        b_p *= self.beta_p[r]
                    else:
                        raise Exception()

            a_res[i] = a
            b_res[i] = b
            ap_res[i] = a_p
            bp_res[i] = b_p

            mu = (a * a_p * self.pi[1]) / (a * a_p * self.pi[1] + b * b_p * self.pi[0])

            mu_res[i][0] = mu
            self.ground_truth_est[i, 1] = mu
            self.ground_truth_est[i, 0] = 1.0 - mu
            #print('self.ground_truth_est',self.ground_truth_est)
            self.ground_truth_train = np.argmax(self.ground_truth_est, axis=1)
        # 恶意工人答案对E步的影响
        # for i in range(self.n_train):
        # print(self.ground_truth_est)
        return self.ground_truth_est,a_res,b_res,ap_res,bp_res,mu_res

    def m_step(self, epochs=1):
        self.pi = np.sum(self.ground_truth_est, axis=0) / self.n_train
        # print('self.pi',self.pi)
        # print "M-step"

        self.alpha = self.alpha_prior * np.ones(self.num_annotators)
        self.beta = self.beta_prior * np.ones(self.num_annotators)
        self.alpha_p = self.alpha_prior * np.ones(self.num_annotators_p)
        self.beta_p = self.beta_prior * np.ones(self.num_annotators_p)
        for r in range(self.num_annotators):
            alpha_norm = self.alpha[r]
            beta_norm = self.beta[r]
            for i in range(self.n_train):
                if self.answers[i, r] != -1:
                    alpha_norm += self.ground_truth_est[i, 1]
                    beta_norm += self.ground_truth_est[i, 0]
                    if self.answers[i, r] == 1:
                        self.alpha[r] += self.ground_truth_est[i, 1]
                    elif self.answers[i, r] == 0:
                        self.beta[r] += self.ground_truth_est[i, 0]
                    else:
                        raise Exception()
            self.alpha[r] /= alpha_norm
            self.beta[r] /= beta_norm

        for r in range(self.num_annotators_p):
            alpha_norm_p = self.alpha_p[r]
            beta_norm_p = self.beta_p[r]
            for i in range(self.n_train):
                if self.answers_p[i, r] != -1:
                    alpha_norm_p += self.ground_truth_est[i, 1]
                    beta_norm_p += self.ground_truth_est[i, 0]
                    if self.answers_p[i, r] == 1:
                        self.alpha_p[r] += self.ground_truth_est[i, 1]
                    elif self.answers_p[i, r] == 0:
                        self.beta_p[r] += self.ground_truth_est[i, 0]
                    else:
                        raise Exception()
            self.alpha_p[r] /= alpha_norm_p
            self.beta_p[r] /= beta_norm_p
        return self.alpha, self.beta,self.alpha_p,self.beta_p,self.pi