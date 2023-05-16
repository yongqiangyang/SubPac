import numpy as np

class EM():

    def __init__(self, answers, answers_p, batch_size=16, alpha_prior=1.0, beta_prior=1.0):
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
        self.ground_truth_est_normal = np.zeros((self.n_train, 2))
        self.final_weights_normal = np.zeros((self.n_train, 2))


        for i in range(self.n_train):
            votes = np.zeros(self.num_classes)
            for r in range(self.num_annotators):
                if answers[i, r] != -1:
                    votes[self.answers[i, r]] += 1
            if np.sum(votes) == 0:
                self.ground_truth_est_normal[i, 0] = 0.5
                self.ground_truth_est_normal[i, 1] = 0.5
            else:
                self.ground_truth_est_normal[i, np.argmax(votes)] = np.max(votes)/np.sum(votes)
                self.ground_truth_est_normal[i, 1-np.argmax(votes)] = 1 - self.ground_truth_est_normal[i, np.argmax(votes)]
            for r in range(self.num_annotators_p):
                if answers_p[i, r] != -1:
                    votes[self.answers_p[i, r]] += 1
            if np.sum(votes) == 0:
                self.ground_truth_est[i, 0] = 0.5
                self.ground_truth_est[i, 1] = 0.5
            else:
                self.ground_truth_est[i, np.argmax(votes)] = np.max(votes)/np.sum(votes)
                self.ground_truth_est[i, 1-np.argmax(votes)] = 1-self.ground_truth_est[i, np.argmax(votes)]

        self.pi = np.sum(self.ground_truth_est, axis=0) / self.n_train
        self.pi_normal = np.sum(self.ground_truth_est_normal, axis=0) / self.n_train
        
    def e_step(self):
        a_res  = np.zeros(self.n_train)
        b_res = np.zeros(self.n_train)
        ap_res = np.zeros(self.n_train)
        bp_res = np.zeros(self.n_train)
        mu_res = np.zeros(self.n_train)

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

            final_weights_normal1 = a * self.pi_normal[1]
            final_weights_normal0 = b * self.pi_normal[0]
            
            mu_res[i] = mu
            self.ground_truth_est[i, 1] = mu
            self.ground_truth_est[i, 0] = 1.0 - mu
            
            if final_weights_normal1 == 1:
                final_weights_normal1 = 0.999
            if final_weights_normal1 == 0:
                final_weights_normal1 = 0.001
            if final_weights_normal0 == 1:
                final_weights_normal0 = 0.999
            if final_weights_normal0 == 0:
                final_weights_normal0 = 0.001
            self.final_weights_normal[i, 1] = np.log(final_weights_normal1)
            self.final_weights_normal[i, 0] = np.log(final_weights_normal0)
            
        return self.final_weights_normal

    def m_step(self, epochs=1):
        self.pi = np.sum(self.ground_truth_est, axis=0) / self.n_train

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
        
        self.weight_p = np.random.uniform(1, 2, (self.n_train, self.num_annotators_p, 4))
        for i in range(self.n_train):
            for j in range(self.num_annotators_p):
                self.weight_p[i][j][0] = np.log(self.beta_p[j])
                if self.alpha_p[j] == 1:
                    self.alpha_p[j] = 0.999
                if self.beta_p[j] == 1:
                    self.beta_p[j] = 0.999
                self.weight_p[i][j][1] = np.log(1 - self.alpha_p[j])
                self.weight_p[i][j][2] = np.log(1 - self.beta_p[j])
                self.weight_p[i][j][3] = np.log(self.alpha_p[j])
        
        for i in range(self.pi.shape[0]):
            self.pi[i] = np.log(self.pi[i])

        return self.weight_p, self.pi





