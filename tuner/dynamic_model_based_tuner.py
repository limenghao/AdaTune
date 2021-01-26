from .model_based_tuner import knob2point,point2knob,submodular_pick
from .tuner import Tuner
from ..util import sample_ints

import numpy as np


class ModelBasedTunerAda(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use an optimizer to
    find the maximums of the cost model as next trials

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    cost_model: CostModel
        The cost model that predicts the speed of a config (IR)
    model_optimizer:
        The optimizer to find local optimum points of cost model in tuning search space
    plan_size: int
        Tuner will re-fit model per `plan_size` new measure samples
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick plan_size of them according to the diversity metric.
    """

    def __init__(self, task, cost_model, model_optimizer, plan_size, diversity_filter_ratio=None, dynamic_ep=True):
        super(ModelBasedTunerAda, self).__init__(task)

        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size
        self.space = task.config_space
        self.space_len = len(task.config_space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, "Diversity filter ratio " \
                                                     "must be larger than one"

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0
        self.dynamic_ep = dynamic_ep
        self.balance_ep = 0.05

    def next_batch(self, batch_size):
        ret = []

        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1
            
            if self.trial_pt >= len(self.trials) - int(self.balance_ep * self.plan_size):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = np.random.randint(len(self.space))
                while index in self.visited:
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))
            self.visited.add(index)

            counter += 1
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
                # self.count_not0 = self.count_not0 + 1
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        # if we have enough new training samples
        if len(self.xs) >= self.plan_size * (self.train_ct + 1) \
                and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size * self.diversity_filter_ratio, self.visited)
                scores = self.cost_model.predict(candidate)
                knobs = [point2knob(x, self.dims) for x in candidate]
                pick_index = submodular_pick(0 * scores, knobs, self.plan_size, knob_weight=1)
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited)
                if self.dynamic_ep:
                    samples = np.array(sample_ints(0, len(self.space), 20))
                    _, mean_of_variance = self.cost_model._expected_imporvement(samples)
                    
                    self.balance_ep = mean_of_variance/self.best_flops

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def load_history(self, data_set):
        # set in_tuning as True to make the feature extraction consistent
        GLOBAL_SCOPE.in_tuning = True

        # fit base model
        base_model = self.cost_model.spawn_base_model()
        success = base_model.fit_log(data_set, self.plan_size)

        if not success:
            GLOBAL_SCOPE.in_tuning = False
            return

        # use base model to select initial points
        if not self.trials:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
            self.trials = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)
        GLOBAL_SCOPE.in_tuning = False

    def has_next(self):
        return len(self.visited) < len(self.space)