## ActionAbstractor class
## use the action abstraction to abstract the action space
## Use the class from simple_rl.abstraction.action_abs.ActionAbstractionClass

from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction

class ActionAbstractor(ActionAbstraction):

    def __init__(self, options=None, prim_actions=..., term_prob=0, prims_on_failure=False):
        super().__init__(options, prim_actions, term_prob, prims_on_failure)