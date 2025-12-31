from imagegym.config import cfg
import numpy as np
import logging

class AdjustBeta():
    def __init__(self, beta_scheduler) -> None:
        self.beta_scheduler_type = beta_scheduler[0]
        self.beta_scheduler_start = beta_scheduler[1]
        self.beta_scheduler_end = beta_scheduler[2]
        self.beta_scheduler_min = beta_scheduler[3]
        self.beta_scheduler_max = beta_scheduler[4]
        self.beta_scheduler_direction = beta_scheduler[5]  # Direction: 'min_to_max' or 'max_to_min'
        # give the summary of the beta scheduler
        logging.info(f'Beta scheduler: {self.beta_scheduler_type} from {self.beta_scheduler_min} to {self.beta_scheduler_max} from epoch {self.beta_scheduler_start} to {self.beta_scheduler_end} with direction {self.beta_scheduler_direction}')
        
    def check_update(self, epoch):
        self.current_epoch = epoch
        if self.beta_scheduler_end >= epoch >= self.beta_scheduler_start:
            if self.beta_scheduler_type is not None:
                assert cfg.dataset.missing_type in ['computed','list'] , 'Missing percentage scheduler only works with computed, list missing type'
                return self.adjust_beta(scheduler_type=self.beta_scheduler_type)  # Update beta
            else:
                return None
        else:
            return None

    def adjust_beta(self, scheduler_type):
        # Relative step from 0 to 1
        relative_step = (self.current_epoch - self.beta_scheduler_start) / (self.beta_scheduler_end - self.beta_scheduler_start)

        # Handle direction for increasing or decreasing
        if self.beta_scheduler_direction == 'min_to_max':
            base_value = self.beta_scheduler_min
            delta = self.beta_scheduler_max - self.beta_scheduler_min
        elif self.beta_scheduler_direction == 'max_to_min':
            base_value = self.beta_scheduler_max
            delta = self.beta_scheduler_min - self.beta_scheduler_max
        
        # Apply the chosen scheduler type
        if scheduler_type == 'Linear':
            new_coef = base_value + delta * relative_step
        elif scheduler_type == 'Poly':
            new_coef = base_value + delta * (relative_step ** 2)
        elif scheduler_type == 'Exp':
            new_coef = base_value + delta * (np.exp(np.log(2) * relative_step) - 1)
        else:
            raise NotImplementedError(f'Scheduler type {scheduler_type} not implemented')
        
        return new_coef



class Scenerios():
  def __init__(self, selected_scenerios, scenerios_start, scenerio_end) -> None:
    self.selected_scenerios = selected_scenerios
    self.scenerio_start = scenerios_start 
    self.scenerio_end = scenerio_end
    self.scenerio_dict = scenerio_dict
  
  def check_scenerio(self,epoch,model):
    if epoch in self.scenerio_start: # or epoch in self.scenerio_end:
      step = 1
      indexes = [self.selected_scenerios[i] for i, x in enumerate(self.scenerio_start) if x == epoch]
      for index in indexes:
        self.apply_scenerio(index,epoch,step,model)
    if epoch in self.scenerio_end: # or epoch in self.scenerio_end:
      step = 2
      indexes = [self.selected_scenerios[i] for i, x in enumerate(self.scenerio_end) if x == epoch]
      for index in indexes:
        self.apply_scenerio(index,epoch,step,model)
    else:
      pass

  def apply_scenerio(self,index,epoch,step,model):
    self.scenerio_dict[str(index)](model,step)
    print(f'Applying scenerio {index} at epoch {epoch} with step {step}')
    

#ONLY 4 is ready
def scenerio_4(model, step:int=1):
    '''
    Initialize with NF but sample from a fixed prior until step 
    '''
    if step==1:
      logging.info("params_nf_fixed true")
      model.prior_z.params_nf_fixed = True
    if step==2:
      logging.info("params_nf_fixed false")
      model.prior_z.params_nf_fixed = False
      
def scenerio_6(model, step:int=1):
  '''
  Fix cat prior in the beginning learn others, then unfreeze cat prior
  '''
  
  if step==1:
    model.fix_categorical_prior = True
    freeze_layer(model.prior_cat_encoder,freeze=True)
  if step==2:
    model.fix_categorical_prior = False
    freeze_layer(model.prior_cat_encoder,freeze=False)
    

def freeze_layer(network,freeze):
    if freeze:
            network.eval()
            for param in network.parameters():
                param.requires_grad=False
    else:
            network.train()
            for param in network.parameters():
                param.requires_grad=True

scenerio_dict = {
    '4': scenerio_4,
    '6': scenerio_6,
}
