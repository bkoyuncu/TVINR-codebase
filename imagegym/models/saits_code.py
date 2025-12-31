from SAITS.modeling.saits import SAITS as SAITS_code


class SAITSWrapper(SAITS_code):
    def __init__(self, **kwargs) -> None:
        super(SAITSWrapper,self).__init__(**kwargs)
        self.name = kwargs['model_type']
        self.model_type = kwargs['model_type']
        self.task = kwargs['task'] # image or pointcloud or sth
        self.missing_perc = kwargs['missing_perc']
        self.reconstruction_loss_weight = kwargs['reconstruction_loss_weight']
        self.imputation_loss_weight = kwargs['imputation_loss_weight']
        self.ORT = kwargs['ORT']
        self.MIT = kwargs['MIT']


    def print_params_count(self, logging):
        num_params = sum(p.numel() for p in self.parameters())
        logging.info('Num parameters: {}'.format(num_params))
        return num_params