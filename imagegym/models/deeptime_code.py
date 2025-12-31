from DeepTime.models.DeepTIMe import DeepTIMe as DeepTIMe_code


class DeepTimeWrapper(DeepTIMe_code):
    def __init__(self, **kwargs) -> None:
        super(DeepTimeWrapper,self).__init__(**kwargs)
        self.name = kwargs['model_type']
        self.model_type = kwargs['model_type']
        self.task = kwargs['task'] # image or pointcloud or sth
        self.missing_perc = kwargs['missing_perc']

    def print_params_count(self, logging):
        num_params = sum(p.numel() for p in self.parameters())
        logging.info('Num parameters: {}'.format(num_params))
        return num_params