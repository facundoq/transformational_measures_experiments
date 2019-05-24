

class VarianceExperimentResult:
    def __init__(self, model_name, dataset_name, activation_names, class_names, transformations, options, rotated_measures, unrotated_measures):
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.activation_names=activation_names
        self.class_names=class_names
        self.transformations=transformations
        self.options=options
        self.rotated_measures=rotated_measures
        self.unrotated_measures=unrotated_measures

