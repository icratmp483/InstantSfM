import importlib

# general options that do not vary with feature_name
GENERAL_OPTIONS = {
    # used in global mapper
    'skip_preprocessing': False,
    'skip_view_graph_calibration': False,
    'skip_relative_pose_estimation': False,
    'skip_rotation_averaging': False,
    'skip_track_establishment': False,
    'skip_global_positioning': False,
    'skip_bundle_adjustment': False,
    'num_iteration_bundle_adjustment': 3,
    'skip_retriangulation': True,
    'num_iteration_retriangulation': 1,
    'skip_pruning': True,
    # used in feature handling
    'uniform_camera': True, 
}

class Config:
    def __init__(self, feature_name, manual_config_name=None, image_num=100):
        self.feature_name = feature_name

        # config for large(>50 images) and small(<=50 images) datasets
        config_module_names = {
            'colmap': 'instantsfm.config.colmap',
        }
        config_modules = {name: path for name, path in config_module_names.items()}
        
        if manual_config_name is not None:
            config_module_name = 'instantsfm.config.' + manual_config_name
        elif feature_name in config_module_names:
            config_module_name = config_modules[feature_name]
        else:
            raise ValueError('Invalid feature_name')
        
        config_module = importlib.import_module(config_module_name)
        CONFIG = getattr(config_module, 'CONFIG')

        self.OPTIONS = GENERAL_OPTIONS
        self.VIEW_GRAPH_CALIBRATOR_OPTIONS = CONFIG['VIEW_GRAPH_CALIBRATOR_OPTIONS']
        self.INLIER_THRESHOLD_OPTIONS = CONFIG['INLIER_THRESHOLD_OPTIONS']
        self.ROTATION_ESTIMATOR_OPTIONS = CONFIG['ROTATION_ESTIMATOR_OPTIONS']
        self.L1_SOLVER_OPTIONS = CONFIG['L1_SOLVER_OPTIONS']
        self.TRACK_ESTABLISHMENT_OPTIONS = CONFIG['TRACK_ESTABLISHMENT_OPTIONS']
        self.GLOBAL_POSITIONER_OPTIONS = CONFIG['GLOBAL_POSITIONER_OPTIONS']
        self.BUNDLE_ADJUSTER_OPTIONS = CONFIG['BUNDLE_ADJUSTER_OPTIONS']
        self.TRIANGULATOR_OPTIONS = CONFIG['TRIANGULATOR_OPTIONS']
        self.FEATURE_HANDLER_OPTIONS = CONFIG['FEATURE_HANDLER_OPTIONS']

        # by default ENABLE_GUI is disabled
        self.ENABLE_GUI = False