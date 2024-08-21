class ModelConfig:
    # Should be set automatically. Maps to file name.
    model_name : str = None
    model_architecture_name : str = None

    input_image_dimension : tuple = None
    sequence_length : int = None
    epoch : int = None
    batch_size : int = None
    testing_fraction : float = None

    train_set_name : str

    validation_sets : list[str]

    # sequence len x height x width x 3
    def get_final_input_shape(self) -> tuple:
        return tuple([self.sequence_length]) + tuple(self.input_image_dimension) + tuple([3])

class PlayerConfig:

    # Should be set automatically. Maps to file name.
    player_name = None

    model_name = None
    
    same_action_wait_frames = None
    min_vertical_confidence = None
    min_horizontal_confidence = None
    min_nothing_confidence = None

    target_fps = None

    def get_model_config() -> ModelConfig:
        return None
