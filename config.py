CAPTURE_LTRB_OFFSET = (9, 38, -9, -9)
#The data may be saved correctly, but I think it's being read as it's transpose. Saving uses (width, height) tensorflow uses (height, width)
INPUT_IMAGE_DIMENSIONS = (172, 80)

TRAINING_IMAGE_DIMENSIONS = INPUT_IMAGE_DIMENSIONS + tuple([3])

EPOCH = 50
BATCH_SIZE = 8
NOTHING_SKIP_RATE = 0.985
MIN_DELAY_BETWEEN_ACTIONS_MS = 100
MAX_BATCH_SIZE = 125

SCREEN_NAME = "Android Emulator - Pixel_4a_API_33:5554"

ORIGINAL_DATA_DIR = "generated/data/original"
DOWNSCALED_DATA_DIR = "generated/data/downscaled"
MODEL_OUTPUT_DIR = "generated/output/"

TRAINING_FRACTION = 0.1
TEST_DATASET = "2024-07-26-23-20-20"
PLAY_MODEL = "2024-08-06-21-58-58"

SHUFFLE_BUFFER_SIZE = 5

FPS_LOCK = 100000000000

END_TRIM_COUT = 7

ACTION_HOLD_FRAME_COUT = 3
UP_EXTRA_HOLD_TIME = 3