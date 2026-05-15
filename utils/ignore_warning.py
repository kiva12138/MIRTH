import warnings, logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

def ignore_warnings():
    warnings.filterwarnings("ignore", message=r"The 'repr' attribute *", category=UserWarning, module=r"pydantic\._internal\._generate_schema",)
    warnings.filterwarnings("ignore", message=r"The 'frozen' attribute *", category=UserWarning, module=r"pydantic\._internal\._generate_schema",)
    warnings.filterwarnings("ignore", message=r"FutureWarning: In the future *")
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)