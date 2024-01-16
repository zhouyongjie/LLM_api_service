import os
import logging


def get_logger(name, log_file_name='chat.log') -> logging.Logger:
    """
    Gets a standard logger with a stream hander to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(current_dir, log_file_name)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)

    # 将日志也输出到标准输出流
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    return logger
