import os
import time

class Logger:
    def __init__(self, log_dir, file_name="log.txt"):
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, file_name)
        with open(self.file_path, "w") as f:
            f.write("Log started at {}\n".format(time.ctime()))
    
    def log(self, message):
        timestamp = time.ctime()
        msg = "[{}] {}\n".format(timestamp, message)
        with open(self.file_path, "a") as f:
            f.write(msg)
        print(msg, end="")

if __name__ == "__main__":
    logger = Logger("logs_test")
    logger.log("This is a test message.")
