import os


class CleanupManager:
    def __init__(self, config):
        self.config = config

    def cleanup_previous_run(self):
        """
        Clears all temporary generation artifacts from phase 1 and phase 2.
        Does NOT touch the permanent 'saved' directory.
        """
        print("Cleaning up previous run artifacts...")
        for directory in [self.config.phase1_dir, self.config.phase2_dir, self.config.temp_dir]:
            if not os.path.exists(directory):
                continue
            for f in os.listdir(directory):
                path = os.path.join(directory, f)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    print(f"Error deleting {path}: {e}")

    def enforce_disk_limit(self):
        """
        Calculates disk usage of output directories (excluding permanent saved files)
        and deletes oldest files if it exceeds the configured maximum.
        """
        total_size = 0
        directories_to_check = [self.config.phase1_dir, self.config.phase2_dir, self.config.temp_dir]

        for directory in directories_to_check:
            if not os.path.exists(directory):
                continue
            for root, _, files in os.walk(directory):
                for f in files:
                    path = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(path)
                    except FileNotFoundError:
                        pass

        gb = total_size / (1024 ** 3)

        if gb > self.config.max_disk_usage_gb:
            self.cleanup_oldest(directories_to_check)

    def cleanup_oldest(self, directories):
        files = []

        for directory in directories:
            if not os.path.exists(directory):
                continue
            for root, _, fs in os.walk(directory):
                for f in fs:
                    path = os.path.join(root, f)
                    try:
                        files.append((os.path.getmtime(path), path))
                    except FileNotFoundError:
                        pass

        files.sort()

        # Delete the 5 oldest files
        for _, path in files[:5]:
            try:
                os.remove(path)
                print(f"Deleted old temporary file: {path}")
            except Exception as e:
                print(f"Error cleaning {path}: {e}")