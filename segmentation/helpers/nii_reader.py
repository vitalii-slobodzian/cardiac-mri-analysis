import os
import re
import nibabel as nib

from glob import glob

class NiiReader:
    regex_patterns = {
        'ED': r'ED:\s*(\d+)',
        'ES': r'ES:\s*(\d+)',
        'Group': r'Group:\s*(\w+)',
        'Height': r'Height:\s*([\d.]+)',
        'NbFrame': r'NbFrame:\s*(\d+)',
        'Weight': r'Weight:\s*([\d.]+)'
    }

    def read_file(self, file_path):
        nii_file = nib.load(file_path)
        slices = nii_file.get_fdata()
        slice_count = slices.shape[2]

        return [slices[:, :, i] for i in range(slice_count)]


    def read_folder(self, folder_path, glob_pattern='**/*.nii.gz', recursive=True):
        files_path = glob(os.path.join(folder_path, glob_pattern), recursive=recursive)
        file_slices = []
        file_names = []
        configs = []

        for file_path in files_path:
            slices = self.read_file(file_path)
            file_slices.extend(slices)
            basename = os.path.basename(file_path)
            file_name = basename.split('.')[0]
            file_names.extend([f'{file_name}_{i}' for i in range(len(slices))])

            config = self.parse_config(file_path.replace(basename, 'Info.cfg'))
            configs.extend([config] * len(slices))

        return file_slices, file_names, configs
    
    def parse_config(self, config_path):
        parsed_data = {}

        with open(config_path, 'r') as file:
            content = file.read()
        
            for key, pattern in self.regex_patterns.items():
                match = re.search(pattern, content)
                if match:
                    parsed_data[key] = match.group(1)
                else:
                    parsed_data[key] = None
        return parsed_data
