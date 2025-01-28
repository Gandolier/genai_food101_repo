from utils.class_registry import ClassRegistry
from pytorch_fid.fid_score import calculate_fid_given_paths
import os

metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID(): # ----------------- TO DO: reconfigure so as to use images from multiple folders ---------------
    def __call__(self, orig_pth, synt_pth, fid_config):
        fid = calculate_fid_given_paths(
                    paths = [orig_pth, synt_pth], 
                    batch_size = fid_config.batch_size, 
                    device = fid_config.device, 
                    dims = fid_config.dims, 
                    num_workers = os.cpu_count()
                    )
        return fid
    
    def get_name(self):
        return "fid"

#print(FID().__call__(path1, path2, fid_config))
